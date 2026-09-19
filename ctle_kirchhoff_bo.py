#!/usr/bin/env python
"""Bayesian optimisation for a compact, adjoint-trained KirchhoffNet CTLE student.

The circuit RHS follows ``kirchhoffnet/bare/circuit.py``: a directed graph of
two-terminal nonlinear devices accumulates edge currents at each node under
Kirchhoff's current law.  The state is evolved with
``torchdiffeq.odeint_adjoint(..., method='dopri5')``.  Heun is never used.

The canonical Phase-A .npz supplies fixed CTLE examples/splits.  For
schema-2 datasets, ``mlp_logits_trial0019`` supplies frozen MLP-teacher logits
for distillation.  The script deliberately searches only compact circuit ODE
and optimiser choices, not MLP architectures.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint

try:
    import optuna
except ImportError as exc:
    raise SystemExit("Optuna is required: pip install optuna") from exc


# Same CTLE parameter order and physical log bounds as ctle_dagger_common.py.
PARAM_LOG_BOUNDS = np.asarray([
    [-12., -8.], [-7., -2.], [-12., -7.], [1., 6.], [-15., -9.], [0., 6.], [-1., 1.],
], dtype=np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def target_logits(params: np.ndarray) -> np.ndarray:
    logs = np.log10(np.clip(params, 1e-30, None))
    lo, hi = PARAM_LOG_BOUNDS[:, 0], PARAM_LOG_BOUNDS[:, 1]
    p = np.clip((logs - lo) / (hi - lo), 1e-5, 1. - 1e-5)
    return np.log(p / (1. - p)).astype(np.float32)


def bounded_logs(logits: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(PARAM_LOG_BOUNDS, dtype=logits.dtype, device=logits.device)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.sigmoid(logits)


class KirchhoffCircuit(nn.Module):
    """Bare-repo KCL RHS with the constrained KNet rail discipline.

    Edges read the bounded broadcast ``x_max*tanh(x/x_max)``.  The integrated
    state remains continuous, while a positive leak and differentiable
    soft-rail current return it from excursions beyond the rail.  This is the
    same smooth constraint pattern used by ``DifferentialStage`` and remains
    suitable for the adjoint solver (unlike an in-RHS hard clamp).
    """
    def __init__(self, src: torch.Tensor, dst: torch.Tensor, *, num_nodes: int, activation: str, x_max: float,
                 clip_current: float, clip_softness: float, leak_init: float,
                 c_eff: float) -> None:
        super().__init__()
        self.register_buffer("src", src.long())
        self.register_buffer("dst", dst.long())
        # Keep the requested state width even if the sampled sparse graph does
        # not happen to touch its highest-index node.
        self.num_nodes = int(num_nodes)  # excludes ground node 0
        self.edge = nn.Parameter(torch.empty(3, len(src)))
        nn.init.normal_(self.edge, std=0.08)
        self.activation = {"tanh": torch.tanh, "relu": F.relu}[activation]
        # Positive programmable leakage, initialized to the constrained KNet
        # default (0.0486). It is counted in the architecture budget.
        raw_leak = float(np.log(np.expm1(leak_init)))
        self.raw_leak = nn.Parameter(torch.full((self.num_nodes,), raw_leak))
        self.x_max, self.clip_current = float(x_max), float(clip_current)
        self.clip_softness, self.c_eff = float(clip_softness), float(c_eff)

    def forward(self, _t: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        # Ground is fixed to zero. Device activation is selected per BO trial.
        bounded = self.x_max * torch.tanh(voltage / self.x_max)
        with_ground = torch.cat((torch.zeros_like(bounded[:, :1]), bounded), dim=1)
        current = self.activation(self.edge[0] * with_ground[:, self.src]
                                  + self.edge[1] * with_ground[:, self.dst] + self.edge[2])
        derivative = torch.zeros_like(with_ground)
        derivative.scatter_add_(1, self.src.expand_as(current), -current)
        derivative.scatter_add_(1, self.dst.expand_as(current), current)
        soft_rail = self.clip_current * (
            torch.sigmoid((voltage - self.x_max) / self.clip_softness)
            - torch.sigmoid((-voltage - self.x_max) / self.clip_softness)
        )
        leak = F.softplus(self.raw_leak).unsqueeze(0) * voltage
        return (derivative[:, 1:] - leak - soft_rail) / self.c_eff


class CTLEKirchhoffNet(nn.Module):
    """4-spec encoder -> Kirchhoff ODE -> 7 CTLE logits."""
    def __init__(self, nodes: int, edge_count: int, activation: str, topology_seed: int, t_end: float,
                 rtol: float, *, x_max: float, clip_current: float,
                 clip_softness: float, leak_init: float, c_eff: float) -> None:
        super().__init__()
        src, dst = make_topology(nodes, edge_count, topology_seed)
        self.encoder = nn.Linear(4, nodes)
        self.circuit = KirchhoffCircuit(src, dst, num_nodes=nodes, activation=activation, x_max=x_max, clip_current=clip_current,
                                        clip_softness=clip_softness, leak_init=leak_init, c_eff=c_eff)
        self.readout = nn.Linear(nodes, 7)
        self.t_end, self.rtol = float(t_end), float(rtol)
        self.x_max = float(x_max)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        initial_voltage = self.x_max * torch.tanh(self.encoder(specs) / self.x_max)
        times = torch.tensor((0., self.t_end), device=specs.device, dtype=specs.dtype)
        # Explicitly adjoint + adaptive Dormand-Prince. No fixed-step/Heun path.
        final_voltage = odeint_adjoint(self.circuit, initial_voltage, times,
                                       rtol=self.rtol, atol=self.rtol * .1, method="dopri5")[-1]
        return self.readout(self.x_max * torch.tanh(final_voltage / self.x_max))


def make_topology(nodes: int, edge_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique directed edges, including ground connections, in original node numbering."""
    pairs = [(s, d) for s in range(nodes + 1) for d in range(1, nodes + 1) if s != d]
    if edge_count > len(pairs): raise ValueError("edge_count exceeds directed graph capacity")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pairs), edge_count, replace=False)
    return (torch.tensor([pairs[i][0] for i in selected]),
            torch.tensor([pairs[i][1] for i in selected]))


def parameter_count(nodes: int, edges: int) -> int:
    # encoder (4N+N bias), three/device, programmable per-node leak, readout.
    return 13 * nodes + 3 * edges + 7


def circuit_candidates(budget: int, tolerance: float) -> list[tuple[int, int]]:
    result = []
    for nodes in range(16, 65):
        max_edges = nodes * (nodes + 1)
        target_edges = round((budget - 12 * nodes - 7) / 3)
        # Nearby counts make edge density a genuine BO-controlled architecture choice.
        for edges in range(max(1, target_edges - 120), min(max_edges, target_edges + 120) + 1, 30):
            count = parameter_count(nodes, edges)
            if abs(count - budget) <= budget * tolerance:
                result.append((nodes, edges))
    if not result: raise ValueError("No directed Kirchhoff topology meets --param-budget/tolerance")
    return result


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        needed = {"specs", "params", "train_idx", "val_idx", "test_idx", "input_log_min", "input_log_max"}
        if missing := needed - set(z.files): raise ValueError(f"Canonical dataset missing {sorted(missing)}")
        specs, params = np.asarray(z["specs"], np.float32), np.asarray(z["params"], np.float32)
        lo, hi = np.asarray(z["input_log_min"], np.float32), np.asarray(z["input_log_max"], np.float32)
        x = np.clip(2 * (np.log10(np.clip(specs, 1e-30, None)) - lo) / (hi - lo) - 1, -4, 4).astype(np.float32)
        teacher = np.asarray(z["mlp_logits_trial0019"], np.float32) if "mlp_logits_trial0019" in z.files else None
        power_norm = float(np.asarray(z["power_norm_const"]).item()) if "power_norm_const" in z.files else float(np.mean(4 * params[:, 6] * params[:, 1]))
        return {"x": x, "y": target_logits(params), "teacher": teacher, "power_norm": power_norm,
                "train": np.asarray(z["train_idx"], np.int64), "val": np.asarray(z["val_idx"], np.int64),
                "test": np.asarray(z["test_idx"], np.int64), "fingerprint": str(np.asarray(z["fingerprint"]).item())}


def loss_fn(logits: torch.Tensor, labels: torch.Tensor, teacher: torch.Tensor | None,
            args: argparse.Namespace, power_norm: float, validity_scale: float = 1.) -> torch.Tensor:
    flow_loss = F.smooth_l1_loss(logits, labels)
    loss = flow_loss * (1. + args.ctle_phase_a_fwd_weight)
    if teacher is not None: loss = loss + args.ctle_phase_a_mlp_weight * F.mse_loss(logits, teacher)
    logs = bounded_logs(logits)
    power = 4 * torch.pow(10., logs[:, 6]) * torch.pow(10., logs[:, 1])
    loss = loss + args.ctle_phase_a_power_weight * power.mean() / power_norm
    # A ZIG model is intentionally not imported into this compact ODE trainer;
    # this is an explicit rail-confidence regularizer, not claimed ZIG validity.
    loss = loss + args.ctle_phase_a_validity_weight * validity_scale * torch.relu(logits.abs() - 8).square().mean()
    return loss


def evaluate(model: nn.Module, data: dict[str, Any], split: str, device: torch.device, args: argparse.Namespace) -> float:
    model.eval(); idx = data[split]
    with torch.no_grad():
        x, y = torch.from_numpy(data["x"][idx]).to(device), torch.from_numpy(data["y"][idx]).to(device)
        teacher = torch.from_numpy(data["teacher"][idx]).to(device) if data["teacher"] is not None else None
        return float(loss_fn(model(x), y, teacher, args, data["power_norm"]).item())


def trial_train(trial: optuna.Trial, data: dict[str, Any], args: argparse.Namespace, device: torch.device,
                candidates: list[tuple[int, int]], output: Path) -> float:
    nodes, edges = candidates[trial.suggest_int("circuit_architecture", 0, len(candidates) - 1)]
    activation = trial.suggest_categorical("device_activation", ["tanh", "relu"])
    t_end = trial.suggest_categorical("t_end", [.25, .5, 1.])
    rtol = trial.suggest_categorical("rtol", [1e-3, 3e-4])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    model = CTLEKirchhoffNet(nodes, edges, activation, args.seed + trial.number, t_end, rtol,
                             x_max=args.x_max, clip_current=args.clip_current,
                             clip_softness=args.clip_softness, leak_init=args.leak_init,
                             c_eff=args.c_eff).to(device)
    trial.set_user_attr("nodes", nodes); trial.set_user_attr("edges", edges)
    trial.set_user_attr("parameter_count", parameter_count(nodes, edges))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    idx = data["train"]
    tensors = [torch.from_numpy(data["x"][idx]), torch.from_numpy(data["y"][idx])]
    if data["teacher"] is not None: tensors.append(torch.from_numpy(data["teacher"][idx]))
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True)
    best, state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        ramp = min(1., max(0., (epoch + 2 - args.ctle_phase_a_validity_ramp_start) / max(1, args.ctle_phase_a_validity_ramp_epochs)))
        for batch in loader:
            opt.zero_grad(set_to_none=True); x, y = batch[0].to(device), batch[1].to(device)
            teacher = batch[2].to(device) if len(batch) == 3 else None
            loss_fn(model(x), y, teacher, args, data["power_norm"], ramp).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        value = evaluate(model, data, "val", device, args)
        if value < best:
            best, state = value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        trial.report(value, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
    torch.save({"state_dict": state, "nodes": nodes, "edges": edges, "activation": activation, "t_end": t_end,
                "rtol": rtol, "x_max": args.x_max, "clip_current": args.clip_current,
                "clip_softness": args.clip_softness, "leak_init": args.leak_init,
                "c_eff": args.c_eff, "validation_loss": best,
                "parameter_count": parameter_count(nodes, edges)},
               output / f"trial_{trial.number:04d}.pt")
    return best


def run(args: argparse.Namespace) -> None:
    if args.dataset != "ctle" or not args.ctle_phase_a: raise ValueError("Use --dataset ctle --ctle-phase-a")
    if args.ctle_canonical_dataset is None or args.output is None or args.param_budget is None: raise ValueError("canonical dataset, output, and parameter budget are required")
    if args.device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    seed_everything(args.seed); device = torch.device(args.device); data = load_dataset(args.ctle_canonical_dataset)
    if args.ctle_phase_a_mlp_weight > 0 and data["teacher"] is None: raise ValueError("Teacher distillation needs schema-2 data with mlp_logits_trial0019")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); candidates = circuit_candidates(args.param_budget, args.param_tolerance)
    (output / "run_config.json").write_text(json.dumps({**vars(args), "fingerprint": data["fingerprint"], "candidate_count": len(candidates), "solver": "odeint_adjoint/dopri5", "device_activation_choices": ["tanh", "relu"], "node_constraint": "bounded tanh broadcast + soft rail + positive leak"}, indent=2, default=str))
    storage = f"sqlite:///{(output / 'study.db').as_posix()}"
    study = optuna.create_study(study_name="ctle_kirchhoff_bo", storage=storage, load_if_exists=args.resume,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, args.n_trials), n_warmup_steps=max(1, args.epochs // 5)))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(lambda t: trial_train(t, data, args, device, candidates, output), n_trials=max(0, args.n_trials - done))
    best = study.best_trial; ckpt = output / f"trial_{best.number:04d}.pt"; payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = CTLEKirchhoffNet(payload["nodes"], payload["edges"], payload["activation"], args.seed + best.number,
                             payload["t_end"], payload["rtol"], x_max=payload["x_max"],
                             clip_current=payload["clip_current"], clip_softness=payload["clip_softness"],
                             leak_init=payload["leak_init"], c_eff=payload["c_eff"]).to(device)
    model.load_state_dict(payload["state_dict"]); summary = {"best_trial": best.number, "validation_loss": best.value,
        "test_loss": evaluate(model, data, "test", device, args), "parameter_count": payload["parameter_count"], "params": best.params}
    shutil.copy2(ckpt, output / "best_model.pt"); (output / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
    engine = getattr(getattr(study._storage, "_backend", study._storage), "engine", None)
    if engine is not None: engine.dispose()


def smoke_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ctle_kirchhoff_bo_") as temp:
        root = Path(temp); rng = np.random.default_rng(7); specs = 10 ** rng.uniform(-2, 2, (32, 4)).astype(np.float32)
        params = 10 ** rng.uniform(PARAM_LOG_BOUNDS[:, 0], PARAM_LOG_BOUNDS[:, 1], (32, 7)).astype(np.float32); logs = np.log10(specs)
        np.savez_compressed(root / "data.npz", specs=specs, params=params, train_idx=np.arange(20), val_idx=np.arange(20, 26), test_idx=np.arange(26, 32), input_log_min=logs.min(0)-.1, input_log_max=logs.max(0)+.1, fingerprint=np.array("smoke"))
        run(parse_args(["--dataset", "ctle", "--ctle-phase-a", "--ctle-canonical-dataset", str(root / "data.npz"), "--param-budget", "300", "--param-tolerance", ".4", "--epochs", "1", "--n-trials", "1", "--batch-size", "10", "--device", "cpu", "--ctle-phase-a-mlp-weight", "0", "--output", str(root / "out")]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="ctle", choices=["ctle"]); p.add_argument("--ctle-phase-a", action="store_true")
    p.add_argument("--ctle-canonical-dataset", type=Path); p.add_argument("--ctle-phase-a-mlp-teacher-ckpt", type=Path, help="Recorded with run provenance; teacher logits are frozen in schema-2 canonical data.")
    p.add_argument("--ctle-phase-a-mlp-weight", type=float, default=1.); p.add_argument("--ctle-phase-a-fwd-weight", type=float, default=0.)
    p.add_argument("--ctle-phase-a-power-weight", type=float, default=0.); p.add_argument("--ctle-phase-a-validity-weight", type=float, default=0.)
    p.add_argument("--ctle-phase-a-validity-ramp-start", type=int, default=10); p.add_argument("--ctle-phase-a-validity-ramp-epochs", type=int, default=30)
    p.add_argument("--ctle-objective", choices=["validation"], default="validation"); p.add_argument("--param-budget", type=int); p.add_argument("--param-tolerance", type=float, default=.15)
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--n-trials", type=int, default=30); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--seed", type=int, default=100)
    p.add_argument("--x-max", type=float, default=3.0, help="Fixed differential node-voltage rail.")
    p.add_argument("--clip-current", type=float, default=0.05, help="Fixed soft-rail restoring-current magnitude.")
    p.add_argument("--clip-softness", type=float, default=0.02, help="Fixed soft-rail transition width.")
    p.add_argument("--leak-init", type=float, default=0.0486, help="Initial positive programmable node leak.")
    p.add_argument("--c-eff", type=float, default=1.0, help="Fixed effective node capacitance.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output", type=Path); p.add_argument("--resume", action="store_true"); p.add_argument("--smoke-test", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); smoke_test() if args.smoke_test else run(args)
