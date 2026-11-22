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
import csv
import json
import logging
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint

from ctle_metrics_config import DEGRADE_REL_THRESHOLD

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
                 c_eff: float, bounded_edge_parameters: bool = False,
                 edge_weight_max: float = 1.0, edge_bias_max: float = 1.0,
                 fixed_leak: bool = False) -> None:
        super().__init__()
        self.register_buffer("src", src.long())
        self.register_buffer("dst", dst.long())
        # Keep the requested state width even if the sampled sparse graph does
        # not happen to touch its highest-index node.
        self.num_nodes = int(num_nodes)  # excludes ground node 0
        self.edge = nn.Parameter(torch.empty(3, len(src)))
        nn.init.normal_(self.edge, std=0.08)
        self.activation = {"tanh": torch.tanh, "relu": F.relu}[activation]
        # ``edge`` remains the three raw per-edge degrees of freedom.  With
        # --bounded-edge-parameters they are mapped into finite physical-like
        # slope/bias ranges before being used by the device.
        self.bounded_edge_parameters = bool(bounded_edge_parameters)
        self.edge_weight_max = float(edge_weight_max)
        self.edge_bias_max = float(edge_bias_max)
        if self.edge_weight_max <= 0 or self.edge_bias_max <= 0:
            raise ValueError("edge parameter bounds must be positive")
        self.fixed_leak_enabled = bool(fixed_leak)
        if self.fixed_leak_enabled:
            self.register_buffer("fixed_leak", torch.full((self.num_nodes,), float(leak_init)))
            self.raw_leak = None
        else:
            # Positive programmable leakage, initialized to the constrained
            # KNet default. It is counted in the architecture budget.
            raw_leak = float(np.log(np.expm1(leak_init)))
            self.raw_leak = nn.Parameter(torch.full((self.num_nodes,), raw_leak))
        self.x_max, self.clip_current = float(x_max), float(clip_current)
        self.clip_softness, self.c_eff = float(clip_softness), float(c_eff)
        self.nfe = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def forward(self, _t: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        # Ground is fixed to zero. Device activation is selected per BO trial.
        bounded = self.x_max * torch.tanh(voltage / self.x_max)
        with_ground = torch.cat((torch.zeros_like(bounded[:, :1]), bounded), dim=1)
        if self.bounded_edge_parameters:
            edge_src = self.edge_weight_max * torch.tanh(self.edge[0])
            edge_dst = self.edge_weight_max * torch.tanh(self.edge[1])
            edge_bias = self.edge_bias_max * torch.tanh(self.edge[2])
        else:
            edge_src, edge_dst, edge_bias = self.edge[0], self.edge[1], self.edge[2]
        current = self.activation(edge_src * with_ground[:, self.src]
                                  + edge_dst * with_ground[:, self.dst] + edge_bias)
        derivative = torch.zeros_like(with_ground)
        derivative.scatter_add_(1, self.src.expand_as(current), -current)
        derivative.scatter_add_(1, self.dst.expand_as(current), current)
        soft_rail = self.clip_current * (
            torch.sigmoid((voltage - self.x_max) / self.clip_softness)
            - torch.sigmoid((-voltage - self.x_max) / self.clip_softness)
        )
        leak_value = self.fixed_leak if self.fixed_leak_enabled else F.softplus(self.raw_leak)
        leak = leak_value.unsqueeze(0) * voltage
        return (derivative[:, 1:] - leak - soft_rail) / self.c_eff


class FixedSparseLinear(nn.Module):
    """A fixed wiring-pattern affine map with only its live connections stored.

    It is deliberately not a masked dense matrix: inactive entries do not
    exist as trainable parameters and therefore do not consume the BO budget.
    """
    def __init__(self, in_features: int, out_features: int, fan_in: int, seed: int) -> None:
        super().__init__()
        if not 1 <= fan_in <= in_features:
            raise ValueError(f"fan_in must lie in [1, {in_features}], got {fan_in}")
        rng = np.random.default_rng(seed)
        indices = np.stack([rng.choice(in_features, fan_in, replace=False) for _ in range(out_features)])
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(out_features, fan_in))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5.0))
        bound = 1.0 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, self.indices] * self.weight.unsqueeze(0)).sum(dim=-1) + self.bias


class CTLEKirchhoffNet(nn.Module):
    """4-spec encoder -> Kirchhoff ODE -> 7 CTLE logits."""
    def __init__(self, nodes: int, edge_count: int, activation: str, topology_seed: int, t_end: float,
                 rtol: float, *, x_max: float, clip_current: float,
                 clip_softness: float, leak_init: float, c_eff: float,
                 bounded_edge_parameters: bool = False, edge_weight_max: float = 1.0,
                 edge_bias_max: float = 1.0, fixed_leak: bool = False,
                 input_mapper_fan_in: int = 0, output_mapper_fan_in: int = 0) -> None:
        super().__init__()
        src, dst = make_topology(nodes, edge_count, topology_seed)
        # Names intentionally match the existing Phase-A differential-LR
        # grouping and OutputAffine-bias initialisation conventions.
        self.input_mapper = (FixedSparseLinear(4, nodes, min(input_mapper_fan_in, 4), topology_seed + 101)
                             if input_mapper_fan_in > 0 else nn.Linear(4, nodes))
        self.circuit = KirchhoffCircuit(src, dst, num_nodes=nodes, activation=activation, x_max=x_max, clip_current=clip_current,
                                        clip_softness=clip_softness, leak_init=leak_init, c_eff=c_eff,
                                        bounded_edge_parameters=bounded_edge_parameters,
                                        edge_weight_max=edge_weight_max, edge_bias_max=edge_bias_max,
                                        fixed_leak=fixed_leak)
        self.output_mapper = (FixedSparseLinear(nodes, 7, min(output_mapper_fan_in, nodes), topology_seed + 211)
                              if output_mapper_fan_in > 0 else nn.Linear(nodes, 7))
        self.t_end, self.rtol = float(t_end), float(rtol)
        self.x_max = float(x_max)
        lo = torch.as_tensor(PARAM_LOG_BOUNDS[:, 0], dtype=torch.float32)
        hi = torch.as_tensor(PARAM_LOG_BOUNDS[:, 1], dtype=torch.float32)
        self.register_buffer("log_lo", lo)
        self.register_buffer("log_hi", hi)
        self.input_log_min: torch.Tensor | None = None
        self.input_log_max: torch.Tensor | None = None

    def attach_scaler(self, *, input_log_min, input_log_max, **_unused: Any) -> None:
        """Use the canonical Phase-A four-log-feature input contract."""
        if input_log_min is None or input_log_max is None:
            raise ValueError("Canonical Phase-A input_log_min/max are required")
        self.input_log_min = torch.as_tensor(input_log_min, dtype=torch.float32)
        self.input_log_max = torch.as_tensor(input_log_max, dtype=torch.float32)

    def scale_input(self, specs: torch.Tensor) -> torch.Tensor:
        if self.input_log_min is None or self.input_log_max is None:
            raise RuntimeError("Call attach_scaler before the first forward pass")
        lo = self.input_log_min.to(specs.device, specs.dtype)
        hi = self.input_log_max.to(specs.device, specs.dtype)
        logs = torch.log10(specs.clamp(min=1e-12))
        return (2.0 * (logs - lo) / (hi - lo).clamp(min=1e-8) - 1.0).clamp(-4.0, 4.0)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        initial_voltage = self.x_max * torch.tanh(self.input_mapper(self.scale_input(specs)) / self.x_max)
        times = torch.tensor((0., self.t_end), device=specs.device, dtype=specs.dtype)
        # Explicitly adjoint + adaptive Dormand-Prince. No fixed-step/Heun path.
        final_voltage = odeint_adjoint(self.circuit, initial_voltage, times,
                                       rtol=self.rtol, atol=self.rtol * .1, method="dopri5")[-1]
        return self.output_mapper(self.x_max * torch.tanh(final_voltage / self.x_max))


def make_topology(nodes: int, edge_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique directed edges, including ground connections, in original node numbering."""
    pairs = [(s, d) for s in range(nodes + 1) for d in range(1, nodes + 1) if s != d]
    if edge_count > len(pairs): raise ValueError("edge_count exceeds directed graph capacity")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(pairs), edge_count, replace=False)
    return (torch.tensor([pairs[i][0] for i in selected]),
            torch.tensor([pairs[i][1] for i in selected]))


def parameter_count(nodes: int, edges: int, *, fixed_leak: bool = False,
                    input_mapper_fan_in: int = 0, output_mapper_fan_in: int = 0) -> int:
    """Count exactly the live trainable tensors for this architecture."""
    input_n = nodes * ((min(input_mapper_fan_in, 4) if input_mapper_fan_in else 4) + 1)
    output_n = 7 * ((min(output_mapper_fan_in, nodes) if output_mapper_fan_in else nodes) + 1)
    return input_n + 3 * edges + (0 if fixed_leak else nodes) + output_n


def circuit_candidates(budget: int, tolerance: float, *, fixed_leak: bool = False,
                       input_mapper_fan_in: int = 0, output_mapper_fan_in: int = 0,
                       fixed_nodes: int | None = None, fixed_edges: int | None = None,
                       max_edge_density: float | None = None) -> list[tuple[int, int]]:
    result = []
    node_range = [fixed_nodes] if fixed_nodes is not None else range(16, 65)
    for nodes in node_range:
        if nodes is None or nodes < 1:
            raise ValueError("--fixed-nodes must be positive")
        max_edges = nodes * (nodes + 1)
        fixed_parameter_n = parameter_count(nodes, 0, fixed_leak=fixed_leak,
                                            input_mapper_fan_in=input_mapper_fan_in,
                                            output_mapper_fan_in=output_mapper_fan_in)
        target_edges = round((budget - fixed_parameter_n) / 3)
        # Nearby counts make edge density a genuine BO-controlled architecture choice.
        edge_range = [fixed_edges] if fixed_edges is not None else range(max(1, target_edges - 120), min(max_edges, target_edges + 120) + 1, 30)
        for edges in edge_range:
            if edges is None or not 1 <= edges <= max_edges:
                continue
            if max_edge_density is not None and edges / max_edges > max_edge_density:
                continue
            count = parameter_count(nodes, edges, fixed_leak=fixed_leak,
                                    input_mapper_fan_in=input_mapper_fan_in,
                                    output_mapper_fan_in=output_mapper_fan_in)
            if abs(count - budget) <= budget * tolerance:
                result.append((nodes, edges))
    if not result:
        raise ValueError("No directed Kirchhoff topology meets the budget/tolerance and requested fixed/sparse constraints")
    return result


TRIAL_SUMMARY_FIELDS = [
    "trial", "state", "objective_validation_loss", "best_validation_loss", "test_loss",
    "failure_rate", "nodes", "edges", "parameter_count", "activation", "t_end", "rtol",
    "lr", "weight_decay", "epochs_completed", "pruned_epoch", "elapsed_seconds", "error",
]


def append_trial_summary(output: Path, row: dict[str, Any]) -> None:
    """Append a human-readable trial index; Optuna's SQLite DB remains authoritative."""
    path = output / "trial_summary.csv"
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_SUMMARY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in TRIAL_SUMMARY_FIELDS})


def batch_size_choices(value: str) -> list[int]:
    choices = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not choices or min(choices) < 1:
        raise ValueError("--batch-size-choices must be a nonempty comma-separated list of positive integers")
    return choices


def selected_parameter_count(nodes: int, edges: int, args: argparse.Namespace) -> int:
    return parameter_count(nodes, edges, fixed_leak=args.fixed_leak,
                           input_mapper_fan_in=args.input_mapper_fan_in,
                           output_mapper_fan_in=args.output_mapper_fan_in)


def select_architecture(trial: optuna.Trial, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    """Avoid a meaningless Optuna dimension when capacity is fully fixed."""
    if len(candidates) == 1:
        return candidates[0]
    return candidates[trial.suggest_int("circuit_architecture", 0, len(candidates) - 1)]


def select_trial_value(trial: optuna.Trial, name: str, choices: list[Any], fixed: Any | None) -> Any:
    """Use a fixed CLI value verbatim, otherwise make the value a BO dimension."""
    return fixed if fixed is not None else trial.suggest_categorical(name, choices)


def trial_topology_seed(args: argparse.Namespace, trial: optuna.Trial) -> int:
    return int(args.topology_seed if args.topology_seed is not None else args.seed + trial.number)


def build_student(nodes: int, edges: int, activation: str, t_end: float, rtol: float,
                  args: argparse.Namespace, trial: optuna.Trial) -> CTLEKirchhoffNet:
    return CTLEKirchhoffNet(
        nodes, edges, activation, trial_topology_seed(args, trial), t_end, rtol,
        x_max=args.x_max, clip_current=args.clip_current,
        clip_softness=args.clip_softness, leak_init=args.leak_init, c_eff=args.c_eff,
        bounded_edge_parameters=args.bounded_edge_parameters,
        edge_weight_max=args.edge_weight_max, edge_bias_max=args.edge_bias_max,
        fixed_leak=args.fixed_leak, input_mapper_fan_in=args.input_mapper_fan_in,
        output_mapper_fan_in=args.output_mapper_fan_in,
    )


def trial_train(trial: optuna.Trial, data: dict[str, Any], args: argparse.Namespace, device: torch.device,
                candidates: list[tuple[int, int]], output: Path) -> float:
    nodes, edges = candidates[trial.suggest_int("circuit_architecture", 0, len(candidates) - 1)]
    activation = trial.suggest_categorical("device_activation", ["tanh", "relu"])
    t_end = trial.suggest_categorical("t_end", [.25, .5, 1.])
    rtol = trial.suggest_categorical("rtol", [1e-3, 3e-4])
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", batch_size_choices(args.batch_size_choices))
    trial_dir = output / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model = CTLEKirchhoffNet(nodes, edges, activation, args.seed + trial.number, t_end, rtol,
                             x_max=args.x_max, clip_current=args.clip_current,
                             clip_softness=args.clip_softness, leak_init=args.leak_init,
                             c_eff=args.c_eff).to(device)
    trial.set_user_attr("nodes", nodes); trial.set_user_attr("edges", edges)
    trial.set_user_attr("parameter_count", parameter_count(nodes, edges))
    metadata = {
        "trial": trial.number, "state": "RUNNING", "dataset_fingerprint": data["fingerprint"],
        "architecture": {"nodes": nodes, "edges": edges, "parameter_count": parameter_count(nodes, edges),
                         "topology_seed": args.seed + trial.number, "device_activation": activation},
        "bo_parameters": dict(trial.params), "ode": {"solver": "odeint_adjoint", "method": "dopri5", "t_end": t_end, "rtol": rtol, "atol": rtol * .1},
        "constraints": {"x_max": args.x_max, "clip_current": args.clip_current, "clip_softness": args.clip_softness, "leak_init": args.leak_init, "c_eff": args.c_eff},
        "metrics_note": "failure_rate is null: this compact baseline does not run the external ZIG/forward-surrogate evaluator.",
    }
    (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    idx = data["train"]
    tensors = [torch.from_numpy(data["x"][idx]), torch.from_numpy(data["y"][idx])]
    if data["teacher"] is not None: tensors.append(torch.from_numpy(data["teacher"][idx]))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)
    best, state = float("inf"), None
    started = time.perf_counter()
    history_path, log_path = trial_dir / "history.csv", trial_dir / "log.txt"
    checkpoint_payload: dict[str, Any] | None = None
    with history_path.open("w", newline="", encoding="utf-8") as history, log_path.open("w", encoding="utf-8") as log:
        history_writer = csv.DictWriter(history, fieldnames=["epoch", "train_loss", "validation_loss", "validity_ramp", "train_nfe", "validation_nfe", "mean_grad_norm", "elapsed_seconds"])
        history_writer.writeheader()
        log.write("# CTLE KirchhoffNet BO trial\n" + json.dumps(metadata, indent=2) + "\n\n# epoch descent\n")
        log.write("epoch train_loss validation_loss ramp train_nfe validation_nfe mean_grad_norm elapsed_seconds\n")
        try:
            for epoch in range(args.epochs):
                epoch_start = time.perf_counter(); model.train(); model.circuit.reset_nfe()
                ramp = min(1., max(0., (epoch + 2 - args.ctle_phase_a_validity_ramp_start) / max(1, args.ctle_phase_a_validity_ramp_epochs)))
                loss_sum = grad_sum = 0.0
                for batch in loader:
                    opt.zero_grad(set_to_none=True); x, y = batch[0].to(device), batch[1].to(device)
                    teacher = batch[2].to(device) if len(batch) == 3 else None
                    train_loss = loss_fn(model(x), y, teacher, args, data["power_norm"], ramp)
                    train_loss.backward()
                    grad_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.).item())
                    opt.step(); loss_sum += float(train_loss.detach().item())
                train_nfe = model.circuit.nfe; model.circuit.reset_nfe()
                value = evaluate(model, data, "val", device, args); validation_nfe = model.circuit.nfe
                row = {"epoch": epoch + 1, "train_loss": loss_sum / max(1, len(loader)), "validation_loss": value,
                       "validity_ramp": ramp, "train_nfe": train_nfe, "validation_nfe": validation_nfe,
                       "mean_grad_norm": grad_sum / max(1, len(loader)), "elapsed_seconds": time.perf_counter() - epoch_start}
                history_writer.writerow(row); history.flush()
                log.write("{epoch} {train_loss:.8g} {validation_loss:.8g} {validity_ramp:.6g} {train_nfe} {validation_nfe} {mean_grad_norm:.8g} {elapsed_seconds:.3f}\n".format(**row)); log.flush()
                if value < best:
                    best, state = value, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                trial.report(value, epoch)
                if trial.should_prune():
                    metadata.update({"state": "PRUNED", "pruned_epoch": epoch + 1, "best_validation_loss": best, "elapsed_seconds": time.perf_counter() - started})
                    if state is not None:
                        torch.save({"state_dict": state, "validation_loss": best, "pruned": True}, trial_dir / "pruned_best_model.pt")
                    (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    append_trial_summary(output, {"trial": trial.number, "state": "PRUNED", "best_validation_loss": best, "failure_rate": None, "nodes": nodes, "edges": edges, "parameter_count": parameter_count(nodes, edges), "activation": activation, "t_end": t_end, "rtol": rtol, "lr": lr, "weight_decay": wd, "epochs_completed": epoch + 1, "pruned_epoch": epoch + 1, "elapsed_seconds": metadata["elapsed_seconds"]})
                    log.write(f"# PRUNED after epoch {epoch + 1}; best_validation_loss={best:.8g}\n")
                    raise optuna.TrialPruned()
            assert state is not None
            checkpoint_payload = {"state_dict": state, "nodes": nodes, "edges": edges, "activation": activation, "t_end": t_end,
                                  "rtol": rtol, "x_max": args.x_max, "clip_current": args.clip_current,
                                  "clip_softness": args.clip_softness, "leak_init": args.leak_init, "c_eff": args.c_eff,
                                  "validation_loss": best, "parameter_count": parameter_count(nodes, edges)}
            torch.save(checkpoint_payload, trial_dir / "best_model.pt")
            model.load_state_dict(state); model.circuit.reset_nfe(); test_loss = evaluate(model, data, "test", device, args)
            metadata.update({"state": "COMPLETE", "best_validation_loss": best, "test_loss": test_loss, "test_nfe": model.circuit.nfe, "elapsed_seconds": time.perf_counter() - started})
            (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            append_trial_summary(output, {"trial": trial.number, "state": "COMPLETE", "objective_validation_loss": best, "best_validation_loss": best, "test_loss": test_loss, "failure_rate": None, "nodes": nodes, "edges": edges, "parameter_count": parameter_count(nodes, edges), "activation": activation, "t_end": t_end, "rtol": rtol, "lr": lr, "weight_decay": wd, "epochs_completed": args.epochs, "elapsed_seconds": metadata["elapsed_seconds"]})
            log.write(f"# COMPLETE best_validation_loss={best:.8g} test_loss={test_loss:.8g} test_nfe={model.circuit.nfe}\n")
        except Exception as exc:
            if isinstance(exc, optuna.TrialPruned):
                raise
            metadata.update({"state": "FAIL", "error": repr(exc), "elapsed_seconds": time.perf_counter() - started})
            (trial_dir / "trial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            append_trial_summary(output, {"trial": trial.number, "state": "FAIL", "failure_rate": None, "nodes": nodes, "edges": edges, "parameter_count": parameter_count(nodes, edges), "activation": activation, "t_end": t_end, "rtol": rtol, "lr": lr, "weight_decay": wd, "elapsed_seconds": metadata["elapsed_seconds"], "error": repr(exc)})
            log.write(f"# FAIL {exc!r}\n")
            raise
    return best


def build_phase_a_context(args: argparse.Namespace, output: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load the exact shared CTLE evaluator/scalers used by KNet Phase A."""
    import ctle_dagger_common as phase

    canonical = phase.load_canonical_flow_dataset(args.ctle_canonical_dataset)
    setup_args = argparse.Namespace(
        teacher_dir=args.teacher_dir or phase.DEFAULT_TEACHER_DIR,
        data_dir=args.data_dir or phase.DEFAULT_DATA_DIR,
        output=str(output / "_phase_a_shared_assets"), device=args.device, seed=args.seed,
        input_preprocessing="knet", param_budget=args.param_budget,
    )
    base = phase.setup(setup_args)
    # Canonical bounds, not the historical-frame bounds setup() derived, are
    # the shared Phase-A preprocessing contract.
    base.update({"canonical_dataset": canonical, "input_log_min": None, "input_log_max": None})
    return phase, base, canonical


def phase_a_trial(trial: optuna.Trial, phase: Any, base_ctx: dict[str, Any], canonical: dict[str, Any],
                  args: argparse.Namespace, candidates: list[tuple[int, int]], output: Path) -> float:
    """Run the existing Phase-A loop unchanged, substituting only the student."""
    nodes, edges = select_architecture(trial, candidates)
    activation = "tanh" if args.tanh_only else trial.suggest_categorical("device_activation", ["tanh", "relu"])
    t_end = select_trial_value(trial, "t_end", [.25, .5, 1.], args.fixed_t_end)
    rtol = select_trial_value(trial, "rtol", [1e-3, 3e-4], args.fixed_rtol)
    lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", batch_size_choices(args.batch_size_choices))
    trial_dir = output / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    student = build_student(nodes, edges, activation, t_end, rtol, args, trial)
    # Align this wrapper's fixed physical decode bounds with the shared phase
    # harness, rather than relying on a duplicated local constant.
    bounds = list(phase.PARAM_LOG_BOUNDS.values())
    with torch.no_grad():
        student.log_lo.copy_(torch.tensor([b[0] for b in bounds], dtype=torch.float32))
        student.log_hi.copy_(torch.tensor([b[1] for b in bounds], dtype=torch.float32))
    phase.init_output_affine_bias_from_labels(
        student, canonical["params"][np.asarray(canonical["train_idx"], dtype=np.int64)], phase.PARAM_LOG_BOUNDS)
    trial.set_user_attr("nodes", nodes); trial.set_user_attr("edges", edges)
    trial.set_user_attr("parameter_count", selected_parameter_count(nodes, edges, args))
    trial.set_user_attr("phase_a_framework", "ctle_dagger_common.run_phase_a_training")
    trial.set_user_attr("constraints", {
        "tanh_only": args.tanh_only, "fixed_leak": args.fixed_leak,
        "bounded_edge_parameters": args.bounded_edge_parameters,
        "edge_weight_max": args.edge_weight_max, "edge_bias_max": args.edge_bias_max,
        "input_mapper_fan_in": args.input_mapper_fan_in,
        "output_mapper_fan_in": args.output_mapper_fan_in,
        "topology_seed": args.topology_seed,
    })
    ctx = dict(base_ctx)
    ctx.update({
        "canonical_dataset": canonical, "epochs": args.epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": wd, "output_dir": str(trial_dir), "grad_clip": 1.0,
        "val_eval_every": args.ctle_eval_every, "earlystop_patience": args.ctle_earlystop_patience,
        "error_threshold": DEGRADE_REL_THRESHOLD, "validity_weight": args.ctle_phase_a_validity_weight,
        "validity_ramp_start": args.ctle_phase_a_validity_ramp_start,
        "validity_ramp_epochs": args.ctle_phase_a_validity_ramp_epochs,
        "mlp_weight": args.ctle_phase_a_mlp_weight, "fwd_weight": args.ctle_phase_a_fwd_weight,
        "power_weight": args.ctle_phase_a_power_weight, "mapper_lr_scale": args.mapper_lr_scale,
        "struct_lr_scale": args.struct_lr_scale, "dyn_lr_scale": args.dyn_lr_scale,
        "input_log_min": None, "input_log_max": None,
    })
    log_path = trial_dir / "log.txt"
    log_path.write_text("# Original KirchhoffNet transplanted into the shared CTLE Phase-A framework\n"
                        f"# BO params: {json.dumps(trial.params, sort_keys=True)}\n"
                        f"# Constraints: {json.dumps(trial.user_attrs['constraints'], sort_keys=True)}\n"
                        f"# Architecture: nodes={nodes} edges={edges} trainable_params={selected_parameter_count(nodes, edges, args)} "
                        f"topology_seed={trial_topology_seed(args, trial)} activation={activation} t_end={t_end} rtol={rtol}\n",
                        encoding="utf-8")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    phase._logger.addHandler(handler)
    try:
        result = phase.run_phase_a_training(student, ctx, model_name="original_kirchhoffnet")
    except Exception as exc:
        trial.set_user_attr("error", repr(exc))
        append_trial_summary(output, {"trial": trial.number, "state": "FAIL", "nodes": nodes, "edges": edges,
            "parameter_count": selected_parameter_count(nodes, edges, args), "activation": activation, "t_end": t_end, "rtol": rtol,
            "lr": lr, "weight_decay": wd, "error": repr(exc)})
        raise
    finally:
        phase._logger.removeHandler(handler)
        handler.close()
    final = result["final"]
    val_failure = float(final["val"]["failure_rate"])
    test_failure = float(final["test"]["failure_rate"])
    # Same Phase-A lexicographic tie-break as kn_bayes_opt: among equal
    # failure rates, prefer lower valid-design power and then lower teacher
    # logit MSE.  It is deliberately tiny and never replaces the failure
    # metric as the primary objective.
    valid_power = float(final["val"].get("valid_mean_power", float("nan")))
    val_mlp_mse = float(final["val"].get("mlp_mse", float("nan")))
    power_ref = float(np.asarray(canonical["power_norm_const"]).item())
    lex_offset = 0.0
    if np.isfinite(valid_power) and valid_power > 0:
        lex_offset += 1e-4 * (valid_power / power_ref - 1.0)
    if np.isfinite(val_mlp_mse):
        lex_offset += 1e-6 * val_mlp_mse
    objective = val_failure + lex_offset
    trial.set_user_attr("validation_failure_rate", val_failure)
    trial.set_user_attr("test_failure_rate", test_failure)
    trial.set_user_attr("best_validation_failure_rate", float(result["best_val_failure_rate"]))
    trial.set_user_attr("phase_a_valid_mean_power", valid_power if np.isfinite(valid_power) else None)
    trial.set_user_attr("phase_a_val_mlp_mse", val_mlp_mse if np.isfinite(val_mlp_mse) else None)
    trial.set_user_attr("lexicographic_offset", lex_offset)
    # This is the exact Phase-A BO metric.  Test is recorded strictly for
    # diagnostics and never participates in the Optuna objective.
    append_trial_summary(output, {"trial": trial.number, "state": "COMPLETE", "objective_validation_loss": objective,
        "best_validation_loss": float(result["best_val_failure_rate"]), "test_loss": test_failure,
        "failure_rate": val_failure, "nodes": nodes, "edges": edges, "parameter_count": selected_parameter_count(nodes, edges, args),
        "activation": activation, "t_end": t_end, "rtol": rtol, "lr": lr, "weight_decay": wd,
        "epochs_completed": len(result["history"]["epoch"])})
    trial.report(objective, step=len(result["history"]["epoch"]))
    return objective


def run(args: argparse.Namespace) -> None:
    if args.dataset != "ctle" or not args.ctle_phase_a: raise ValueError("Use --dataset ctle --ctle-phase-a")
    if args.ctle_canonical_dataset is None or args.output is None or args.param_budget is None: raise ValueError("canonical dataset, output, and parameter budget are required")
    if args.device == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA requested but unavailable")
    seed_everything(args.seed)
    if args.max_edge_density is not None and not 0 < args.max_edge_density <= 1:
        raise ValueError("--max-edge-density must lie in (0, 1]")
    if args.input_mapper_fan_in < 0 or args.output_mapper_fan_in < 0:
        raise ValueError("mapper fan-in constraints must be nonnegative (0 means dense)")
    if args.input_mapper_fan_in > 4:
        raise ValueError("--input-mapper-fan-in cannot exceed the four CTLE input features")
    if args.fixed_t_end is not None and args.fixed_t_end <= 0:
        raise ValueError("--fixed-t-end must be positive")
    if args.fixed_rtol is not None and args.fixed_rtol <= 0:
        raise ValueError("--fixed-rtol must be positive")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    candidates = circuit_candidates(
        args.param_budget, args.param_tolerance, fixed_leak=args.fixed_leak,
        input_mapper_fan_in=args.input_mapper_fan_in, output_mapper_fan_in=args.output_mapper_fan_in,
        fixed_nodes=args.fixed_nodes, fixed_edges=args.fixed_edges,
        max_edge_density=args.max_edge_density,
    )
    phase, base_ctx, canonical = build_phase_a_context(args, output)
    if args.ctle_phase_a_mlp_weight > 0 and "mlp_logits_trial0019" not in canonical:
        raise ValueError("Teacher distillation needs schema-2 data with mlp_logits_trial0019")
    activation_choices = ["tanh"] if args.tanh_only else ["tanh", "relu"]
    (output / "run_config.json").write_text(json.dumps({**vars(args), "fingerprint": str(np.asarray(canonical["fingerprint"]).item()), "candidate_count": len(candidates), "candidate_architectures": candidates, "solver": "odeint_adjoint/dopri5", "device_activation_choices": activation_choices, "node_constraint": "bounded tanh broadcast + soft rail + positive leak", "training_framework": "ctle_dagger_common.run_phase_a_training"}, indent=2, default=str))
    storage = f"sqlite:///{(output / 'study.db').as_posix()}"
    study = optuna.create_study(study_name="ctle_kirchhoff_bo", storage=storage, load_if_exists=args.resume,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, args.n_trials), n_warmup_steps=max(1, args.epochs // 5)))
    done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(lambda t: phase_a_trial(t, phase, base_ctx, canonical, args, candidates, output), n_trials=max(0, args.n_trials - done))
    best = study.best_trial
    history_path = output / f"trial_{best.number:04d}" / "phase_a_history.json"
    with history_path.open(encoding="utf-8") as handle:
        best_history = json.load(handle)
    summary = {"best_trial": best.number, "validation_failure_rate": best.value,
        "test_failure_rate": best_history["final"]["test"]["failure_rate"],
        "parameter_count": best.user_attrs["parameter_count"], "params": best.params}
    shutil.copy2(output / f"trial_{best.number:04d}" / "phase_a_best.pt", output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
    engine = getattr(getattr(study._storage, "_backend", study._storage), "engine", None)
    if engine is not None: engine.dispose()


def smoke_test() -> None:
    """Small adjoint/backward check; full tests require the external CTLE assets."""
    model = CTLEKirchhoffNet(16, 80, "tanh", 7, .25, 1e-3, x_max=3.,
                             clip_current=.05, clip_softness=.02, leak_init=.0486, c_eff=1.)
    model.attach_scaler(input_log_min=np.full(4, -2., dtype=np.float32),
                        input_log_max=np.full(4, 2., dtype=np.float32))
    x = torch.full((4, 4), .1, dtype=torch.float32)
    loss = model(x).square().mean(); loss.backward()
    assert model.circuit.edge.grad is not None and torch.isfinite(model.circuit.edge.grad).all()
    # Exercise the optional finite-device, fixed-leak, and truly sparse-mapper
    # paths independently of external CTLE/ZIG assets.
    constrained = CTLEKirchhoffNet(16, 80, "tanh", 11, .25, 1e-3, x_max=3.,
                                   clip_current=.05, clip_softness=.02, leak_init=.0486, c_eff=1.,
                                   bounded_edge_parameters=True, fixed_leak=True,
                                   input_mapper_fan_in=2, output_mapper_fan_in=3)
    constrained.attach_scaler(input_log_min=np.full(4, -2., dtype=np.float32),
                               input_log_max=np.full(4, 2., dtype=np.float32))
    constrained(x).square().mean().backward()
    assert constrained.circuit.raw_leak is None
    assert constrained.circuit.edge.grad is not None and torch.isfinite(constrained.circuit.edge.grad).all()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", default="ctle", choices=["ctle"]); p.add_argument("--ctle-phase-a", action="store_true")
    p.add_argument("--ctle-canonical-dataset", type=Path); p.add_argument("--ctle-phase-a-mlp-teacher-ckpt", type=Path, help="Recorded with run provenance; teacher logits are frozen in schema-2 canonical data.")
    p.add_argument("--teacher-dir", type=str, default=None, help="ZIG/scaler artifact directory; defaults to the existing Phase-A harness path.")
    p.add_argument("--data-dir", type=str, default=None, help="Historical CTLE CSV directory; defaults to the existing Phase-A harness path.")
    p.add_argument("--ctle-phase-a-mlp-weight", type=float, default=1.); p.add_argument("--ctle-phase-a-fwd-weight", type=float, default=0.)
    p.add_argument("--ctle-phase-a-power-weight", type=float, default=0.); p.add_argument("--ctle-phase-a-validity-weight", type=float, default=0.)
    p.add_argument("--ctle-phase-a-validity-ramp-start", type=int, default=10); p.add_argument("--ctle-phase-a-validity-ramp-epochs", type=int, default=30)
    p.add_argument("--ctle-objective", choices=["validation"], default="validation"); p.add_argument("--param-budget", type=int); p.add_argument("--param-tolerance", type=float, default=.15)
    p.add_argument("--epochs", type=int, default=300); p.add_argument("--n-trials", type=int, default=30); p.add_argument("--seed", type=int, default=100)
    p.add_argument("--batch-size-choices", default="512,1024,2048,4096", help="BO batch-size choices; matches kn_bayes_opt by default.")
    p.add_argument("--lr-min", type=float, default=1e-4); p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--ctle-eval-every", type=int, default=10, help="ZIG failure-rate evaluation cadence, matching Phase-A.")
    p.add_argument("--ctle-earlystop-patience", type=int, default=200, help="Phase-A no-improvement patience measured in evaluation cycles.")
    p.add_argument("--mapper-lr-scale", type=float, default=1.0)
    p.add_argument("--struct-lr-scale", type=float, default=4.0)
    p.add_argument("--dyn-lr-scale", type=float, default=1.0)
    constraints = p.add_argument_group("optional fairness constraints (each independent)")
    constraints.add_argument("--tanh-only", action="store_true",
                             help="Disable the ReLU BO choice and use tanh devices only.")
    constraints.add_argument("--fixed-leak", action="store_true",
                             help="Make every node leak exactly --leak-init rather than trainable.")
    constraints.add_argument("--bounded-edge-parameters", action="store_true",
                             help="Map all three raw edge coefficients through tanh into finite ranges.")
    constraints.add_argument("--edge-weight-max", type=float, default=1.0,
                             help="Absolute source/destination coefficient bound with --bounded-edge-parameters.")
    constraints.add_argument("--edge-bias-max", type=float, default=1.0,
                             help="Absolute edge-bias bound with --bounded-edge-parameters.")
    constraints.add_argument("--fixed-nodes", type=int, default=None,
                             help="Fix circuit node count; omit to let BO choose it.")
    constraints.add_argument("--fixed-edges", type=int, default=None,
                             help="Fix directed edge count; omit to let BO choose it.")
    constraints.add_argument("--max-edge-density", type=float, default=None,
                             help="Keep only graphs with E / (N*(N+1)) no greater than this value.")
    constraints.add_argument("--topology-seed", type=int, default=None,
                             help="Use this one deterministic graph seed for every trial (default: a different seed per trial).")
    constraints.add_argument("--input-mapper-fan-in", type=int, default=0,
                             help="Connections per circuit node in the 4->N encoder; 0 retains dense mapping.")
    constraints.add_argument("--output-mapper-fan-in", type=int, default=0,
                             help="Connections per output in the N->7 readout; 0 retains dense mapping.")
    constraints.add_argument("--fixed-t-end", type=float, default=None,
                             help="Fix ODE integration horizon and remove t_end from BO.")
    constraints.add_argument("--fixed-rtol", type=float, default=None,
                             help="Fix Dopri5 rtol and remove rtol from BO.")
    p.add_argument("--x-max", type=float, default=3.0, help="Fixed differential node-voltage rail.")
    p.add_argument("--clip-current", type=float, default=0.05, help="Fixed soft-rail restoring-current magnitude.")
    p.add_argument("--clip-softness", type=float, default=0.02, help="Fixed soft-rail transition width.")
    p.add_argument("--leak-init", type=float, default=0.0486, help="Initial positive programmable node leak.")
    p.add_argument("--c-eff", type=float, default=1.0, help="Fixed effective node capacitance.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output", type=Path); p.add_argument("--resume", action="store_true"); p.add_argument("--smoke-test", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(); smoke_test() if args.smoke_test else run(args)
