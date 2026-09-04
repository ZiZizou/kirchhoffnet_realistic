"""Dump the per-trial CLI flag table from a fixed-distillation ``study.db``.

Read-only reporting for ``fixed-distillation-bayes-opt.py`` studies (any
sampler revision): the exact harness flags live in each trial's
``user_attrs["config"]``, so no SQLite spelunking or ``arch_idx`` decoding
is needed. Works mid-run (RUNNING trials included).

Examples:
    python summarize_fixed_bo.py --study-db outputs/fixed_knet_bo_plain_mlp_teacher/study.db
    python summarize_fixed_bo.py --study-db ./study.db --trial 3 --show-command
    python summarize_fixed_bo.py --study-db ./study.db --csv trial_table.csv

Uses ``optuna`` when importable, otherwise falls back to a dependency-free
read of the SQLite schema (standard library ``sqlite3`` in read-only mode),
so the table works on machines without the ML environment. Never writes to
the database.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-db", type=Path, required=True,
                        help="Path to the Optuna SQLite file (study.db).")
    parser.add_argument("--study-name", default=None,
                        help="Study to dump; required only if the DB holds several.")
    parser.add_argument("--trials-dir", type=Path, default=None,
                        help="BO --output directory holding trial_XXXX/metrics.json. "
                             "Defaults to the study.db parent directory.")
    parser.add_argument("--trial", type=int, default=None,
                        help="Show full detail (flags, attrs, metrics.json) for one trial.")
    parser.add_argument("--show-command", action="store_true",
                        help="Print the re-runnable student/training flag block per trial. "
                             "Teacher/dataset/output flags come from the base command "
                             "and are not stored in the DB.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Also write the table to this CSV path.")
    return parser.parse_args()


def load_study(db_path: Path, study_name: str | None):
    try:
        import optuna
    except ImportError:
        optuna = None
    if optuna is not None:
        storage = f"sqlite:///{db_path.resolve()}"
        names = optuna.get_all_study_names(storage)
        if study_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"{db_path} holds {len(names)} studies {names}; pass --study-name."
                )
            study_name = names[0]
        return optuna.load_study(study_name=study_name, storage=storage)
    return _load_sqlite(db_path, study_name)


def _load_sqlite(db_path: Path, study_name: str | None):
    """Minimal dependency-free study reader (stdlib ``sqlite3``, read-only).

    Returns an object with the same surface the table builder needs:
    ``study_name``, ``user_attrs``, ``trials`` (each with ``number``,
    ``state.name``, ``params``, ``values``, ``user_attrs``).
    """
    import sqlite3
    from types import SimpleNamespace

    con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        names = [row[0] for row in con.execute("SELECT study_name FROM studies")]
        if study_name is None:
            if len(names) != 1:
                raise ValueError(
                    f"{db_path} holds {len(names)} studies {names}; pass --study-name."
                )
            study_name = names[0]
        study_id = con.execute(
            "SELECT study_id FROM studies WHERE study_name = ?", (study_name,)
        ).fetchone()
        if study_id is None:
            raise ValueError(f"study '{study_name}' not found in {db_path}")
        study_id = study_id[0]
        study_attrs = {
            key: json.loads(value)
            for key, value in con.execute(
                "SELECT \"key\", value_json FROM study_user_attributes WHERE study_id = ?",
                (study_id,),
            )
        }
        trials = []
        for trial_id, number, state in con.execute(
            "SELECT trial_id, number, state FROM trials WHERE study_id = ? ORDER BY number",
            (study_id,),
        ):
            params: dict = {}
            for name, internal, dist_json in con.execute(
                "SELECT param_name, param_value, distribution_json "
                "FROM trial_params WHERE trial_id = ?",
                (trial_id,),
            ):
                dist = json.loads(dist_json)
                if dist.get("name") == "CategoricalDistribution":
                    choices = dist["attributes"]["choices"]
                    params[name] = choices[int(internal)]
                else:
                    params[name] = internal
            values = [
                row[0] for row in con.execute(
                    "SELECT value FROM trial_values WHERE trial_id = ? ORDER BY objective",
                    (trial_id,),
                )
            ]
            user_attrs = {
                key: json.loads(value)
                for key, value in con.execute(
                    "SELECT \"key\", value_json FROM trial_user_attributes WHERE trial_id = ?",
                    (trial_id,),
                )
            }
            trials.append(SimpleNamespace(
                number=number,
                state=SimpleNamespace(name=state),
                params=params,
                values=values or None,
                user_attrs=user_attrs,
            ))
    finally:
        con.close()
    return SimpleNamespace(study_name=study_name, user_attrs=study_attrs, trials=trials)


def detect_kind(study) -> str:
    raw = study.user_attrs.get("sampling_fingerprint", "")
    try:
        kind = json.loads(raw).get("student_kind", "")
    except (TypeError, json.JSONDecodeError):
        kind = ""
    if kind in ("knet", "mlp"):
        return kind
    if study.study_name.endswith("_mlp"):
        return "mlp"
    return "knet"


def arch_fields(kind: str, trial) -> dict[str, str]:
    cfg = trial.user_attrs.get("config") or {}
    if kind == "mlp":
        fields = {
            "layers": str(cfg.get("--student-layers", "")),
            "width": str(cfg.get("--student-width", "")),
            "ln": str(cfg.get("--student-use-layernorm", "")),
            "act": str(cfg.get("--student-activation", "")),
        }
        if not fields["layers"] and "mlp_arch_idx" in trial.params:
            fields["arch_idx"] = str(trial.params["mlp_arch_idx"])
        return fields
    fields = {
        "hidden": str(cfg.get("--kn-num-hidden", "")),
        "stages": str(cfg.get("--kn-num-stages", "")),
        "k": str(cfg.get("--kn-small-world-k", "")),
        "rank": str(cfg.get("--kn-vca-rank", "")),
        "x_max": str(cfg.get("--kn-x-max", "")),
    }
    if not fields["hidden"] and "knet_arch_idx" in trial.params:
        fields["arch_idx"] = str(trial.params["knet_arch_idx"])
    return fields


def fmt_number(value: object, *, percent: bool = False) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return f"{100.0 * number:.2f}" if percent else f"{number:.8g}"


def trial_note(trial) -> str:
    if trial.user_attrs.get("penalized") is True:
        return str(trial.user_attrs.get("penalty_reason", "penalized"))[:60]
    if trial.user_attrs.get("promoted") is False:
        return "probe-only"
    if trial.user_attrs.get("promoted") is True and not trial.values:
        return "promoted, training"
    return ""


def build_rows(study, kind: str) -> tuple[list[str], list[list[str]]]:
    columns = (["trial", "state", "layers", "width", "ln", "act"] if kind == "mlp"
               else ["trial", "state", "hidden", "stages", "k", "rank", "x_max"])
    columns += ["lr", "batch", "params", "fail%", "mse", "note"]
    ordered = sorted(study.trials, key=lambda t: t.number)
    if any("arch_idx" in arch_fields(kind, trial) for trial in ordered):
        columns.insert(columns.index("lr"), "arch_idx")
    rows: list[list[str]] = []
    for trial in ordered:
        values = list(trial.values) if trial.values else []
        row = {
            "trial": f"{trial.number:04d}",
            "state": trial.state.name,
            "lr": str(trial.params.get("lr", "")),
            "batch": str(trial.params.get("batch_size", "")),
            "params": str(trial.user_attrs.get("actual_params", "")),
            "fail%": fmt_number(values[0], percent=True) if len(values) > 0 else "",
            "mse": fmt_number(values[1]) if len(values) > 1 else "",
            "note": trial_note(trial),
            **arch_fields(kind, trial),
        }
        rows.append([row.get(col, "") for col in columns])
    return columns, rows


def print_table(columns: list[str], rows: list[list[str]]) -> None:
    widths = [len(col) for col in columns]
    for row in rows:
        for pos, cell in enumerate(row):
            widths[pos] = max(widths[pos], len(cell))
    print("  ".join(col.ljust(widths[pos]) for pos, col in enumerate(columns)))
    for row in rows:
        print("  ".join(cell.ljust(widths[pos]) for pos, cell in enumerate(row)))


def trial_metrics(trials_dir: Path | None, number: int) -> dict | None:
    if trials_dir is None:
        return None
    path = trials_dir / f"trial_{number:04d}" / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def show_trial_detail(study, kind: str, number: int, trials_dir: Path | None,
                      show_command: bool) -> int:
    matches = [t for t in study.trials if t.number == number]
    if not matches:
        print(f"trial {number} not found in study '{study.study_name}'", file=sys.stderr)
        return 1
    trial = matches[0]
    print(f"trial {trial.number:04d}  state={trial.state.name}")
    print(f"optuna params: {json.dumps(trial.params, sort_keys=True)}")
    print(f"values: {list(trial.values) if trial.values else None}")
    print(f"arch: {json.dumps(arch_fields(kind, trial), sort_keys=True)}")
    print("config flags:")
    for flag in sorted((trial.user_attrs.get("config") or {}).items()):
        print(f"  {flag[0]}={flag[1]}")
    interesting = {k: v for k, v in trial.user_attrs.items() if k != "config"}
    print(f"user attrs: {json.dumps(interesting, sort_keys=True, default=str)}")
    metrics = trial_metrics(trials_dir, number)
    if metrics is not None:
        final = metrics.get("final", {})
        print(f"metrics.json final: {json.dumps(final, sort_keys=True, default=str)[:2000]}")
        if show_command:
            print("metrics.json args keys: "
                  + ", ".join(sorted(metrics.get("args", {}).keys())))
    elif trials_dir is not None:
        print(f"no metrics.json under {trials_dir}/trial_{number:04d}/")
    if show_command:
        cfg = trial.user_attrs.get("config") or {}
        flags = " ".join(f"{flag} {value}" for flag, value in sorted(cfg.items()))
        print(f"command fragment: python fixed-mlp-distillation-kirchhoffnet.py "
              f"--student-kind {kind} {flags}")
        print("(prepend teacher/dataset/output/device/seed/epochs flags from the base command)")
    return 0


def main() -> int:
    args = parse_args()
    if not args.study_db.is_file():
        print(f"study.db not found: {args.study_db}", file=sys.stderr)
        return 1
    trials_dir = args.trials_dir or args.study_db.resolve().parent
    study = load_study(args.study_db, args.study_name)
    kind = detect_kind(study)
    print(f"study '{study.study_name}' kind={kind} trials={len(study.trials)} db={args.study_db}")
    if args.trial is not None:
        return show_trial_detail(study, kind, args.trial, trials_dir, args.show_command)
    columns, rows = build_rows(study, kind)
    print_table(columns, rows)
    if args.csv is not None:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    if args.show_command:
        print("\n(hint: add --trial N --show-command for one trial's re-runnable flag block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
