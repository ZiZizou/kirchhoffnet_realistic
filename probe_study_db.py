"""Diagnose a crashed fixed-distillation Optuna study DB.

Run on the failing host (where the offending study.db lives):

    python probe_study_db.py --db /path/to/study.db [--study-name NAME]

Reports the study's declared directions, the per-trial (number, state,
len(values), values) rows, and flags any trial whose `values` shape is
inconsistent with the declared direction count.  The (2,) + inhomogeneous
crash in optuna.samplers._tpe.sampler._calculate_weights_below_for_multi_objective
is exactly caused by such mismatched trials.

Outputs JSON to stdout for downstream inspection.

Exit codes:
  0  study shape is internally consistent
  2  no studies / ambiguous / missing db / optuna error
  3  offending trials detected (inhomogeneous values)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True,
                        help="Path to the study.sqlite3 / study.db file (without sqlite:///).")
    parser.add_argument("--study-name", default=None,
                        help="Optuna study name; auto-discovered if omitted.")
    args = parser.parse_args()

    if not args.db.exists():
        print(json.dumps({"error": f"db not found: {args.db}", "path": str(args.db)},
                         indent=2))
        sys.exit(2)

    storage = f"sqlite:///{args.db.resolve()}"
    try:
        summaries = optuna.get_all_study_summaries(storage=storage)
    except Exception as exc:
        print(json.dumps({"error": f"optuna failed to read storage: {exc}",
                          "storage": storage}, indent=2))
        sys.exit(2)

    if not summaries:
        print(json.dumps({"error": "no studies found", "storage": storage}, indent=2))
        sys.exit(2)
    if args.study_name is None:
        if len(summaries) > 1:
            print(json.dumps({"error": "multiple studies present, pass --study-name",
                              "names": [s.study_name for s in summaries]}, indent=2))
            sys.exit(2)
        study_name = summaries[0].study_name
    else:
        study_name = args.study_name

    study = optuna.load_study(study_name=study_name, storage=storage)
    directions = [d.name for d in study.directions]
    n_expected = len(directions)
    trials_info = []
    offenders = []
    for t in study.trials:
        n_values = len(t.values) if t.values is not None else None
        info = {
            "number": t.number,
            "state": t.state.name,
            "n_values": n_values,
            "values": list(t.values) if t.values is not None else None,
        }
        trials_info.append(info)
        # Any non-None values with wrong length breaks np.asarray(...) inside TPE,
        # regardless of state.  We still report them as offenders.
        if t.values is not None and n_values != n_expected:
            offenders.append(info)

    out = {
        "storage": storage,
        "study_name": study_name,
        "directions": directions,
        "n_expected_values": n_expected,
        "n_trials": len(study.trials),
        "offending_trials": offenders,
        "trials": trials_info,
    }
    print(json.dumps(out, indent=2, default=str))

    if offenders:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
