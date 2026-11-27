"""Small structural checks for Kaggle run controls (no model training)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_steps_per_stage_cli_is_wired_and_default_is_unchanged():
    source = (ROOT / "dagger-nuance-distillation-kirchhoffnet.py").read_text()
    assert "--kn-steps-per-stage', type=int" in source
    assert "stage_num_steps=_bo_args.kn_steps_per_stage" in source
    # Constructor retains the legacy solver-derived count when no override exists.
    assert "stage_num_steps if stage_num_steps is not None else max(1, int(round(SOLVER[\"num_steps\"] / num_stages)))" in source


def test_phase_a_resume_saves_and_restores_training_state():
    source = (ROOT / "ctle_dagger_common.py").read_text()
    for key in ("model", "optimizer", "scheduler", "history", "best_val_failure",
                "best_state", "best_val_loss", "best_val_loss_state", "no_improve"):
        assert f'"{key}"' in source
    assert 'start_epoch = int(state["epoch"]) + 1' in source
    assert 'for epoch in range(start_epoch, epochs):' in source


def test_smoke_sample_limit_applies_without_training():
    source = (ROOT / "dagger-nuance-distillation-kirchhoffnet.py").read_text()
    common = (ROOT / "ctle_dagger_common.py").read_text()
    assert "--phase-a-sample-limit" in source
    assert '"sample_limit": _bo_args.phase_a_sample_limit' in source
    assert "Subset(ds, range(min(limit, len(ds))))" in common


def test_compile_mode_and_max_hours_cli_and_history_fields():
    source = (ROOT / "dagger-nuance-distillation-kirchhoffnet.py").read_text()
    common = (ROOT / "ctle_dagger_common.py").read_text()
    assert "--compile-mode" in source
    assert "--max-hours" in source
    assert '"compile_mode": _bo_args.compile_mode' in source
    assert '"max_hours": _bo_args.max_hours' in source
    assert '"stopped_by_time_budget": stopped_by_time_budget' in common
    assert '"completed_epochs": completed_epochs' in common
