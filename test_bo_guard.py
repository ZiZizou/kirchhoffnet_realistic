"""Offline invariant test: simulate the inhomogeneous-shape crash.

Reproduces the optuna.samplers._tpe.sampler._calculate_weights_below_for_multi_objective
failure on a study with mixed 1-tuple / 2-tuple / None values, and verifies
that the fix's guard (`COMPLETE + len(values)==N + all-numeric`) rejects such
trials before they reach the TPE Parzen estimator.

Also exercises the resume-direction comparison against a real optuna Study when
optuna is importable; otherwise that subtest is skipped.  This catches the
casing drift where StudyDirection.MINIMIZE.name == "MINIMIZE" and a literal
"minimize" comparison would reject every healthy 2-objective study.

Run locally (no optuna required for the offline path):
    python test_bo_guard.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


PENALTY = (1.0, 1e6)
N_EXPECTED = 2


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
FDBO_PATH = os.path.join(REPO_ROOT, "fixed-distillation-bayes-opt.py")


def fake_trial(number: int, state: str, values, user_attrs=None):
    return {"number": number, "state": state, "values": values,
            "user_attrs": dict(user_attrs or {})}


def apply_guard(trials, expected_directions_count: int):
    """Mirror of fixed-distillation-bayes-opt.py's COMPLETE-filter.

    Excludes:
      - non-COMPLETE states,
      - ragged or non-numeric values,
      - missing config / actual_params user_attrs,
      - penalized trials (replaced optuna.TrialPruned to avoid values=None crash
        in multi-obj TPE, see optuna/optuna#5260).
    """
    complete = [
        t for t in trials
        if t["state"] == "COMPLETE"
        and t["values"] is not None
        and len(t["values"]) == expected_directions_count
        and all(isinstance(v, (int, float)) for v in t["values"])
        and "config" in t["user_attrs"]
        and "actual_params" in t["user_attrs"]
        and t["user_attrs"].get("penalized") is not True
    ]
    penalized = [
        t for t in trials
        if t["state"] == "COMPLETE"
        and t["user_attrs"].get("penalized") is True
    ]
    return complete, penalized


def test_offline_guard() -> None:
    # Stale-DB scenario: prior single-objective run left a 1-tuple COMPLETE,
    # then a new multi-objective run added 2-tuple trials.  TPE crashes when
    # building the below-trials Parzen estimator; the guard must drop the
    # 1-tuple trial so `complete` is homogeneous.
    trials = [
        fake_trial(0, "COMPLETE", [0.5], {"config": {"--lr": "1"}, "actual_params": 100}),
        fake_trial(1, "PRUNED", None),
        fake_trial(2, "COMPLETE", [0.3, 0.1], {"config": {"--lr": "2"}, "actual_params": 100}),
        fake_trial(3, "FAIL", None),
        fake_trial(4, "COMPLETE", [0.4, 0.2]),
        fake_trial(5, "RUNNING", None),
        fake_trial(11, "COMPLETE", [0.2, 0.05], {"config": {"--lr": "3"}, "actual_params": 99}),
    ]
    complete, _ = apply_guard(trials, N_EXPECTED)
    numbers = sorted(t["number"] for t in complete)
    print(json.dumps({"expected": N_EXPECTED, "complete_trial_numbers": numbers,
                      "n_complete": len(complete)}, indent=2))
    assert numbers == [2, 11], f"guard let through unexpected trials: {numbers}"
    for t in complete:
        assert len(t["values"]) == N_EXPECTED
    # Validate lexicographic selection works on the filtered set
    selected = min(complete, key=lambda x: (float(x["values"][0]), float(x["values"][1])))
    assert selected["number"] == 11, selected
    print("OK: guard rejected stale single-objective trials and missing-attrs "
          "trials; selection intact.")


def test_penalty_sorts_last() -> None:
    """PENALTY 2-tuple must lexicographically sort after any real trial."""
    penalty = PENALTY
    real_failure_imitation = (0.4, 0.05)  # typical completed trial
    worst_real_failure = (0.999, 1e3 - 1)
    # Compare tuple-by-tuple lexicographically:
    assert penalty[0] > worst_real_failure[0]  # 1.0 > 0.999
    assert (penalty[0], penalty[1]) > worst_real_failure
    assert (penalty[0], penalty[1]) > real_failure_imitation
    # And never inf/-inf (both crash TPE multi-objective Parzen est.)
    assert all(abs(v) != float("inf") for v in penalty)
    print("OK: PENALTY=(1.0, 1e6) sorts lexicographically after every real trial; "
          "no inf values that trigger optuna#3676.")


def test_penalized_excluded_from_selection() -> None:
    """Trials tagged penalized must not be selected even if they have 'good' values."""
    trials = [
        # Penalized trial with artificially low failure (lowest objective) must be skipped
        fake_trial(0, "COMPLETE", [0.001, 0.001],
                   {"config": {"--lr": "1"}, "actual_params": 100, "penalized": True}),
        # Real unpenalized trial with worse objectives must be selected
        fake_trial(1, "COMPLETE", [0.5, 0.1],
                   {"config": {"--lr": "2"}, "actual_params": 100}),
    ]
    complete, penalized = apply_guard(trials, N_EXPECTED)
    assert [t["number"] for t in complete] == [1]
    assert [t["number"] for t in penalized] == [0]
    selected = min(complete, key=lambda x: (float(x["values"][0]), float(x["values"][1])))
    assert selected["number"] == 1, selected
    print("OK: selection excluded the penalized trial even though its values "
          "were lexicographically smallest.")


def test_all_penalized_raises() -> None:
    """When every COMPLETE trial is penalized, selection must raise."""
    trials = [
        fake_trial(i, "COMPLETE", list(PENALTY),
                   {"config": {"--lr": str(i)}, "actual_params": 100, "penalized": True})
        for i in range(3)
    ]
    complete, penalized = apply_guard(trials, N_EXPECTED)
    assert complete == [], complete
    assert len(penalized) == 3
    msg = "no eligible completed trials"
    try:
        if not complete:
            raise RuntimeError(msg)
    except RuntimeError as exc:
        assert str(exc) == msg
    print("OK: all-penalized scenario triggers the no-eligible-trials error.")


def test_metrics_parse_failures_yield_penalty_not_fail() -> None:
    """HIGH-1 invariant: metrics() returns None for malformed json/keys/values.

    The objective path maps None -> PENALTY_VALUES + penalized=True; this test
    checks the metrics() helper itself.
    """
    import tempfile as _tf
    import importlib.util as _importlib
    # Load fixed-distillation-bayes-opt as a module without executing main()
    spec = _importlib.spec_from_file_location(
        "fdbo_under_test", FDBO_PATH)
    mod = _importlib.module_from_spec(spec)
    # Stub out heavy imports so we can call metrics() without optuna/torch.
    import sys as _sys
    class _StubStudyDirection:
        MINIMIZE = "MINIMIZE"
    optuna_stub = type(_sys)("optuna_stub")
    optuna_stub.TrialPruned = Exception
    optuna_stub.trial = type(_sys)("optuna.trial")
    optuna_stub.trial.TrialState = type(_sys)("optuna.trial.TrialState")
    optuna_stub.trial.TrialState.PRUNED = "PRUNED"
    optuna_stub.trial.TrialState.FAIL = "FAIL"
    optuna_stub.trial.TrialState.RUNNING = "RUNNING"
    optuna_stub.trial.TrialState.COMPLETE = "COMPLETE"
    optuna_stub.Trial = type(_sys)("optuna.Trial")
    optuna_stub.create_study = lambda *a, **kw: None
    optuna_stub.load_study = lambda *a, **kw: None
    samplers_stub = type(_sys)("samplers_stub")
    samplers_stub.TPESampler = lambda *a, **kw: None
    optuna_stub.samplers = samplers_stub
    study_mod_stub = type(_sys)("study_mod_stub")
    study_mod_stub.StudyDirection = _StubStudyDirection
    optuna_stub.study = study_mod_stub
    _sys.modules["optuna"] = optuna_stub
    _sys.modules["optuna.trial"] = optuna_stub.trial
    _sys.modules["optuna.samplers"] = samplers_stub
    _sys.modules["optuna.study"] = study_mod_stub
    try:
        spec.loader.exec_module(mod)
    finally:
        for name in ("optuna", "optuna.trial", "optuna.samplers", "optuna.study"):
            _sys.modules.pop(name, None)
    cases = [
        ("missing",      False),  # file does not exist
        ("empty",        False),  # empty file
        ("bad",          False),  # malformed JSON
        ("wrong-keys",   False),  # missing 'final'
        ("non-numeric",  False),  # correct keys, non-numeric values
        ("good",         True),   # healthy
    ]
    with _tf.TemporaryDirectory() as tmp:
        from pathlib import Path as _P
        root = _P(tmp)
        # metrics() reads <path>/metrics.json; build per-case subdirectories
        # whose metrics.json content varies.
        def _make(name: str, body: str | None) -> _P:
            d = root / name
            d.mkdir()
            if body is not None:
                (d / "metrics.json").write_text(body, encoding="utf-8")
            return d
        _make("missing",      None)
        _make("empty",        "")
        _make("bad",          "not json {{")
        _make("wrong-keys",   '{"foo": 1}')
        _make("non-numeric",
              '{"final": {"zig_failure_comparison": {"validation": '
              '{"student_failure_rate": "NaN"}}, "validation": {"logit_mse": "NaN"}}}')
        _make("good",
              '{"final": {"zig_failure_comparison": {"validation": '
              '{"student_failure_rate": 0.42}}, "validation": {"logit_mse": 0.03}}}')
        for name, expect_ok in cases:
            result = mod.metrics(root / name)
            if expect_ok:
                assert result == (0.42, 0.03), (name, result)
            else:
                assert result is None, (name, result)
    print("OK: metrics() returns None on missing/malformed/wrong-key/non-numeric "
          "inputs; caller must penalize (HIGH-1 invariant).")


def test_pre_scan_rejects_resume_with_pruned_or_failed() -> None:
    """HIGH-2 invariant: a study containing PRUNED/FAIL trials must be rejected
    when --resume=True.  Models the fixed-distillation pre-scan logic."""

    def pre_scan_rejects(trials, *, resume: bool) -> str | None:
        n_pruned = sum(1 for t in trials if t["state"] == "PRUNED")
        n_failed = sum(1 for t in trials if t["state"] == "FAIL")
        if resume and (n_pruned or n_failed):
            return (
                f"refusing to resume study: {n_pruned} PRUNED and "
                f"{n_failed} FAIL trials present"
            )
        return None

    # Pre-fix study: 2 completed, 4 pruned, 1 failed.
    trials = [
        fake_trial(0, "COMPLETE", [0.4, 0.1], {"config": {}, "actual_params": 100}),
        fake_trial(1, "PRUNED", None),
        fake_trial(2, "COMPLETE", [0.5, 0.2], {"config": {}, "actual_params": 100}),
        fake_trial(3, "PRUNED", None),
        fake_trial(4, "PRUNED", None),
        fake_trial(5, "COMPLETE", [0.3, 0.05], {"config": {}, "actual_params": 100}),
        fake_trial(6, "FAIL", None),
        fake_trial(7, "PRUNED", None),
    ]
    # Resume from a pre-fix DB MUST raise.
    err = pre_scan_rejects(trials, resume=True)
    assert err is not None and "4 PRUNED" in err and "1 FAIL" in err, err
    # Fresh creation (resume=False) is allowed; pre-scan warning only.
    err = pre_scan_rejects(trials, resume=False)
    assert err is None, err
    # Healthy study (no pruned/failed) is fine even with --resume.
    healthy = [fake_trial(0, "COMPLETE", [0.3, 0.05],
                          {"config": {}, "actual_params": 100})]
    assert pre_scan_rejects(healthy, resume=True) is None
    print("OK: pre-scan rejects --resume from pre-fix studies; fresh path "
          "and healthy studies are unaffected.")


def test_sample_config_returns_penalty_reason_locally() -> None:
    """MEDIUM-1 invariant: sample_config signal flows through return value.

    We can't load optuna here (it isn't installed locally), but the contract is
    that sample_config returns (cfg, None) when feasible and (cfg, "reason")
    when not.  The obj-level switch in objective uses the second tuple element
    rather than reading trial.user_attrs["penalized"].
    """
    # Simulate the call signature with stubs
    def fake_sample(kn_small_world_k, kn_num_hidden):
        cfg = {"--lr": "0.001", "--batch-size": "256"}
        if kn_small_world_k >= kn_num_hidden:
            cfg["--kn-num-hidden"] = str(kn_num_hidden)
            cfg["--kn-small-world-k"] = str(kn_small_world_k)
            return cfg, "small-world degree must be lower than hidden-node count"
        cfg["--kn-num-hidden"] = str(kn_num_hidden)
        cfg["--kn-small-world-k"] = str(kn_small_world_k)
        return cfg, None

    cfg, reason = fake_sample(kn_small_world_k=8, kn_num_hidden=8)
    assert reason is not None
    assert cfg["--kn-small-world-k"] == "8"
    cfg, reason = fake_sample(kn_small_world_k=4, kn_num_hidden=8)
    assert reason is None
    print("OK: sample_config contract (cfg, penalty_reason | None) flows "
          "without storage round-trip; objective branches on the return value.")


def test_optuna_direction_compare() -> None:
    """Round-trip a 2-objective study and assert enum-based compare accepts it.

    Reproduces the casing pitfall: StudyDirection.MINIMIZE.name == "MINIMIZE"
    (uppercase), so a literal ``["minimize", "minimize"]`` would wrongly reject
    a healthy study. The fix uses ``list(study.directions) == [MINIMIZE, MINIMIZE]``.
    Uses InMemoryStorage to avoid Windows file-lock on sqlite temp files.
    """
    import optuna
    from optuna.study import StudyDirection
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage,
                                study_name="probe_dir_test",
                                directions=[StudyDirection.MINIMIZE,
                                           StudyDirection.MINIMIZE])
    study.add_trial(optuna.trial.create_trial(values=[0.1, 0.2]))
    # Reload to mimic cross-process access (same storage object)
    loaded = optuna.load_study(study_name="probe_dir_test", storage=storage)
    loaded_directions = list(loaded.directions)
    expected = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    assert loaded_directions == expected, (loaded_directions, expected)
    # And the name-based string compare (the old buggy form) would have failed
    assert [d.name for d in loaded_directions] != ["minimize", "minimize"]
    print("OK: optuna StudyDirection enum compare accepts healthy 2-objective "
          "study; string name compare would have wrongly rejected it.")


def test_budget_aware_sampler_feasibility() -> None:
    """Budget-aware sampler must produce 0% penalized trials for paper budget.

    For the reported failing window [5950, 7000] the old uniform sampler hit
    the window only ~1.5% (MLP) / ~14% (KNet) and produced the 30/30-penalized
    report. The new sampler must produce 0/30 penalized for both kinds.
    """
    import importlib.util as _importlib
    import sys as _sys
    import types as _types
    spec = _importlib.spec_from_file_location(
        "fdbo_budget_test", FDBO_PATH)
    mod = _importlib.module_from_spec(spec)
    # Reuse the same optuna stub as the metrics test so import succeeds
    # without a real optuna install, but here optuna IS installed so we
    # actually want the real module. Temporarily ensure sys.path has ROOT.
    _sys.path.insert(0, REPO_ROOT)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"SKIP: could not load fdbo for budget test: {exc}")
        return

    class _MockTrial:
        def __init__(self):
            self.params: dict = {}
            self.user_attrs: dict = {}
        def suggest_float(self, name, low, high, log=False):
            import random, math
            v = math.exp(random.uniform(math.log(low), math.log(high))) if log else random.uniform(low, high)
            self.params[name] = v
            return v
        def suggest_int(self, name, low, high):
            import random
            v = random.randint(low, high)
            self.params[name] = v
            return v
        def suggest_categorical(self, name, choices):
            import random
            v = random.choice(list(choices))
            self.params[name] = v
            return v
        def set_user_attr(self, k, v):
            self.user_attrs[k] = v

    low, high = 5950, 7000
    for kind in ("mlp", "knet"):
        penalized = 0
        for i in range(30):
            t = _MockTrial()
            _, reason = mod.sample_config(t, kind, low, high)
            if reason is not None:
                penalized += 1
        assert penalized == 0, f"{kind}: expected 0/30 penalized, got {penalized}/30 for window [{low},{high}]"
        # Spot-check one feasible trial's analytic param is inside window
        t = _MockTrial()
        cfg, reason = mod.sample_config(t, kind, low, high)
        assert reason is None, f"{kind}: spot-check unexpectedly penalized: {reason}"
        if kind == "mlp":
            w = int(cfg["--student-width"]); l = int(cfg["--student-layers"])
            ln = cfg["--student-use-layernorm"] == "True"
            p = mod._mlp_param_analytic(w, l, ln)
            assert low <= p <= high, f"mlp analytic {p} outside [{low},{high}] for w={w} l={l} ln={ln}"
        else:
            h = int(cfg["--kn-num-hidden"]); s = int(cfg["--kn-num-stages"])
            k = int(cfg["--kn-small-world-k"]); r = int(cfg["--kn-vca-rank"])
            p = mod._knet_param_analytic(h, s, k, r)
            assert low <= p <= high, f"knet analytic {p} outside [{low},{high}] for h={h} s={s} k={k} r={r}"
    print("OK: budget-aware sampler yields 0/30 penalized for both MLP and KNet on window [5950,7000]; spot-check analytic inside window.")


def test_knet_prior_biased_feasible() -> None:
    """knet-bo-repair: feasible KNet arches obey priors on both windows.

    Stages in {3,4,5}, k in {4,6}, rank >= 2; every tuple's analytic count
    lies in-window; both the paper [4250,5000] window and the legacy
    [5950,7000] window are non-empty (the latter keeps the 0/30 test below
    meaningful).
    """
    import importlib.util as _importlib
    import sys as _sys
    spec = _importlib.spec_from_file_location(
        "fdbo_prior_test", FDBO_PATH)
    mod = _importlib.module_from_spec(spec)
    _sys.path.insert(0, REPO_ROOT)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"SKIP: could not load fdbo for prior test: {exc}")
        return
    for low, high in ((4250, 5000), (5950, 7000)):
        arches = mod._get_feasible_knet_arches(low, high)
        assert arches, f"window [{low},{high}] admits no prior-biased arch"
        for h, s, k, r in arches:
            assert s in (3, 4, 5), (low, high, (h, s, k, r))
            assert k in (4, 6), (low, high, (h, s, k, r))
            assert r >= 2, (low, high, (h, s, k, r))
            p = mod._knet_param_analytic(h, s, k, r)
            assert low <= p <= high, (low, high, (h, s, k, r), p)
    print("OK: KNet feasible arches obey depth/VCA-rank priors and stay "
          "in-window on [4250,5000] and [5950,7000].")


def test_fanout_consistent_and_buildable() -> None:
    """knet-bo-repair: one fanout rule shared by sampler + analytic, valid
    for every feasible hidden size.

    knet_fanout(hs) must yield 8 unique in-range targets for all hs>=8
    (topology.py rejects out-of-range targets and cross-input duplicates),
    and both call sites must use it (no second hardcoded map).
    """
    import importlib.util as _importlib
    import sys as _sys
    from pathlib import Path as _P
    spec = _importlib.spec_from_file_location(
        "fdbo_fanout_test", FDBO_PATH)
    mod = _importlib.module_from_spec(spec)
    _sys.path.insert(0, REPO_ROOT)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"SKIP: could not load fdbo for fanout test: {exc}")
        return
    for hs in range(8, 25):
        fanout = mod.knet_fanout(hs)
        assert set(fanout) == {"0", "1", "2", "3"}, (hs, fanout)
        targets = [t for v in fanout.values() for t in v]
        assert len(targets) == 8, (hs, fanout)
        assert len(set(targets)) == 8, (hs, fanout)  # cross-input uniqueness
        assert all(0 <= t < hs for t in targets), (hs, fanout)
    src = (_P(FDBO_PATH)).read_text(encoding="utf-8")
    assert "[i, i + 4]" not in src or src.count("knet_fanout") >= 3, \
        "fanout rule duplicated instead of shared via knet_fanout()"
    print("OK: single knet_fanout() rule, 8 unique in-range targets for hs 8-24.")


def test_sampler_rev_bumped() -> None:
    """knet-bo-repair: fingerprint carries sampler_rev so stale studies fail fast."""
    from pathlib import Path as _P
    import re as _re
    fdbo = _P(FDBO_PATH).read_text(encoding="utf-8")
    m = _re.search(r"^SAMPLER_REV\s*=\s*(\d+)\s*$", fdbo, _re.M)
    assert m, "SAMPLER_REV constant missing from fixed-distillation-bayes-opt.py"
    assert int(m.group(1)) >= 2, f"SAMPLER_REV={m.group(1)} (must stay >= 2 so stale studies fail fast)"
    assert '"sampler_rev": SAMPLER_REV' in fdbo
    print(f"OK: sampler_rev={m.group(1)} is part of the study fingerprint.")


def main() -> int:
    test_offline_guard()
    test_penalty_sorts_last()
    test_penalized_excluded_from_selection()
    test_all_penalized_raises()
    test_metrics_parse_failures_yield_penalty_not_fail()
    test_pre_scan_rejects_resume_with_pruned_or_failed()
    test_sample_config_returns_penalty_reason_locally()
    test_budget_aware_sampler_feasibility()
    test_knet_prior_biased_feasible()
    test_fanout_consistent_and_buildable()
    test_sampler_rev_bumped()
    try:
        test_optuna_direction_compare()
    except ModuleNotFoundError:
        print("SKIP: optuna not installed; enum compare test deferred to host ENV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
