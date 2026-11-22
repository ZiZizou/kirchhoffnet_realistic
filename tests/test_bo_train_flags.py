"""Every flag kn_bayes_opt.py can emit for train_script.py must exist there.

Regression guard (Phase 3b W5 step 1): the surgery removed train_script's
``--solver`` flag while kn_bayes_opt's EXTRA_FLAGS_FOR_PROBLEM still passed
``--solver heun`` for friedman1/2 and smooth2d, so every BO trial on those
datasets died in argparse.  The check is static: kn_bayes_opt.py is parsed
with ``ast`` (not imported), and every ``--flag`` string literal in
``_build_command``, ``EXTRA_FLAGS_FOR_PROBLEM`` and the train-side hw argv is
looked up in train_script's real argparse parser.
"""

import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_script_options() -> set:
    sys.path.insert(0, ROOT)
    from train_script import _add_argparse_args
    p = argparse.ArgumentParser()
    _add_argparse_args(p)
    return {o for a in p._actions for o in a.option_strings}


def _flag_literals(node) -> set:
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and n.value.startswith("--") and len(n.value) > 2}


def _bo_train_flags() -> dict:
    with open(os.path.join(ROOT, "kn_bayes_opt.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_command":
            found["_build_command"] = _flag_literals(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "EXTRA_FLAGS_FOR_PROBLEM"
                   for t in targets):
                found["EXTRA_FLAGS_FOR_PROBLEM"] = _flag_literals(node.value)
    # train-side hw argv: hw_argv_train = ["--hw-config", ...]
    hw = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "hw_argv_train"
                                             for t in n.targets):
            hw |= _flag_literals(n.value)
    found["hw_argv_train"] = hw
    return found


def test_bo_emits_only_existing_train_script_flags():
    opts = _train_script_options()
    found = _bo_train_flags()
    assert set(found) == {"_build_command", "EXTRA_FLAGS_FOR_PROBLEM", "hw_argv_train"}
    assert len(found["_build_command"]) > 20 and found["EXTRA_FLAGS_FOR_PROBLEM"]
    assert "--hw-config" in found["hw_argv_train"]
    missing = {src: sorted(f for f in flags if f not in opts) for src, flags in found.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"kn_bayes_opt passes flags train_script does not accept: {missing}"


def test_solver_flag_is_gone_from_both_sides():
    assert "--solver" not in _train_script_options()
    assert all("--solver" not in f for f in _bo_train_flags().values())
