import ast
import inspect
from pathlib import Path

import kirchhoff_net
import differential_stage


def test_forward_signatures_reject_no_repository_call_keywords():
    signatures = {
        "net": inspect.signature(kirchhoff_net.KirchhoffNetWithIO.forward),
        "core": inspect.signature(kirchhoff_net.KirchhoffNet.forward),
        "stage": inspect.signature(differential_stage.DifferentialStage.forward),
    }
    root = Path(__file__).resolve().parents[1]
    violations = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name not in signatures:
                continue
            accepted = set(signatures[name].parameters)
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in accepted:
                    violations.append(f"{path.name}:{node.lineno}: {name} got {kw.arg}")
    assert not violations, "unsupported forward kwargs:\n" + "\n".join(violations)
