"""Locate a Phase-A checkpoint to resume a Kaggle rung.

Pure helpers: no copies, no torch. The notebook decides whether to copy the
chosen file into the run directory (Kaggle inputs are read-only) and prints
the hardware-config check result.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_phase_a_checkpoints(search_root, rung: str, seed: int) -> list[Path]:
    """Return ``phase_a_last.pt`` paths whose parent dir is exactly ``{rung}_seed{seed}``.

    Sorted newest mtime first. A longer rung name does not match a shorter one
    (``R1`` does not match ``R1d_seed100``). Missing roots yield an empty list.
    """
    root = Path(search_root)
    if not root.exists():
        return []
    want = f"{rung}_seed{int(seed)}"
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "phase_a_last.pt" not in filenames:
            continue
        parent = Path(dirpath)
        if parent.name != want:
            continue
        found.append(parent / "phase_a_last.pt")
    found.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return found


def compare_resume_hw_config(sibling_path, preset):
    """Compare a checkpoint's sibling ``hw_config.json`` to the rung preset.

    ``preset`` is the ``--hw-preset`` name the trainer will build (``None`` for
    R0, which passes no preset). ``sibling_path`` may be missing.

    Returns a message for the notebook to print. Raises ``ValueError`` when a
    present sibling disagrees with the preset — both hashes are in the message.
    For R0 the sibling, if present, must say hardware is off. A missing sibling
    returns a warning string and does not raise.

    The digest is ``HWConfig.hash()``, the same value logged as
    ``[hw] HWConfig ON hash=...``. At this commit that reproduces the abb9de7
    runs (``phase3a_core`` = ``eda59b5b6e0c``, ``phase3a_full`` =
    ``9c9fe3fe9ec3``). ``--kn-isat-max`` / ``--kn-gm-max`` /
    ``--kn-steps-per-stage`` are not part of ``HWConfig`` and do not affect it.
    """
    path = None if sibling_path is None else Path(sibling_path)
    if path is None or not path.is_file():
        where = f" ({path})" if path is not None else ""
        return (
            "WARNING: no sibling hw_config.json beside the external checkpoint"
            f"{where}; cannot verify the hardware preset before resume"
        )
    import argparse

    from hw import HWConfig, hw_config_from_args

    loaded = HWConfig.from_json(str(path))
    expected = hw_config_from_args(
        argparse.Namespace(hw_preset=preset or None, hw_config=None)
    )
    loaded_hash = loaded.hash()
    expected_hash = expected.hash()
    if preset in (None, "", "none"):
        if not loaded.is_off():
            raise ValueError(
                "external checkpoint sibling hw_config.json says hardware is ON "
                f"(hash={loaded_hash}) but this rung has no --hw-preset "
                f"(expected off, hash={expected_hash})"
            )
        return (
            "sibling hw_config.json matches R0 (hardware off): "
            f"sibling hash={loaded_hash} expected hash={expected_hash}"
        )
    if loaded_hash != expected_hash:
        raise ValueError(
            "external checkpoint sibling hw_config.json does not match "
            f"--hw-preset {preset}: sibling hash={loaded_hash} "
            f"preset hash={expected_hash}. Refusing to resume under a different "
            "hardware preset (parameter shapes match, so the trainer would "
            "otherwise load it silently)."
        )
    return (
        f"sibling hw_config.json matches --hw-preset {preset}: "
        f"sibling hash={loaded_hash} preset hash={expected_hash}"
    )

