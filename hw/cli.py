"""Entry-script glue for the hardware CLI (Phase 3b section 2.2).

The entry scripts (``dagger-nuance-distillation-kirchhoffnet.py``,
``train_script.py``, ``fixed-mlp-distillation-kirchhoffnet.py``) all do the
same four things with the hardware flags; this module holds them so they can be
unit-tested without running a script's heavy prologue:

``add_entry_hw_args(parser)``
    ``parse_hw_args`` (``--hw-config``, ``--hw-preset``, ``--hw-<flag>``,
    ``--hw-tm*``) plus the optional ``--zig-degrade-thr`` override.
``parse_known_args_hw_guarded(parser, argv)``
    ``parse_known_args`` that refuses leftover ``--hw-*`` / ``--zig-*``
    arguments.  A script that uses ``parse_known_args`` silently drops unknown
    flags; for a hardware flag that would train a *non*-hardware model while the
    log claims otherwise, so a typo (``--hw-slope-bnd``) or a flag from a newer
    branch is a hard error (``SystemExit``) instead.
``resolve_degrade_threshold(args, default)``
    ``--zig-degrade-thr`` when given, ``default`` otherwise.
``setup_hw_config(args, output_dir, logger)``
    build the ``HWConfig``, log its JSON and hash, write ``hw_config.json``.
``hw_passthrough_argv(args)``
    the hardware flags of ``args`` as an argv list, for drivers (the BO driver)
    that launch an entry script as a subprocess.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from . import HWConfig, hw_config_from_args

__all__ = ["GUARDED_PREFIXES", "add_entry_hw_args", "add_degrade_thr_arg",
           "check_unknown_hw_args", "parse_known_args_hw_guarded",
           "resolve_degrade_threshold", "setup_hw_config", "hw_passthrough_argv",
           "HW_CONFIG_FILENAME"]

#: leftover (unrecognised) arguments with these prefixes abort the run
GUARDED_PREFIXES = ("--hw-", "--zig-")
HW_CONFIG_FILENAME = "hw_config.json"


def add_degrade_thr_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--zig-degrade-thr", dest="zig_degrade_thr", type=float, default=None, metavar="T",
        help="optional override of ctle_metrics_config.DEGRADE_REL_THRESHOLD (0.10): the "
             "relative forward-error threshold of the failure rule.  Absent = the unified "
             "constant; only for threshold experiments (Phase 3b section 3.1)")
    return parser


def add_entry_hw_args(parser: argparse.ArgumentParser,
                      degrade_thr: bool = True) -> argparse.ArgumentParser:
    """Add every hardware flag (and, by default, ``--zig-degrade-thr``) to ``parser``."""
    from . import parse_hw_args
    parse_hw_args(parser)
    if degrade_thr:
        add_degrade_thr_arg(parser)
    return parser


def _flag_name(arg: str) -> str:
    return arg.split("=", 1)[0]


def check_unknown_hw_args(unknown: Iterable[str], prog: Optional[str] = None,
                          parser: Optional[argparse.ArgumentParser] = None) -> None:
    """Raise ``SystemExit`` if any leftover argument is a ``--hw-*`` / ``--zig-*`` flag."""
    bad = [a for a in (unknown or ()) if isinstance(a, str) and a.startswith(GUARDED_PREFIXES)]
    if not bad:
        return
    known: List[str] = []
    if parser is not None:
        for act in parser._actions:  # noqa: SLF001 - argparse has no public listing
            known += [o for o in act.option_strings if o.startswith(GUARDED_PREFIXES)]
    msg = (f"{prog or 'error'}: unrecognised hardware argument(s): "
           f"{' '.join(_flag_name(a) for a in bad)}.  This script parses its flags with "
           f"parse_known_args, so an unknown --hw-*/--zig-* flag would otherwise be "
           f"silently dropped and the run would NOT use the requested hardware model.")
    if known:
        msg += "  Known flags: " + " ".join(sorted(set(known)))
    raise SystemExit(msg)


def parse_known_args_hw_guarded(parser: argparse.ArgumentParser,
                                argv: Optional[Sequence[str]] = None
                                ) -> Tuple[argparse.Namespace, List[str]]:
    """``parser.parse_known_args(argv)`` that refuses unknown hardware flags."""
    args, unknown = parser.parse_known_args(argv)
    check_unknown_hw_args(unknown, prog=parser.prog, parser=parser)
    return args, unknown


def resolve_degrade_threshold(args, default: float) -> float:
    """``--zig-degrade-thr`` when given (must be > 0), else ``default``."""
    thr = getattr(args, "zig_degrade_thr", None)
    if thr is None:
        return float(default)
    thr = float(thr)
    if not thr > 0.0:
        raise SystemExit(f"--zig-degrade-thr must be > 0, got {thr!r}")
    return thr


def setup_hw_config(args, output_dir: Optional[str] = None,
                    logger: Optional[logging.Logger] = None,
                    filename: str = HW_CONFIG_FILENAME) -> HWConfig:
    """Build the ``HWConfig`` of ``args``, log its JSON and hash, save it to ``output_dir``.

    Always returns a config (possibly all-default); pass it through
    ``hw.hw_or_none`` before giving it to the builder.  Errors (missing preset,
    bad JSON key) propagate - they must never be demoted to a warning.
    """
    hw = hw_config_from_args(args)
    # no logger -> print (train_script.py reports through stdout)
    emit = logger.info if logger is not None else (lambda m: print(m, flush=True))
    state = "OFF (legacy path)" if hw.is_off() else "ON"
    emit(f"[hw] HWConfig {state} hash={hw.hash()} "
         f"preset={getattr(args, 'hw_preset', None)} "
         f"config={getattr(args, 'hw_config', None)}\n{hw.to_json()}")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        hw.to_json(path)
        emit(f"[hw] wrote {path}")
    return hw


def hw_passthrough_argv(args, include_degrade_thr: bool = True) -> List[str]:
    """The ``--hw-config`` / ``--hw-preset`` / ``--zig-degrade-thr`` of ``args`` as argv."""
    out: List[str] = []
    if getattr(args, "hw_config", None):
        out += ["--hw-config", os.path.abspath(str(args.hw_config))]
    if getattr(args, "hw_preset", None):
        out += ["--hw-preset", str(args.hw_preset)]
    if include_degrade_thr and getattr(args, "zig_degrade_thr", None) is not None:
        out += ["--zig-degrade-thr", repr(float(args.zig_degrade_thr))]
    return out
