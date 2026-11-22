"""Hardware layer for HR-KNet: units, retrain constraints, and the aggregate ``HWConfig``.

    HWConfig = {units: HWUnits, constraints: HWConstraints, timemux: Optional[TimeMuxConfig]}

``timemux`` is the ``TimeMuxConfig`` object of ``hw/timemux.py`` (``None`` when the
time-multiplexed fabric model is off); this module parses it from the JSON/CLI,
serialises it and hashes it.

CLI (Phase 3b 2.2): ``--hw-config PATH``, ``--hw-preset NAME`` and one ``--hw-<flag>`` per constraint
flag.  Precedence: CLI flag > ``--hw-config`` > preset > default.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .constraints import MISMATCH_DEFAULT, HWConstraints
from .timemux import XFER_MODES, TimeMuxConfig
from .units import HWUnits

__all__ = ["HWConfig", "HWConstraints", "HWUnits", "TimeMuxConfig", "parse_hw_args",
           "hw_config_from_args", "hw_or_none", "PRESET_DIR"]

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
_SECTIONS = ("units", "constraints", "timemux")


def _timemux_to_plain(tm: Any):
    if tm is None:
        return None
    if isinstance(tm, dict):
        return copy.deepcopy(tm)
    if hasattr(tm, "to_dict"):
        return tm.to_dict()
    if dataclasses.is_dataclass(tm):
        return dataclasses.asdict(tm)
    raise TypeError(f"cannot serialise timemux config of type {type(tm).__name__}")


def _timemux_from_any(tm: Any) -> Optional[TimeMuxConfig]:
    """Build a :class:`TimeMuxConfig` from whatever the JSON / the CLI gave us.

    ``None`` and an empty mapping both mean "no time-multiplexing model"
    (the section then stays ``None`` and not an empty config).
    """
    if tm is None:
        return None
    if isinstance(tm, TimeMuxConfig):
        return TimeMuxConfig.from_dict(tm.to_dict())
    if isinstance(tm, dict):
        if not tm:
            return None
        return TimeMuxConfig.from_dict(tm)
    raise TypeError(
        "timemux config must be a mapping or a TimeMuxConfig, got "
        f"{type(tm).__name__}"
    )


def _timemux_enabled(tm: Any) -> bool:
    """Opaque 'is the time-multiplexing model on' test: None, {} or enable=False are off."""
    if tm is None:
        return False
    plain = _timemux_to_plain(tm)
    if not plain:
        return False
    return bool(plain.get("enable", False))


@dataclass
class HWConfig:
    units: HWUnits = field(default_factory=HWUnits)
    constraints: HWConstraints = field(default_factory=HWConstraints)
    timemux: Optional[TimeMuxConfig] = None

    # ------------------------------------------------------------------
    def is_off(self) -> bool:
        """True only when every constraint flag is at its default and time-multiplexing is off."""
        return self.constraints.is_off() and not _timemux_enabled(self.timemux)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "units": {f.name: getattr(self.units, f.name) for f in fields(self.units)},
            "constraints": self.constraints.to_dict(),
            "timemux": _timemux_to_plain(self.timemux),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HWConfig":
        d = {} if d is None else d
        unknown = set(d) - set(_SECTIONS)
        if unknown:
            raise KeyError(f"HWConfig: unknown top-level keys {sorted(unknown)}; valid: {list(_SECTIONS)}")
        uf = {f.name for f in fields(HWUnits)}
        ud = d.get("units") or {}
        bad = set(ud) - uf
        if bad:
            raise KeyError(f"HWConfig.units: unknown keys {sorted(bad)}; valid: {sorted(uf)}")
        return cls(units=HWUnits(**ud),
                   constraints=HWConstraints.from_dict(d.get("constraints") or {}),
                   timemux=_timemux_from_any(d.get("timemux")))

    def to_json(self, path: Optional[str] = None, indent: Optional[int] = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(s + "\n")
        return s

    @classmethod
    def from_json(cls, src: str) -> "HWConfig":
        """``src`` is a path to a JSON file or a JSON string."""
        if os.path.isfile(src):
            with open(src) as fh:
                return cls.from_dict(json.load(fh))
        return cls.from_dict(json.loads(src))

    def hash(self) -> str:
        """Stable 12-hex digest of the full configuration (for study fingerprints)."""
        s = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def copy(self) -> "HWConfig":
        return HWConfig.from_dict(self.to_dict())


def hw_or_none(hw: Optional[HWConfig]) -> Optional[HWConfig]:
    """The builder rule of Phase 3b 2.1: an all-default config becomes ``None`` (legacy code path)."""
    if hw is None or hw.is_off():
        return None
    return hw


# ---------------------------------------------------------------------------
# presets / merge
# ---------------------------------------------------------------------------
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for sec, val in (over or {}).items():
        if sec not in _SECTIONS:
            raise KeyError(f"HW config: unknown section {sec!r}; valid: {list(_SECTIONS)}")
        if val is None:
            out[sec] = None
        elif isinstance(val, dict):
            cur = out.get(sec)
            out[sec] = {**(cur if isinstance(cur, dict) else {}), **copy.deepcopy(val)}
        else:
            out[sec] = copy.deepcopy(val)
    return out


def load_preset(name: str) -> dict:
    if name in (None, "", "none"):
        return {}
    path = os.path.join(PRESET_DIR, f"{name}.json")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json")) \
            if os.path.isdir(PRESET_DIR) else []
        raise FileNotFoundError(f"hw preset {name!r} not found at {path}; available: {have}")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on", "t", "y"):
        return True
    if s in ("0", "false", "no", "off", "f", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def _opt_int(v):
    return None if str(v).lower() in ("none", "null", "") else int(v)


def _opt_str(v):
    return None if str(v).lower() in ("none", "null", "") else str(v)


def _mismatch(v):
    """JSON dict, ``default`` (3a 3.9 sigmas), ``none``, or ``isat=0.05,g=0.1``."""
    s = str(v).strip()
    if s.lower() in ("none", "null", "off", ""):
        return None
    if s.lower() == "default":
        return dict(MISMATCH_DEFAULT)
    if s.startswith("{"):
        return json.loads(s)
    out = {}
    for part in s.split(","):
        k, _, val = part.partition("=")
        out[k.strip()] = float(val)
    return out


_TYPES = {
    "hw_vref_bits": _opt_int, "hw_xbar_bits": _opt_int, "hw_vca_bits": _opt_int,
    "hw_alpha_bwd": float, "hw_power_lambda": float, "hw_gln_delta_max": float,
    "hw_xbar_mode": str, "hw_c_node": _opt_str, "hw_gln_mode": _opt_str,
    "hw_mismatch_sigma": _mismatch,
}


def _flag_to_opt(name: str) -> str:
    return "--" + name.replace("_", "-")


def parse_hw_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the hardware flags to ``parser`` (all default to *unset* so precedence can be resolved)."""
    g = parser.add_argument_group("hardware constraints (all OFF by default)")
    g.add_argument("--hw-config", default=None, metavar="PATH",
                   help="JSON with units/constraints/timemux sections; unspecified keys keep defaults")
    g.add_argument("--hw-preset", default=None, metavar="NAME",
                   help="named bundle hw/presets/<NAME>.json (CLI flags and --hw-config override it)")
    for f in fields(HWConstraints):
        opt = _flag_to_opt(f.name)
        if f.name in _TYPES:
            g.add_argument(opt, dest=f.name, type=_TYPES[f.name], default=None, metavar="V",
                           help=f"HWConstraints.{f.name} (default {f.default!r})")
        else:
            g.add_argument(opt, dest=f.name, type=_str2bool, nargs="?", const=True, default=None,
                           metavar="BOOL", help=f"HWConstraints.{f.name} (default {f.default!r})")
    g.add_argument("--hw-noise-eval", dest="hw_mismatch_eval", type=_str2bool, nargs="?", const=True,
                   default=None, metavar="BOOL", help="alias of --hw-mismatch-eval")
    # -- the time-multiplexed fabric model (Phase 3b section 2.2) --
    g.add_argument("--hw-tm", dest="hw_tm", type=_str2bool, nargs="?", const=True,
                   default=None, metavar="BOOL",
                   help="enable the time-multiplexed fabric model (S&H, injection, droop, kT/C)")
    g.add_argument("--hw-tm-xfer-mode", dest="hw_tm_xfer_mode", default=None,
                   metavar="MODE", choices=list(XFER_MODES),
                   help="interstage transfer variant: " + ", ".join(XFER_MODES))
    g.add_argument("--hw-tm-param-bits", dest="hw_tm_param_bits", type=_opt_int,
                   default=None, metavar="N",
                   help="QAT bit width of the stage parameters (None = off)")
    g.add_argument("--hw-tm-c-hold-ff", dest="hw_tm_c_hold_ff", type=float,
                   default=None, metavar="F",
                   help="hold capacitance per sampled node [fF]")
    g.add_argument("--hw-tm-node-quant-bits", dest="hw_tm_node_quant_bits", type=_opt_int,
                   default=None, metavar="N",
                   help="node-voltage quantiser width at the boundary (None = off)")
    return parser


def hw_config_from_args(args) -> HWConfig:
    """Resolve ``--hw-preset`` < ``--hw-config`` < ``--hw-<flag>`` into an ``HWConfig``.

    The result may be all-default; use ``hw_or_none`` (or ``is_off()``) to decide on the legacy path."""
    merged: dict = {}
    preset = getattr(args, "hw_preset", None)
    if preset:
        merged = _merge(merged, load_preset(preset))
    cfg_path = getattr(args, "hw_config", None)
    if cfg_path:
        with open(cfg_path) as fh:
            merged = _merge(merged, json.load(fh))
    cli = {}
    for f in fields(HWConstraints):
        v = getattr(args, f.name, None)
        if v is not None:
            cli[f.name] = v
    if cli:
        merged = _merge(merged, {"constraints": cli})
    tm_cli = {}
    if getattr(args, "hw_tm", None) is not None:
        tm_cli["enable"] = bool(args.hw_tm)
    for _name in ("hw_tm_xfer_mode", "hw_tm_param_bits", "hw_tm_c_hold_ff",
                  "hw_tm_node_quant_bits"):
        _v = getattr(args, _name, None)
        if _v is not None:
            tm_cli[_name[len("hw_tm_"):]] = _v
    if tm_cli:
        merged = _merge(merged, {"timemux": tm_cli})
    return HWConfig.from_dict(merged)
