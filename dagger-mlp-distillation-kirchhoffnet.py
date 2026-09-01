"""CTLE Inverse Design: KirchhoffNet Student (DAgger) — PlainMLP Teacher.

Standalone training script mirroring ``dagger-nuance-distillation-kirchhoffnet.py``
but replacing the generative ConditionalSplineFlow+ZIG teacher with a frozen
PlainMLP teacher trained by ``generative-distillation-plain-mlp.py``. Single
deterministic MLP forward replaces candidate sampling; ZIG/HybridHurdleModel is
retained as the evaluation oracle (failure definition) and as the L_spec /
L_invalid component of :class:`RegimeAwareLoss`.

Defaults match the user's reference run:
    ``python generative-distillation-plain-mlp.py --param-budget 5559 \\
       --plain-trunk-layers 3 --input-preprocessing knet --seed 100 \\
       --output outputs/dagger_output_plain_mlp_w48_preprocessing_knet``
which derives W=48 L=3 SiLU (no LN, knet preprocessing) for ``~5479`` params
(:func:`plain_param_count`, ctle_dagger_common.py:171).

Student architecture (configured via the ``KN_*`` constants below) is identical
to ``dagger-nuance-distillation-kirchhoffnet.py``: a 3-stage KirchhoffNetWithIO
with small-world (k=4, p=0.2) hidden topology, 14 hidden nodes per stage,
edge_repeats=2, boundary fan-out map, temporal-readout, FreeTanhLibrary
non-programmable leak, freeze_read, residual-relu-tanh interstage activation,
and the same VCA gating flags.
"""

import os
import sys
import json
import time
import math
import logging
import subprocess
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths to edit for local runs (these default to Kaggle-mount paths).
# ---------------------------------------------------------------------------
DATA_DIR = '/home/annaik/Documents/augmented-cvae-ctle/'
DEFAULT_TEACHER_CKPT = 'outputs/dagger_output_plain_mlp_w48_preprocessing_knet/dagger_student_plain.pt'
MLP_TEACHER_CKPT = DEFAULT_TEACHER_CKPT
OUTPUT_DIR = '/home/annaik/Documents/dagger_output_mlp_knet'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint / resume helpers.
#
# Two complementary mechanisms:
#   1. ``dagger_checkpoint.pt`` — a rolling, atomic checkpoint saved at the
#      baseline (after initial dataset build) and at the end of every DAgger
#      iteration. Contains full dataset, student/optimizer/scheduler state,
#      histories, and RNG state. On startup it's loaded and the run resumes at
#      ``dagger_iter = dagger_iter + 1`` from the latest checkpoint.
#   2. ``initial_dataset_<hash>.pkl`` — a joblib cache of the initial distillation
#      dataset, keyed by a hash of the labeling-relevant hyperparams + df
#      fingerprint + MLP teacher checkpoint identity. Consulted only on a *fresh*
#      start (no checkpoint) so the most expensive single step is never re-run.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'dagger_checkpoint.pt')
_CKPT_VERSION = 1
INITIAL_DATASET_CACHE_DIR = None


def _mlp_teacher_identity():
    """Stable identifier (mtime, size, path) of the MLP teacher checkpoint.

    Used in :func:`_config_hash` so the initial-dataset cache is invalidated when
    the teacher checkpoint changes (vs being reused from a previous flow run).
    Returns ``(mtime, size, basename)`` or ``(0, 0, '<missing>')`` if missing.
    """
    try:
        st = os.stat(MLP_TEACHER_CKPT)
        return (int(st.st_mtime), int(st.st_size), os.path.basename(MLP_TEACHER_CKPT))
    except OSError:
        return (0, 0, '<missing>')


def _config_hash():
    """sha256 over labeling-relevant hyperparams + df fingerprint + MLP teacher id.

    Includes the MLP teacher checkpoint identity so cache hits only occur for
    the exact same teacher (prevents silent reuse of stale flow-labeled data).
    """
    try:
        df_n = int(len(df))
    except Exception:
        df_n = -1
    try:
        spec_cols = ['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']
        if 'df' in globals() and df_n > 0:
            spec_log = np.log10(np.clip(df[spec_cols].values, 1e-12, None))
            spec_min = np.round(spec_log.min(axis=0), 6).tolist()
            spec_max = np.round(spec_log.max(axis=0), 6).tolist()
        else:
            spec_min, spec_max = [], []
    except Exception:
        spec_min, spec_max = [], []
    payload = {
        'N_EMPIRICAL_SAMPLES': N_EMPIRICAL_SAMPLES,
        'BOUNDARY_RATIO': BOUNDARY_RATIO,
        'np_seed': _NP_SEED,
        'torch_seed': _TORCH_SEED,
        'mlp_teacher_ckpt': MLP_TEACHER_CKPT,
        'mlp_teacher_id': _mlp_teacher_identity(),
        'mlp_teacher_width': MLP_TEACHER_WIDTH,
        'mlp_teacher_layers': MLP_TEACHER_LAYERS,
        'mlp_teacher_activation': MLP_TEACHER_ACTIVATION,
        'mlp_teacher_layernorm': MLP_TEACHER_LAYERNORM,
        'mlp_teacher_input_preprocessing': MLP_TEACHER_INPUT_PREPROCESSING,
        'PARAM_COLS': list(PARAM_COLS),
        'df_n': df_n,
        'spec_min': spec_min,
        'spec_max': spec_max,
        'TEACHER_KIND': 'plain_mlp',
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    return h[:16]


def _initial_dataset_cache_path():
    cache_dir = INITIAL_DATASET_CACHE_DIR or OUTPUT_DIR
    return os.path.join(cache_dir, f'initial_dataset_mlp_{_config_hash()}.pkl')


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        return torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to load checkpoint at {CHECKPOINT_PATH}: {e}; ignoring")
        return None


def save_checkpoint(ckpt):
    tmp = CHECKPOINT_PATH + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, CHECKPOINT_PATH)


def _capture_rng():
    states = {
        'np': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            states['torch_cuda'] = torch.cuda.get_rng_state_all()
        except Exception:
            states['torch_cuda'] = None
    else:
        states['torch_cuda'] = None
    return states


def _restore_rng(states):
    if states is None:
        return
    try:
        np.random.set_state(states['np'])
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore numpy RNG state: {e}")
    try:
        torch.random.set_rng_state(states['torch'])
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore torch RNG state: {e}")
    cuda_states = states.get('torch_cuda')
    if cuda_states is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to restore cuda RNG state: {e}")


def _ckpt_baseline_payload(dagger_iter=0, converged=False):
    payload = {
        'format_version': _CKPT_VERSION,
        'dagger_iter': int(dagger_iter),
        'converged': bool(converged),
        'student_state': student.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'dataset': list(distillation_dataset.data),
        'train_indices': list(distillation_dataset._train_indices) if distillation_dataset._train_indices is not None else None,
        'val_indices': list(distillation_dataset._val_indices) if distillation_dataset._val_indices is not None else None,
        'hard_buffer_start_idx': int(distillation_dataset._hard_buffer_start_idx),
        'hard_indices': list(distillation_dataset._hard_indices),
        'dagger_history': {k: list(v) for k, v in dagger_history.items()},
        'loss_history': {k: list(v) for k, v in loss_history.items()},
        'rng': _capture_rng(),
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mlp_teacher_ckpt': MLP_TEACHER_CKPT,
    }
    if globals().get('COMMON_EVAL_SPECS') is not None:
        payload['common_eval_specs'] = np.asarray(globals()['COMMON_EVAL_SPECS']).tolist()
    return payload


def _restore_dataset_from_ckpt(ckpt):
    distillation_dataset.data = list(ckpt['dataset'])
    distillation_dataset._train_indices = list(ckpt['train_indices']) if ckpt['train_indices'] is not None else None
    distillation_dataset._val_indices = list(ckpt['val_indices']) if ckpt['val_indices'] is not None else None
    distillation_dataset._hard_buffer_start_idx = int(ckpt['hard_buffer_start_idx'])
    distillation_dataset._hard_indices = list(ckpt['hard_indices'])
    distillation_dataset._loader = None
    distillation_dataset._val_loader = None


def _restore_histories_from_ckpt(ckpt):
    for hist_name, target in (('dagger_history', dagger_history), ('loss_history', loss_history)):
        saved = ckpt.get(hist_name)
        if not isinstance(saved, dict):
            continue
        for k, v in saved.items():
            target[k] = list(v)


def _restore_common_eval_specs_from_ckpt(ckpt):
    if 'COMMON_EVAL_SPECS' not in globals():
        return
    saved = ckpt.get('common_eval_specs')
    if saved is None:
        return
    try:
        restored = np.asarray(saved, dtype=np.float32)
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore common_eval_specs: {e}; keeping rebuilt set")
        return
    if restored.ndim != 2 or restored.shape[1] != 4:
        _logger.warning(f"[CKPT] Restored common_eval_specs has unexpected shape "
                        f"{restored.shape}; keeping rebuilt set")
        return
    globals()['COMMON_EVAL_SPECS'] = restored
    _logger.info(f"[CKPT] Restored COMMON_EVAL_SPECS from checkpoint: "
                 f"shape={restored.shape}")


# ---------------------------------------------------------------------------
# Local KirchhoffNet codebase import.
# ---------------------------------------------------------------------------
_LOCAL_KIRCHHOFF_DIR = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_KIRCHHOFF_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_KIRCHHOFF_DIR)

# Common infra (ZIG/HybridHurdleModel, scalers, DistillationDataset,
# StudentEvaluator, RegimeAwareLoss, RampedRegimeLoss, helpers, MlpTeacherLabeler)
from ctle_dagger_common import (
    HybridHurdleModel,
    PlainMLP,
    RegimeAwareLoss,
    RampedRegimeLoss,
    DistillationDataset,
    StudentEvaluator,
    sample_validation_specs,
    filter_by_zig_validity,
    filter_by_zig_consistency,
    filter_by_zig_consistency_relaxed,
    empirical_fallback_label,
    is_boundary_spec,
    compute_forward_errors,
    compute_forward_errors_relaxed,
    log_failure_breakdown,
    log_label_quality_summary,
    PARAM_COLS,
    PARAM_LOG_BOUNDS,
    BOUNDARY_ABS_TOLERANCES,
    BOUNDARY_ABS_TOLERANCES_RELAXED,
    COL_MAPPING,
    DATASET_CSV_FILES,
    SPEC_RANGES,
    SPEC_INPUT_COLS,
    HPARAMS_DEFAULTS,
    LOG_BOUNDS,
    MlpTeacherLabeler,
    create_mlp_distillation_dataset,
)

from config import SOLVER, VCA
from cell_library import make_cell_library
from topology import build_net_from_config
from kirchhoff_net import format_parameter_breakdown


_NP_SEED = 42
_TORCH_SEED = 42

np.random.seed(_NP_SEED)
torch.manual_seed(_TORCH_SEED)

DEVICE = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

_LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
_DATE_FMT = "%H:%M:%S"


class _Tee:
    """File-like stream that mirrors writes to an original stream and a log file."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log_fh = log_fh

    def write(self, message):
        try:
            self._stream.write(message)
        except UnicodeEncodeError:
            enc = getattr(self._stream, "encoding", "ascii") or "ascii"
            self._stream.write(
                message.encode(enc, errors="replace").decode(enc, errors="replace")
            )
        try:
            self._log_fh.write(message)
            self._log_fh.flush()
        except (ValueError, OSError):
            pass
        try:
            self._stream.flush()
        except (ValueError, OSError, AttributeError):
            pass

    def flush(self):
        try:
            self._log_fh.flush()
        except (ValueError, OSError):
            pass
        try:
            self._stream.flush()
        except (ValueError, OSError, AttributeError):
            pass

    def isatty(self):
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", None)

    class _Buffer:
        def __init__(self, tee):
            self._tee = tee

        def write(self, data):
            if isinstance(data, (bytes, bytearray)):
                try:
                    msg = data.decode("utf-8", errors="replace")
                except Exception:
                    msg = data.decode("ascii", errors="replace")
            else:
                msg = data
            self._tee.write(msg)
            return len(data)

        def flush(self):
            self._tee.flush()

    @property
    def buffer(self):
        return self._Buffer(self)


_LOG_FILE = os.path.join(OUTPUT_DIR, "dagger_training.log")
_log_fh = open(_LOG_FILE, "a", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_DATE_FMT,
    force=True,
    stream=sys.stderr,
)
_logger = logging.getLogger("ctle_dagger_mlp_kirchhoff")

try:
    _git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8", errors="replace").strip()
except Exception:
    _git_commit = None

_logger.info("=" * 80)
_logger.info(
    f"=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} "
    f"(pid {os.getpid()}, platform={sys.platform}) ==="
)
if _git_commit:
    _logger.info(f"git_commit={_git_commit}")
_logger.info(f"log_file={_LOG_FILE}")
_logger.info("=" * 80)


class Timer:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger or _logger
        self.start = None
        self.elapsed = None
    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"[TIMER] {self.name} started")
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        self.logger.info(f"[TIMER] {self.name} done ({self.elapsed:.1f}s)")
    def __str__(self):
        return f"{self.elapsed:.1f}s" if self.elapsed is not None else "incomplete"


def _log_hyperparameters():
    _HYPERPARAMS = [
        ("MLP_TEACHER_CKPT", MLP_TEACHER_CKPT),
        ("MLP_TEACHER_WIDTH", MLP_TEACHER_WIDTH),
        ("MLP_TEACHER_LAYERS", MLP_TEACHER_LAYERS),
        ("MLP_TEACHER_ACTIVATION", MLP_TEACHER_ACTIVATION),
        ("MLP_TEACHER_LAYERNORM", MLP_TEACHER_LAYERNORM),
        ("MLP_TEACHER_INPUT_PREPROCESSING", MLP_TEACHER_INPUT_PREPROCESSING),
        ("DATA_DIR", DATA_DIR),
        ("OUTPUT_DIR", OUTPUT_DIR),
        ("DEVICE", str(DEVICE)),
        ("dataset_rows", len(globals()["df"])) if "df" in globals() else ("dataset_rows", "n/a"),
        ("DAGGER_ITERATIONS", DAGGER_ITERATIONS),
        ("EPOCHS_PER_ITER", EPOCHS_PER_ITER),
        ("BATCH_SIZE", BATCH_SIZE),
        ("ERROR_THRESHOLD", ERROR_THRESHOLD),
        ("VALIDATION_SIZE", VALIDATION_SIZE),
        ("BOUNDARY_RATIO", BOUNDARY_RATIO),
        ("VALIDITY_THRESHOLD", VALIDITY_THRESHOLD),
        ("DEGRADE_REL_THRESHOLD", DEGRADE_REL_THRESHOLD),
        ("MIN_DEGRADED_DIMS", MIN_DEGRADED_DIMS),
        ("LR_INITIAL", LR_INITIAL),
        ("LR_DECAY_AFTER_ITER", LR_DECAY_AFTER_ITER),
        ("LR_FLOOR", LR_FLOOR),
        ("FAILURE_CAP_RATIO", FAILURE_CAP_RATIO),
        ("CONVERGENCE_THRESHOLD", CONVERGENCE_THRESHOLD),
        ("EARLYSTOP_EVAL_EVERY", EARLYSTOP_EVAL_EVERY),
        ("EARLYSTOP_LOG_EVERY", EARLYSTOP_LOG_EVERY),
        ("EARLYSTOP_SKIP_EPOCHS", EARLYSTOP_SKIP_EPOCHS),
        ("EARLYSTOP_PATIENCE_EPOCHS", EARLYSTOP_PATIENCE_EPOCHS),
        ("COMMON_EVAL_SIZE", COMMON_EVAL_SIZE),
        ("COMMON_EVAL_SEED", COMMON_EVAL_SEED),
        ("MIN_FAILURE_IMPROVEMENT", MIN_FAILURE_IMPROVEMENT),
        ("DIVERGENCE_ABORT", DIVERGENCE_ABORT),
        ("DIVERGENCE_MARGIN", DIVERGENCE_MARGIN),
        ("DIVERGENCE_CONSEC_EVALS", DIVERGENCE_CONSEC_EVALS),
        ("N_EMPIRICAL_SAMPLES", N_EMPIRICAL_SAMPLES),
        ("WEIGHT_DECAY", WEIGHT_DECAY),
        ("LOSS_WEIGHT_EMPIRIC", LOSS_WEIGHT_EMPIRIC),
        ("HARD_BUFFER_WEIGHT", HARD_BUFFER_WEIGHT),
        ("ALPHA_SPEC", ALPHA_SPEC),
        ("BETA_PHYS", BETA_PHYS),
        ("GAMMA_MONO", GAMMA_MONO),
        ("ALPHA_INVALID", ALPHA_INVALID),
        ("K_MANIFOLD", K_MANIFOLD),
        ("ALPHA_MANIFOLD", ALPHA_MANIFOLD),
        ("WARMUP_EPOCHS", WARMUP_EPOCHS),
        ("KN_NUM_STAGES", KN_NUM_STAGES),
        ("KN_NUM_HIDDEN", KN_NUM_HIDDEN),
        ("KN_SMALL_WORLD_K", KN_SMALL_WORLD_K),
        ("KN_SMALL_WORLD_P", KN_SMALL_WORLD_P),
        ("KN_SMALL_WORLD_SEED", KN_SMALL_WORLD_SEED),
        ("KN_EDGE_REPEATS", KN_EDGE_REPEATS),
        ("KN_CELL_LIBRARY", KN_CELL_LIBRARY),
        ("KN_LEAK_MODE", KN_LEAK_MODE),
        ("KN_INTERSTAGE_ACTIVATION", KN_INTERSTAGE_ACTIVATION),
        ("KN_FREEZE_READ", KN_FREEZE_READ),
        ("KN_TEMPORAL_READOUT", KN_TEMPORAL_READOUT),
        ("KN_BOUNDARY_FAN_OUT", KN_BOUNDARY_FAN_OUT),
        ("KN_INPUT_RAIL", KN_INPUT_RAIL),
        ("KN_X_MAX", KN_X_MAX),
        ("KN_VCA", KN_VCA),
        ("KN_VCA_RANK", KN_VCA_RANK if KN_VCA_RANK is not None else int(VCA["rank"])),
        ("KN_VCA_CORE", KN_VCA_CORE),
        ("KN_VCA_GATE_SHUNT", KN_VCA_GATE_SHUNT),
        ("KN_VCA_SEPARATE_CORE_BUS", KN_VCA_SEPARATE_CORE_BUS),
        ("KN_INPUT_LOG_PAD_FRAC", KN_INPUT_LOG_PAD_FRAC),
        ("LOG_BOUNDS", LOG_BOUNDS),
    ]

    def _coerce(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    _logger.info("-" * 80)
    _logger.info("[CFG] Run hyperparameters:")
    _cfg_dict = {}
    for _name, _val in _HYPERPARAMS:
        _val_coerced = _coerce(_val)
        _cfg_dict[_name] = _val_coerced
        try:
            _val_repr = json.dumps(_val_coerced, default=str)
        except (TypeError, ValueError):
            _val_repr = repr(_val_coerced)
        _logger.info(f"[CFG] {_name}={_val_repr}")
    _logger.info("[CFG_JSON] " + json.dumps(_cfg_dict, default=str))
    _logger.info("-" * 80)


_logger.info(f"Using device: {DEVICE}")


# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
DAGGER_ITERATIONS     = 10
EPOCHS_PER_ITER       = 200
BATCH_SIZE            = 256
INITIAL_DATASET_CACHE_DIR_DEFAULT = None
ERROR_THRESHOLD       = 0.10
VALIDATION_SIZE       = 2000
BOUNDARY_RATIO        = 0.50
VALIDITY_THRESHOLD    = 0.50
DEGRADE_REL_THRESHOLD = 0.20
MIN_DEGRADED_DIMS     = 2
LR_INITIAL            = 1e-3
LR_DECAY_AFTER_ITER   = 3
LR_FLOOR              = 1e-4
MAPPER_LR_SCALE       = 1.0
STRUCT_LR_SCALE       = 4.0
DYN_LR_SCALE          = 1.0
FAILURE_CAP_RATIO     = 0.40
CONVERGENCE_THRESHOLD = 0.02
COMMON_EVAL_SIZE      = 2000
COMMON_EVAL_SEED      = 24681012
EARLYSTOP_EVAL_EVERY  = 1
EARLYSTOP_LOG_EVERY   = 10
EARLYSTOP_SKIP_EPOCHS = 5
EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER
MIN_FAILURE_IMPROVEMENT = 0.01
DIVERGENCE_ABORT          = True
DIVERGENCE_MARGIN         = 0.20
DIVERGENCE_CONSEC_EVALS   = 5
COMMON_EVAL_SPECS = None
N_EMPIRICAL_SAMPLES   = 20000
WEIGHT_DECAY          = 1e-4
LOSS_WEIGHT_EMPIRIC   = 1.0
HARD_BUFFER_WEIGHT    = 10.0

# Regime-aware loss weights
ALPHA_SPEC      = 1.5
BETA_PHYS       = 0.1
GAMMA_MONO      = 0.01
ALPHA_INVALID   = 1.5
K_MANIFOLD      = 5
ALPHA_MANIFOLD  = 0.1
WARMUP_EPOCHS   = 5

# ---------------------------------------------------------------------------
# PlainMLP teacher defaults (match user run: --param-budget 5559 --input-preprocessing knet)
# ---------------------------------------------------------------------------
MLP_TEACHER_WIDTH = 48
MLP_TEACHER_LAYERS = 3
MLP_TEACHER_ACTIVATION = "silu"
MLP_TEACHER_LAYERNORM = False
MLP_TEACHER_INPUT_PREPROCESSING = "knet"
MLP_TEACHER_BATCH_SIZE = 1024

# ---------------------------------------------------------------------------
# Local KirchhoffNet student topology (mirrors dagger-nuance-*.py defaults)
# ---------------------------------------------------------------------------
KN_NUM_STAGES       = 4
KN_NUM_HIDDEN       = 14
KN_SMALL_WORLD_K    = 4
KN_SMALL_WORLD_P    = 0.2
KN_SMALL_WORLD_SEED = 1
KN_EDGE_REPEATS     = 2
KN_CELL_LIBRARY     = "tanh_free"
KN_LEAK_MODE        = "non-programmable"
KN_INTERSTAGE_ACTIVATION = "residual-relu-tanh"
KN_FREEZE_READ          = True
KN_TEMPORAL_READOUT     = True
KN_BOUNDARY_FAN_OUT = '{"0": [2, 4], "1": [1, 3], "2": [12, 5], "3": [7, 9]}'
KN_INPUT_RAIL       = 4.0
KN_X_MAX            = 4.0

KN_VCA                   = True
KN_VCA_CORE              = True
KN_VCA_BIAS              = bool(VCA.get("bias", False))
KN_VCA_GATE_SHUNT        = False
KN_VCA_SEPARATE_CORE_BUS = True
KN_VCA_RANK              = 2

KN_INPUT_LOG_MIN_DEFAULT = np.array([np.log10(1e-4), np.log10(1.0), np.log10(0.5), np.log10(1.0)], dtype=np.float32)
KN_INPUT_LOG_MAX_DEFAULT = np.array([np.log10(1e-1), np.log10(1e3), np.log10(2e2), np.log10(5e2)], dtype=np.float32)
KN_INPUT_LOG_MIN = KN_INPUT_LOG_MIN_DEFAULT
KN_INPUT_LOG_MAX = KN_INPUT_LOG_MAX_DEFAULT
KN_INPUT_LOG_PAD_FRAC = 0.05
INITIAL_DATASET_CACHE_DIR = INITIAL_DATASET_CACHE_DIR_DEFAULT


# ---------------------------------------------------------------------------
# CLI overrides (matches dagger-nuance-*.py BO override surface).
# ---------------------------------------------------------------------------
try:
    import argparse as _argparse
    _bo_parser = _argparse.ArgumentParser(add_help=False)
    _bo_parser.add_argument('--dagger-iterations', type=int, default=None)
    _bo_parser.add_argument('--epochs-per-iter', type=int, default=None)
    _bo_parser.add_argument('--common-eval-size', type=int, default=None)
    _bo_parser.add_argument('--common-eval-seed', type=int, default=None)
    _bo_parser.add_argument('--earlystop-eval-every', type=int, default=None)
    _bo_parser.add_argument('--batch-size', type=int, default=None)
    _bo_parser.add_argument('--initial-dataset-cache-dir', type=str, default=None)
    _bo_parser.add_argument('--count-params-only', action='store_true',
                            help='Construct the student, print trainable parameter count, and exit.')
    _bo_parser.add_argument('--kn-num-stages', type=int, default=None)
    _bo_parser.add_argument('--kn-num-hidden', type=int, default=None)
    _bo_parser.add_argument('--kn-small-world-k', type=int, default=None)
    _bo_parser.add_argument('--kn-small-world-p', type=float, default=None)
    _bo_parser.add_argument('--kn-vca-rank', type=int, default=None)
    _bo_parser.add_argument('--vca-bias', action='store_true', default=False)
    _bo_parser.add_argument('--kn-x-max', type=float, default=None)
    _bo_parser.add_argument('--kn-gm-max', type=float, default=None)
    _bo_parser.add_argument('--kn-isat-max', type=float, default=None)
    _bo_parser.add_argument('--t-span', type=float, default=None)
    _bo_parser.add_argument('--boundary-fan-out', type=str, default=None)
    _bo_parser.add_argument('--lr', type=float, default=None)
    _bo_parser.add_argument('--weight-decay', type=float, default=None)
    _bo_parser.add_argument('--output', type=str, default=None)
    _bo_parser.add_argument('--device', type=str, default=None)
    _bo_parser.add_argument('--data-dir', type=str, default=None)
    _bo_parser.add_argument('--mlp-teacher-ckpt', type=str, default=None)
    _bo_parser.add_argument('--mlp-teacher-width', type=int, default=None)
    _bo_parser.add_argument('--mlp-teacher-layers', type=int, default=None)
    _bo_parser.add_argument('--mlp-teacher-activation', type=str, default=None,
                            choices=['silu', 'gelu'])
    _bo_parser.add_argument('--mlp-teacher-use-layernorm', action='store_true', default=False)
    _bo_parser.add_argument('--mlp-teacher-input-preprocessing', type=str, default=None,
                            choices=['knet', 'q75'])
    _bo_parser.add_argument('--mlp-teacher-batch-size', type=int, default=None)
    _bo_parser.add_argument('--seed', type=int, default=None)
    _bo_args, _ = _bo_parser.parse_known_args()
    if _bo_args.dagger_iterations is not None:
        DAGGER_ITERATIONS = _bo_args.dagger_iterations
    if _bo_args.epochs_per_iter is not None:
        EPOCHS_PER_ITER = _bo_args.epochs_per_iter
        EARLYSTOP_PATIENCE_EPOCHS = EPOCHS_PER_ITER
    if _bo_args.common_eval_size is not None:
        COMMON_EVAL_SIZE = _bo_args.common_eval_size
    if _bo_args.common_eval_seed is not None:
        COMMON_EVAL_SEED = _bo_args.common_eval_seed
    if _bo_args.earlystop_eval_every is not None:
        if _bo_args.earlystop_eval_every < 1:
            raise ValueError('--earlystop-eval-every must be >= 1')
        EARLYSTOP_EVAL_EVERY = _bo_args.earlystop_eval_every
    if _bo_args.batch_size is not None:
        if _bo_args.batch_size < 1:
            raise ValueError('--batch-size must be >= 1')
        BATCH_SIZE = _bo_args.batch_size
    if _bo_args.initial_dataset_cache_dir is not None:
        INITIAL_DATASET_CACHE_DIR = _bo_args.initial_dataset_cache_dir
        os.makedirs(INITIAL_DATASET_CACHE_DIR, exist_ok=True)
    if _bo_args.kn_num_stages is not None:
        KN_NUM_STAGES = _bo_args.kn_num_stages
    if _bo_args.kn_num_hidden is not None:
        KN_NUM_HIDDEN = _bo_args.kn_num_hidden
    if _bo_args.kn_small_world_k is not None:
        KN_SMALL_WORLD_K = _bo_args.kn_small_world_k
    if _bo_args.kn_small_world_p is not None:
        KN_SMALL_WORLD_P = _bo_args.kn_small_world_p
    if _bo_args.kn_vca_rank is not None:
        KN_VCA_RANK = _bo_args.kn_vca_rank
    if _bo_args.vca_bias:
        KN_VCA_BIAS = True
    if _bo_args.kn_x_max is not None:
        KN_X_MAX = _bo_args.kn_x_max
    if _bo_args.boundary_fan_out is not None:
        KN_BOUNDARY_FAN_OUT = _bo_args.boundary_fan_out
    if _bo_args.lr is not None:
        LR_INITIAL = _bo_args.lr
    if _bo_args.weight_decay is not None:
        WEIGHT_DECAY = _bo_args.weight_decay
    if _bo_args.output is not None:
        OUTPUT_DIR = _bo_args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'dagger_checkpoint.pt')
    if _bo_args.device is not None:
        DEVICE = torch.device(_bo_args.device) if _bo_args.device != 'auto' else DEVICE
    if _bo_args.data_dir is not None:
        DATA_DIR = _bo_args.data_dir
    if _bo_args.mlp_teacher_ckpt is not None:
        MLP_TEACHER_CKPT = _bo_args.mlp_teacher_ckpt
    if _bo_args.mlp_teacher_width is not None:
        MLP_TEACHER_WIDTH = _bo_args.mlp_teacher_width
    if _bo_args.mlp_teacher_layers is not None:
        MLP_TEACHER_LAYERS = _bo_args.mlp_teacher_layers
    if _bo_args.mlp_teacher_activation is not None:
        MLP_TEACHER_ACTIVATION = _bo_args.mlp_teacher_activation
    if _bo_args.mlp_teacher_use_layernorm:
        MLP_TEACHER_LAYERNORM = True
    if _bo_args.mlp_teacher_input_preprocessing is not None:
        MLP_TEACHER_INPUT_PREPROCESSING = _bo_args.mlp_teacher_input_preprocessing
    if _bo_args.mlp_teacher_batch_size is not None:
        MLP_TEACHER_BATCH_SIZE = _bo_args.mlp_teacher_batch_size
    if _bo_args.seed is not None:
        _NP_SEED = _bo_args.seed
        _TORCH_SEED = _bo_args.seed
        np.random.seed(_NP_SEED)
        torch.manual_seed(_TORCH_SEED)
    if _bo_args.t_span is not None:
        try:
            SOLVER["t_span"] = float(_bo_args.t_span)
        except Exception:
            pass
except Exception as _e:
    _logger.warning(f"[BO] override parsing failed: {_e}")


# Quick-exit flag for kn_bayes_opt.py param preflight; honored before any
# heavy artifact loads (ZIG/teacher/df). Set after _bo_args parsing so the
# student-construction code path can still complete and print the count.
_COUNT_PARAMS_ONLY = bool(getattr(_bo_args, 'count_params_only', False))


# ---------------------------------------------------------------------------
# Load teacher artifacts (ZIG + scalers) — skipped under --count-params-only.
# ---------------------------------------------------------------------------
_ZIG_ARTIFACT_DIR = None
zig_scaler_X = None
zig_scaler_y_p = None
zig_config = None
flow_scaler_X = None
scaler_X = None
scaler_y_p = None
zig_model = None
eye_scale_h = 0.0
eye_scale_w = 0.0
eye_scale_j = 0.0
PER_TARGET_HURDLE = False


def _resolve_zig_artifact_dir():
    candidates = [
        os.path.dirname(os.path.abspath(MLP_TEACHER_CKPT)),
        os.path.dirname(os.path.dirname(os.path.abspath(MLP_TEACHER_CKPT))),
        '/home/annaik/Documents/improved-zig-nf-spline-pytorch-default-v2/',
        '/home/annaik/Documents/augmented-cvae-ctle/',
    ]
    for cand in candidates:
        if not cand:
            continue
        for fname in ('hybrid_hurdle_ctle_model.pt',
                       'hybrid_hurdle_scaler_X.pkl',
                       'hybrid_hurdle_scaler_y_power.pkl',
                       'hybrid_hurdle_config.pkl'):
            if os.path.exists(os.path.join(cand, fname)):
                return cand
    return candidates[0] if candidates else '.'


def _load_zig_artifacts():
    """Load ZIG/HybridHurdleModel + scalers into the module-level globals."""
    global _ZIG_ARTIFACT_DIR, zig_scaler_X, zig_scaler_y_p, zig_config
    global scaler_X, scaler_y_p, eye_scale_h, eye_scale_w, eye_scale_j
    global PER_TARGET_HURDLE, zig_model, flow_scaler_X
    _ZIG_ARTIFACT_DIR = _resolve_zig_artifact_dir()
    _logger.info(f"Using ZIG artifact directory: {_ZIG_ARTIFACT_DIR}")
    flow_scaler_X_or_none = os.path.join(_ZIG_ARTIFACT_DIR, 'flow_scaler_X.pkl')
    if os.path.exists(flow_scaler_X_or_none):
        flow_scaler_X = joblib.load(flow_scaler_X_or_none)
    zig_scaler_X = joblib.load(os.path.join(_ZIG_ARTIFACT_DIR, 'hybrid_hurdle_scaler_X.pkl'))
    zig_scaler_y_p = joblib.load(os.path.join(_ZIG_ARTIFACT_DIR, 'hybrid_hurdle_scaler_y_power.pkl'))
    zig_config = joblib.load(os.path.join(_ZIG_ARTIFACT_DIR, 'hybrid_hurdle_config.pkl'))
    eye_scale_h = float(zig_config['eye_scale_h'])
    eye_scale_w = float(zig_config['eye_scale_w'])
    eye_scale_j = float(zig_config['eye_scale_j'])
    PER_TARGET_HURDLE = bool(zig_config.get('per_target_hurdle', False))
    scaler_X = zig_scaler_X
    scaler_y_p = zig_scaler_y_p
    _logger.info("Teacher (ZIG) artifacts loaded")
    _logger.info(f"  zig scaler_X mean: {zig_scaler_X.mean_[:3]}...")
    _logger.info(f"  zig scaler_y_p: mean={zig_scaler_y_p.mean_[0]:.4f}, scale={zig_scaler_y_p.scale_[0]:.4f}")
    _logger.info(f"  eye_scale_h (Q75): {eye_scale_h:.6f}")
    _logger.info(f"  eye_scale_w (Q75): {eye_scale_w:.6f}")
    _logger.info(f"  eye_scale_j (Q75): {eye_scale_j:.6f}")
    _logger.info(f"  per_target_hurdle: {PER_TARGET_HURDLE}")
    zig_model = HybridHurdleModel(dropout=0.0, per_target=PER_TARGET_HURDLE).to(DEVICE)
    zig_model.load_state_dict(torch.load(
        os.path.join(_ZIG_ARTIFACT_DIR, 'hybrid_hurdle_ctle_model.pt'),
        map_location=DEVICE,
    ))
    zig_model.eval()
    for p in zig_model.parameters():
        p.requires_grad = False
    _logger.info("HybridHurdleModel (ZIG) loaded from checkpoint")
    return zig_model


if not _COUNT_PARAMS_ONLY:
    _load_zig_artifacts()


# ===========================================================================
# Local KirchhoffNet student wrapper
# ===========================================================================
def _parse_boundary_fan_out(spec):
    if isinstance(spec, dict):
        return {int(k): list(v) for k, v in spec.items()}
    parsed = json.loads(spec)
    return {int(k): list(v) for k, v in parsed.items()}


class LocalKirchhoffStudentWrapper(nn.Module):
    def __init__(self,
                 num_stages: int = 4,
                 num_hidden: int = 14,
                 small_world_k: int = 4,
                 small_world_p: float = 0.2,
                 small_world_seed: int = 1,
                 edge_repeats: int = 2,
                 cell_library: str = "tanh_free",
                 leak_mode: str = "non-programmable",
                 interstage_activation: str = "residual-relu-tanh",
                 freeze_read: bool = True,
                 boundary_fan_out=None,
                 enable_temporal_readout: bool = True,
                 num_targets: int = 7,
                 input_rail: float = 4.0,
                 x_max: float = 4.0,
                 param_log_bounds=None,
                 input_log_min=None,
                 input_log_max=None,
                 stage_t_span: float | None = None,
                 stage_num_steps: int | None = None,
                 seed: int | None = None,
                 vca_enabled: bool = False,
                 vca_rank: int | None = None,
                 vca_core_enabled: bool = False,
                 vca_gate_shunt: bool = False,
                 vca_separate_core_bus: bool = False,
                 vca_bias: bool = False):
        super().__init__()

        if param_log_bounds is None:
            param_log_bounds = PARAM_LOG_BOUNDS
        if input_log_min is None:
            input_log_min = KN_INPUT_LOG_MIN
        if input_log_max is None:
            input_log_max = KN_INPUT_LOG_MAX

        self.num_stages = int(num_stages)
        self.num_hidden = int(num_hidden)
        self.small_world_k = int(small_world_k)
        self.small_world_p = float(small_world_p)
        self.small_world_seed = int(small_world_seed)
        self.edge_repeats = int(edge_repeats)
        self.cell_library = cell_library
        self.leak_mode = leak_mode
        self.interstage_activation = interstage_activation
        self.freeze_read = bool(freeze_read)
        self.boundary_fan_out = _parse_boundary_fan_out(boundary_fan_out)
        self.enable_temporal_readout = bool(enable_temporal_readout)
        self.num_targets = int(num_targets)
        self.input_rail = float(input_rail)
        self.x_max = float(x_max)
        self.param_log_bounds = param_log_bounds

        self._seed = int(seed if seed is not None else small_world_seed)

        self.vca_enabled = bool(vca_enabled)
        self.vca_rank = (
            int(vca_rank) if vca_rank is not None else int(VCA["rank"])
        )
        if self.vca_enabled and self.vca_rank < VCA["min_rank"]:
            raise ValueError(
                f"vca_rank must be >= {VCA['min_rank']}, got {self.vca_rank}"
            )
        self.vca_core_enabled = bool(vca_core_enabled)
        self.vca_gate_shunt = bool(vca_gate_shunt)
        self.vca_separate_core_bus = bool(vca_separate_core_bus)
        self.vca_bias = bool(vca_bias)

        self.register_buffer("input_log_min", torch.as_tensor(input_log_min, dtype=torch.float32))
        self.register_buffer("input_log_max", torch.as_tensor(input_log_max, dtype=torch.float32))
        self._clip_elements = 0
        self._input_elements = 0

        self.log_lo = nn.Parameter(torch.zeros(num_targets), requires_grad=False)
        self.log_hi = nn.Parameter(torch.zeros(num_targets), requires_grad=False)
        for i, (name, (lo, hi)) in enumerate(param_log_bounds.items()):
            self.log_lo.data[i] = lo
            self.log_hi.data[i] = hi

        t_span_stage = stage_t_span if stage_t_span is not None else SOLVER["t_span"] / num_stages
        num_steps_stage = stage_num_steps if stage_num_steps is not None else max(1, int(round(SOLVER["num_steps"] / num_stages)))
        cfg = {
            "stages": [
                {
                    "num_inputs": 4,
                    "num_hidden": num_hidden,
                    "num_proj": 0,
                    "num_outputs": 0,
                    "hidden_family": "small_world",
                    "hidden_kwargs": {
                        "k": int(small_world_k),
                        "p": float(small_world_p),
                        "seed": int(small_world_seed),
                        "bidirectional": False,
                    },
                    "input_pattern": "all_to_all",
                    "output_pattern": "all_to_all",
                    "proj_pattern": "all_to_all",
                    "edge_repeats": int(edge_repeats),
                    "t_span": float(t_span_stage),
                    "num_steps": int(num_steps_stage),
                }
                for _ in range(num_stages)
            ],
            "out_dim": int(num_targets),
            "write_mode": "sparse_proj",
            "read_mode": "dense",
            "use_robust_input": False,
        }
        cell_lib_template = make_cell_library(cell_library)
        self.net = build_net_from_config(
            cfg,
            cell_lib=cell_lib_template,
            leak_mode=leak_mode,
            freeze_read=freeze_read,
            interstage_activation=interstage_activation,
            boundary_fan_out=self.boundary_fan_out,
            enable_temporal_readout=enable_temporal_readout,
            x_max=self.x_max,
            vca_enabled=self.vca_enabled,
            vca_rank=self.vca_rank,
            vca_core_enabled=self.vca_core_enabled,
            vca_gate_shunt=self.vca_gate_shunt,
            vca_separate_core_bus=self.vca_separate_core_bus,
            vca_bias=self.vca_bias,
        )
        self._state_dim = num_hidden
        self._output_dim = int(num_targets)

    def scale_input(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        lo = self.input_log_min.to(x)
        hi = self.input_log_max.to(x)
        log_x = torch.log10(x.clamp(min=eps))
        span = (hi - lo).clamp(min=1e-8)
        u = 2.0 * (log_x - lo) / span - 1.0
        with torch.no_grad():
            self._clip_elements += int((u.abs() >= self.input_rail).sum().item())
            self._input_elements += int(u.numel())
        return u.clamp(min=-self.input_rail, max=self.input_rail)

    def clipping_stats(self) -> dict:
        clipped = self._clip_elements
        total = self._input_elements
        return {
            'clipped_elements': clipped,
            'input_elements': total,
            'clip_fraction': clipped / max(1, total),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.scale_input(x)
        logits, _trajs = self.net(u, store_trajectory=False, solver="heun")
        return logits

    def predict(self, x: torch.Tensor) -> dict:
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        results = {}
        for i, name in enumerate(self.param_log_bounds.keys()):
            log_val = self.log_lo[i] + (self.log_hi[i] - self.log_lo[i]) * probs[..., i]
            results[name] = torch.pow(10.0, log_val)
        return results

    def get_bounded_output(self, x: torch.Tensor):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        bounded_log = self.log_lo.unsqueeze(0) + (self.log_hi.unsqueeze(0) - self.log_lo.unsqueeze(0)) * probs
        physical = {name: torch.pow(10.0, bounded_log[:, i])
                    for i, name in enumerate(self.param_log_bounds.keys())}
        return logits, bounded_log, physical

    def extra_repr(self):
        return (f"num_stages={self.num_stages}, num_hidden={self.num_hidden}, "
                f"cell_library={self.cell_library}, leak_mode={self.leak_mode}, "
                f"freeze_read={self.freeze_read}, "
                f"interstage_activation={self.interstage_activation}, "
                f"temporal_readout={self.enable_temporal_readout}, "
                f"vca_enabled={self.vca_enabled}, vca_rank={self.vca_rank}, "
                f"vca_core={self.vca_core_enabled}, "
                f"vca_gate_shunt={self.vca_gate_shunt}, "
                f"vca_separate_core_bus={self.vca_separate_core_bus}, "
                f"out_dim={self._output_dim}")

    def count_params_by_component(self):
        bd = self.net.parameter_breakdown()
        groups = bd.get("groups", {})
        total = bd.get("total", 0)
        encoder = int(groups.get("input_mapper", 0))
        projector = int(groups.get("output_mapper", 0)) + int(groups.get("skip_linear", 0))
        drive = int(groups.get("drive_mappers", 0))
        circuit = int(total) - encoder - projector - drive
        return {
            "encoder": encoder,
            "circuit": circuit,
            "projector": projector,
            "total": int(total),
        }


# ===========================================================================
# Load historical data
# ===========================================================================
df = pd.DataFrame(columns=PARAM_COLS + ['power', 'stage_2_jitter',
                                         'stage_2_eye_max_height', 'stage_2_eye_max_width'])

if not _COUNT_PARAMS_ONLY:
    history_dir = DATA_DIR
    dfs = []
    for fname in DATASET_CSV_FILES:
        fpath = os.path.join(history_dir, fname)
        if not os.path.exists(fpath):
            _logger.warning(f"  Warning: {fname} not found, skipping")
            continue
        df = pd.read_csv(fpath)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True) if dfs else df
    _logger.info(f"Total rows: {len(combined)}")
    _logger.info(f"Columns: {list(combined.columns)[:10]}...")

    required_cols = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD',
                      'power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']
    combined = combined.rename(columns=COL_MAPPING)

    if all(c in combined.columns for c in required_cols):
        df = combined[required_cols].copy()
        mask = ((df['stage_2_jitter'] > 0) & (df['power'] > 0)
                & (df['stage_2_eye_max_height'] > 0) & (df['stage_2_eye_max_width'] > 0)
                & ~df.isna().any(axis=1))
        df = df[mask].reset_index(drop=True)
    else:
        _logger.warning(f"Missing required columns; have {list(combined.columns)}. Using empty df.")
        df = combined
    _logger.info(f"Filtered rows: {len(df)}")

    for col in ['power', 'stage_2_eye_max_height', 'stage_2_eye_max_width', 'stage_2_jitter']:
        nz = df[df[col] > 0][col] if len(df) else df[col]
        _logger.info(f"{col}: n={len(nz)}, min={nz.min() if len(nz) else 'n/a'}, "
                     f"max={nz.max() if len(nz) else 'n/a'}, median={nz.median() if len(nz) else 'n/a'}")

    if len(df) > 0:
        _spec_log = np.log10(np.clip(
            df[['power', 'stage_2_jitter', 'stage_2_eye_max_height', 'stage_2_eye_max_width']].values,
            1e-12, None,
        ))
        _lo = _spec_log.min(axis=0)
        _hi = _spec_log.max(axis=0)
        _pad = KN_INPUT_LOG_PAD_FRAC * np.maximum(_hi - _lo, 1e-8)
        KN_INPUT_LOG_MIN = (_lo - _pad).astype(np.float32)
        KN_INPUT_LOG_MAX = (_hi + _pad).astype(np.float32)
        for _i, _name in enumerate(['power', 'jitter', 'height', 'width']):
            _logger.info(
                f"KN input log bound [{_name}]: "
                f"[{10 ** KN_INPUT_LOG_MIN[_i]:.4e}, {10 ** KN_INPUT_LOG_MAX[_i]:.4e}] "
                f"(log10 [{KN_INPUT_LOG_MIN[_i]:.3f}, {KN_INPUT_LOG_MAX[_i]:.3f}])"
            )


# ===========================================================================
# Load PlainMLP teacher (skipped under --count-params-only)
# ===========================================================================
from ctle_dagger_common import load_plain_mlp_teacher, _set_active_context

teacher_model = None
teacher_labeler = None

if not _COUNT_PARAMS_ONLY:
    _logger.info(f"[MLP-TEACHER] Loading frozen PlainMLP teacher from {MLP_TEACHER_CKPT}")
    activation_cls = {"silu": nn.SiLU, "gelu": nn.GELU}[MLP_TEACHER_ACTIVATION]
    teacher_model = load_plain_mlp_teacher(
        MLP_TEACHER_CKPT, DEVICE,
        trunk_width=MLP_TEACHER_WIDTH,
        trunk_layers=MLP_TEACHER_LAYERS,
        activation=activation_cls,
        use_layernorm=MLP_TEACHER_LAYERNORM,
        input_preprocessing=MLP_TEACHER_INPUT_PREPROCESSING,
    )

    teacher_model.attach_scaler(
        scaler_p_scale=float(scaler_y_p.scale_[0]),
        scaler_p_mean=float(scaler_y_p.mean_[0]),
        eye_scale_j=eye_scale_j,
        eye_scale_h=eye_scale_h,
        eye_scale_w=eye_scale_w,
        input_preprocessing=MLP_TEACHER_INPUT_PREPROCESSING,
        input_log_min=KN_INPUT_LOG_MIN,
        input_log_max=KN_INPUT_LOG_MAX,
    )
    _logger.info(f"[MLP-TEACHER] attached scalers (input_preprocessing="
                 f"{MLP_TEACHER_INPUT_PREPROCESSING})")

    teacher_labeler = MlpTeacherLabeler(teacher_model, device=DEVICE, batch_size=MLP_TEACHER_BATCH_SIZE)
    _n_teacher = sum(p.numel() for p in teacher_model.parameters())
    _logger.info(f"[MLP-TEACHER] Frozen teacher loaded: {_n_teacher:,} params")

    # Populate ctle_dagger_common _active_ctx so helpers (filter_by_zig_*, etc.)
    # that default to _active_ctx work in this standalone script.
    try:
        _set_active_context({
            'DEVICE': DEVICE,
            'scaler_X': scaler_X,
            'scaler_y_p': scaler_y_p,
            'zig_model': zig_model,
            'eye_scale_h': eye_scale_h,
            'eye_scale_w': eye_scale_w,
            'eye_scale_j': eye_scale_j,
            'VALIDITY_THRESHOLD': VALIDITY_THRESHOLD,
            'ERROR_THRESHOLD': ERROR_THRESHOLD,
            'DEGRADE_REL_THRESHOLD': DEGRADE_REL_THRESHOLD,
            'MIN_DEGRADED_DIMS': MIN_DEGRADED_DIMS,
            'BOUNDARY_ABS_TOLERANCES': BOUNDARY_ABS_TOLERANCES,
            'df': df,
        })
        _logger.info("[CTX] _active_ctx populated for ctle_dagger_common helpers")
    except Exception as e:
        _logger.warning(f"[CTX] Failed to populate _active_ctx: {e}")


# ===========================================================================
# Eval epoch + per-group optimizer
# ===========================================================================
def eval_epoch(student, val_loader, criterion, device):
    student.eval()
    total_loss = 0
    losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0}
    with torch.no_grad():
        for specs, params_target in val_loader:
            specs = specs.to(device)
            params_target = params_target.to(device)
            spec_targets = {
                'power': specs[:, 0],
                'height': specs[:, 2],
                'width': specs[:, 3],
                'jitter': specs[:, 1],
            }
            loss_dict = criterion(student, spec_targets, params_target)
            total_loss += loss_dict['total'].item()
            for k in ('imit', 'spec', 'phys', 'invalid', 'manifold'):
                losses[k] += loss_dict[k]
    n = len(val_loader)
    return {k: float(v / n) for k, v in losses.items()}


# ===========================================================================
# Diagnostics
# ===========================================================================
def run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, device, n=200):
    _logger.info("=" * 60)
    _logger.info("  RUNNING DIAGNOSTICS")
    _logger.info("=" * 60)
    if len(df) == 0:
        _logger.info("No data — skipping diagnostics")
        _logger.info("=" * 60)
        return {'knn_yield': None, 'zig_real_acc': None}
    boundary_specs = sample_validation_specs(df, n_samples=n, boundary_ratio=1.0)
    _logger.info(f"1. MLP teacher validity yield on {n} boundary specs:")
    try:
        teacher_labels = teacher_labeler.label_batch(boundary_specs)
        keep_flow = filter_by_zig_validity(
            teacher_labels, zig_model, scaler_X, threshold=0.5, device=device
        )
        _logger.info(f"   MLP teacher yield: {keep_flow.sum()}/{n} ({keep_flow.mean() * 100:.1f}%)")
    except Exception as e:
        _logger.info(f"   MLP teacher error: {e}")
        keep_flow = None
    _logger.info(f"2. Empirical k-NN validity yield on {n} boundary specs (k=3):")
    knn_labels = np.array([empirical_fallback_label(s, df, k=3) for s in boundary_specs])
    keep_knn = filter_by_zig_validity(
        knn_labels, zig_model, scaler_X, threshold=0.5, device=device
    )
    _logger.info(f"   k-NN empirical yield: {keep_knn.sum()}/{n} ({keep_knn.mean() * 100:.1f}%)")
    _logger.info(f"3. ZIG validity on {min(1000, len(df))} real empirical rows:")
    real_subset = df.sample(min(1000, len(df)), random_state=42)
    real_params = real_subset[PARAM_COLS].values
    keep_real = filter_by_zig_validity(
        real_params, zig_model, scaler_X, threshold=0.5, device=device
    )
    _logger.info(f"   ZIG validity on real data: {keep_real.sum()}/{len(keep_real)} "
                 f"({keep_real.mean() * 100:.1f}%)")
    _logger.info("=" * 60)
    _logger.info("  DIAGNOSTICS COMPLETE")
    _logger.info("=" * 60)
    _logger.info("Interpretation:")
    _logger.info("  If (1) >50%:  teacher is fine -> focus on loss/replay")
    _logger.info("  If (1) <50%:  MLP teacher is weak -> use empirical fallback more")
    _logger.info("  If (2) <50%:  empirical data is weak -> need better training set")
    _logger.info("  If (3) <50%:  ZIG p_valid threshold may be too high (lower to 0.3)")
    return {
        'mlp_yield': keep_flow.mean() if keep_flow is not None else None,
        'knn_yield': keep_knn.mean(),
        'zig_real_acc': keep_real.mean(),
    }


def log_param_bound_feasibility(name, params_arr, log_lo=None, log_hi=None,
                                param_cols=PARAM_COLS, edge_eps=0.02):
    """Log per-param buckets of where teacher labels sit within [log_lo, log_hi]."""
    if params_arr is None or len(params_arr) == 0:
        _logger.info(f"[BOUNDS] {name}: no samples")
        return
    try:
        if log_lo is None or log_hi is None:
            lo_arr = np.array([PARAM_LOG_BOUNDS[k][0] for k in param_cols], dtype=np.float64)
            hi_arr = np.array([PARAM_LOG_BOUNDS[k][1] for k in param_cols], dtype=np.float64)
        else:
            lo_arr = np.asarray(log_lo, dtype=np.float64).flatten()
            hi_arr = np.asarray(log_hi, dtype=np.float64).flatten()
        if len(lo_arr) != len(param_cols) or len(hi_arr) != len(param_cols):
            raise ValueError(f"log_lo/log_hi length {len(lo_arr)}/{len(hi_arr)} != #param_cols {len(param_cols)}")
        params = np.asarray(params_arr, dtype=np.float64)
        if params.ndim != 2 or params.shape[1] != len(param_cols):
            raise ValueError(f"params_arr shape {params.shape} does not match #param_cols {len(param_cols)}")
        safe = np.clip(params, 1e-12, None)
        log_p = np.log10(safe)
        span = (hi_arr - lo_arr)
        if np.any(span <= 0):
            raise ValueError("non-positive log span in PARAM_LOG_BOUNDS")
        frac = (log_p - lo_arr) / span
        frac = np.nan_to_num(frac, nan=0.5, posinf=1.0, neginf=0.0)
        below = (frac < 0).mean(axis=0) * 100
        at_lo = ((frac >= 0) & (frac <= edge_eps)).mean(axis=0) * 100
        mid = ((frac > edge_eps) & (frac < 1 - edge_eps)).mean(axis=0) * 100
        at_hi = ((frac >= 1 - edge_eps) & (frac <= 1)).mean(axis=0) * 100
        above = (frac > 1).mean(axis=0) * 100
        _logger.info(f"[BOUNDS] {name} feasibility (frac in [log_lo, log_hi], edges={edge_eps:.2f}):")
        for i, pname in enumerate(param_cols):
            _logger.info(
                f"  {pname:>8}: below={below[i]:4.1f}%  at_lo={at_lo[i]:4.1f}%  "
                f"mid={mid[i]:5.1f}%  at_hi={at_hi[i]:4.1f}%  above={above[i]:4.1f}%"
            )
    except Exception as e:
        _logger.warning(f"[BOUNDS] Failed to compute feasibility for {name}: {e}")


def log_saturation_breakdown(metrics, prefix="  val/", param_cols=PARAM_COLS, edge_eps=0.02):
    """Log how close the student's sigmoid outputs are to the log_lo/log_hi edges."""
    try:
        probs = metrics.get('probs') if metrics else None
        logits = metrics.get('logits') if metrics else None
        if probs is None:
            _logger.info(f"{prefix}saturation: (no probs in metrics — skipping)")
            return
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] != len(param_cols):
            _logger.info(f"{prefix}saturation: unexpected probs shape {probs.shape} — skipping")
            return
        probs = np.nan_to_num(probs, nan=0.5)
        if logits is not None:
            logits = np.asarray(logits, dtype=np.float64)
            logits = np.nan_to_num(logits, nan=0.0)
            max_abs = np.abs(logits).max(axis=0)
        else:
            max_abs = np.full(len(param_cols), float('nan'))
        mean = probs.mean(axis=0)
        med = np.median(probs, axis=0)
        sat_lo = (probs <= edge_eps).mean(axis=0) * 100
        sat_hi = (probs >= 1 - edge_eps).mean(axis=0) * 100
        _logger.info(f"{prefix}saturation (sigmoid -> 0=log_lo, 1=log_hi, edges={edge_eps:.2f}):")
        for i, pname in enumerate(param_cols):
            _logger.info(
                f"  {pname:>8}: mean={mean[i]:.3f}  med={med[i]:.3f}  "
                f"sat_lo={sat_lo[i]:4.1f}%  sat_hi={sat_hi[i]:4.1f}%  max|logit|={max_abs[i]:.2f}"
            )
    except Exception as e:
        _logger.warning(f"{prefix}saturation logging failed: {e}")


def log_rail_probe(student, specs_arr, n=256, x_max=None, device=None):
    """Lightweight rail probe (re-added for parity with flow script; logs max|x|/x_max)."""
    if device is None:
        device = DEVICE
    if x_max is None:
        x_max = KN_X_MAX
    if student is None or specs_arr is None or len(specs_arr) == 0:
        _logger.info(f"[RAIL] no specs — skipping (x_max={x_max})")
        return
    out_dim = int(getattr(student.net, "output_ode_count", 0))
    try:
        subsample = specs_arr[: min(n, len(specs_arr))]
        specs_t = torch.from_numpy(subsample.astype(np.float32)).to(device)
        was_training = student.training
        student.eval()
        with torch.no_grad():
            u = student.scale_input(specs_t)
            _, trajs = student.net(u, store_trajectory=True, solver="heun")
        if was_training:
            student.train()
        if not trajs:
            _logger.info(f"[RAIL] no per-stage trajectories (x_max={x_max})")
            return
        _logger.info(f"[RAIL] node-voltage saturation vs x_max={x_max} (n={len(subsample)}):")
        for s, traj in enumerate(trajs):
            if traj is None:
                _logger.info(f"  stage {s}: (no trajectory)")
                continue
            v = traj.detach().abs()
            v_flat = torch.nan_to_num(v.reshape(-1), nan=0.0)
            max_v = v_flat.max().item()
            max_ratio = max_v / x_max if x_max > 0 else float('inf')
            frac = (v_flat > 0.9 * x_max).float().mean().item() * 100
            _logger.info(f"  stage {s}: max|x|/x_max={max_ratio:.2f}  frac(|x|>0.9*x_max)={frac:4.1f}%")
    except Exception as e:
        _logger.warning(f"[RAIL] probe failed: {e}")


# ===========================================================================
# Instantiate student (KNet)
# ===========================================================================
student = LocalKirchhoffStudentWrapper(
    num_stages=KN_NUM_STAGES,
    num_hidden=KN_NUM_HIDDEN,
    small_world_k=KN_SMALL_WORLD_K,
    small_world_p=KN_SMALL_WORLD_P,
    small_world_seed=KN_SMALL_WORLD_SEED,
    edge_repeats=KN_EDGE_REPEATS,
    cell_library=KN_CELL_LIBRARY,
    leak_mode=KN_LEAK_MODE,
    interstage_activation=KN_INTERSTAGE_ACTIVATION,
    freeze_read=KN_FREEZE_READ,
    boundary_fan_out=KN_BOUNDARY_FAN_OUT,
    enable_temporal_readout=KN_TEMPORAL_READOUT,
    num_targets=7,
    input_rail=KN_INPUT_RAIL,
    x_max=KN_X_MAX,
    param_log_bounds=PARAM_LOG_BOUNDS,
    seed=KN_SMALL_WORLD_SEED,
    vca_enabled=KN_VCA,
    vca_rank=KN_VCA_RANK,
    vca_core_enabled=KN_VCA_CORE,
    vca_gate_shunt=KN_VCA_GATE_SHUNT,
    vca_separate_core_bus=KN_VCA_SEPARATE_CORE_BUS,
    vca_bias=KN_VCA_BIAS,
).to(DEVICE)

_logger.info(
    f"Student VCA config: enabled={KN_VCA}, rank="
    f"{int(KN_VCA_RANK) if KN_VCA_RANK is not None else int(VCA['rank'])}, "
    f"core={KN_VCA_CORE}, bias={KN_VCA_BIAS}, gate_shunt={KN_VCA_GATE_SHUNT}, "
    f"separate_core_bus={KN_VCA_SEPARATE_CORE_BUS}"
)

_param_counts = student.count_params_by_component()
_logger.info(
    f"Student (Local KirchhoffNet): "
    f"encoder={_param_counts['encoder']}, "
    f"circuit={_param_counts['circuit']}, "
    f"projector={_param_counts['projector']}, "
    f"total={_param_counts['total']:,} params  "
    f"(num_stages={KN_NUM_STAGES}, num_hidden={KN_NUM_HIDDEN}, "
    f"small_world_k={KN_SMALL_WORLD_K}, small_world_p={KN_SMALL_WORLD_P}, "
    f"edge_repeats={KN_EDGE_REPEATS}, cell_library={KN_CELL_LIBRARY}, "
    f"leak_mode={KN_LEAK_MODE}, interstage_activation={KN_INTERSTAGE_ACTIVATION}, "
    f"freeze_read={KN_FREEZE_READ}, temporal_readout={KN_TEMPORAL_READOUT}, "
    f"input_rail={KN_INPUT_RAIL}, x_max={KN_X_MAX})"
)
_n_trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
_logger.info(f"Student trainable params: {_n_trainable:,}")
_bd = student.net.parameter_breakdown()
_logger.info(f"Student param breakdown:\n{format_parameter_breakdown(_bd)}")

if getattr(_bo_args, 'count_params_only', False):
    print(f"trainable params: {_n_trainable:,}", flush=True)
    raise SystemExit(0)


def _make_dagger_optimizer(student, lr, weight_decay):
    _mapper, _struct, _dyn, _other = [], [], [], []
    for _name, _p in student.named_parameters():
        if not _p.requires_grad:
            continue
        if "input_mapper" in _name or "output_mapper" in _name:
            _mapper.append(_p)
        elif _name.endswith(".z_logits"):
            _struct.append(_p)
        elif (
            _name.endswith(".vca_W")
            or _name.endswith(".vca_W_core")
            or _name.endswith(".vca_v_boundary")
            or _name.endswith(".vca_v_readout")
            or _name.endswith(".vca_v_core")
        ):
            _struct.append(_p)
        elif _name.endswith(".raw_leak") or _name.endswith(".raw_drive_g"):
            _dyn.append(_p)
        else:
            _other.append(_p)
    groups = []
    if _other:
        groups.append({"params": _other, "lr": lr, "lr_scale": 1.0})
    if _mapper:
        groups.append({"params": _mapper, "lr": lr * MAPPER_LR_SCALE, "lr_scale": MAPPER_LR_SCALE})
    if _struct:
        groups.append({"params": _struct, "lr": lr * STRUCT_LR_SCALE, "lr_scale": STRUCT_LR_SCALE})
    if _dyn:
        groups.append({"params": _dyn, "lr": lr * DYN_LR_SCALE, "lr_scale": DYN_LR_SCALE})
    _logger.info(f"[OPT] Param groups: other={sum(p.numel() for p in _other)} "
                 f"({len(_other)} tensors, lr={lr:.2e}); "
                 f"mapper={sum(p.numel() for p in _mapper)} "
                 f"({len(_mapper)} tensors, lr={lr * MAPPER_LR_SCALE:.2e}); "
                 f"struct={sum(p.numel() for p in _struct)} "
                 f"({len(_struct)} tensors, lr={lr * STRUCT_LR_SCALE:.2e}); "
                 f"dyn={sum(p.numel() for p in _dyn)} "
                 f"({len(_dyn)} tensors, lr={lr * DYN_LR_SCALE:.2e})")
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)


optimizer = _make_dagger_optimizer(student, LR_INITIAL, WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
)

# ---------------------------------------------------------------------------
# Checkpoint / initial dataset
# ---------------------------------------------------------------------------
ckpt = load_checkpoint()
distillation_dataset = None
if ckpt is not None:
    try:
        student.load_state_dict(ckpt['student_state'])
        student.to(DEVICE)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        if 'rng' in ckpt:
            _restore_rng(ckpt['rng'])
        distillation_dataset = DistillationDataset([])
        _restore_dataset_from_ckpt(ckpt)
        _restore_histories_from_ckpt(ckpt)
        _restore_common_eval_specs_from_ckpt(ckpt)
        _logger.info(
            f"[CKPT] Resumed from {CHECKPOINT_PATH}  "
            f"next_iter={int(ckpt['dagger_iter'])}, "
            f"dataset_size={len(ckpt.get('dataset', []))}, "
            f"format_version={ckpt.get('format_version')}"
        )
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to restore from checkpoint: {e}; starting fresh")
        ckpt = None
        distillation_dataset = None

if ckpt is not None:
    pass
else:
    _cache_path = _initial_dataset_cache_path()
    initial_data = None
    if os.path.exists(_cache_path):
        try:
            with Timer("load initial distillation dataset from cache"):
                initial_data = joblib.load(_cache_path)
            _logger.info(f"[CKPT] Initial dataset loaded from cache: {_cache_path} "
                         f"({len(initial_data)} samples, hash={os.path.basename(_cache_path)})")
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to load initial-dataset cache ({_cache_path}): "
                            f"{e}; rebuilding from scratch")
            initial_data = None
    if initial_data is None:
        with Timer("build initial MLP distillation dataset"):
            initial_data = create_mlp_distillation_dataset(
                df, n_samples=N_EMPIRICAL_SAMPLES,
                boundary_ratio=BOUNDARY_RATIO,
                teacher_labeler=teacher_labeler,
            )
        try:
            _cache_tmp = _cache_path + '.tmp'
            joblib.dump(initial_data, _cache_tmp)
            os.replace(_cache_tmp, _cache_path)
            _logger.info(f"[CKPT] Initial dataset cached at {_cache_path} "
                         f"(hash={os.path.basename(_cache_path)})")
        except Exception as e:
            _logger.warning(f"[CKPT] Failed to write initial dataset cache: {e}")
    _logger.info(f"Initial dataset (PlainMLP teacher labels): {len(initial_data)} samples")
    distillation_dataset = DistillationDataset(initial_data)
    if len(df) > 0:
        all_specs = np.array([[d['power'], d['jitter'], d['height'], d['width']]
                               for d in distillation_dataset.data])
        all_params = np.array([d['params'] for d in distillation_dataset.data])
        with Timer("ZIG forward-consistency check on initial dataset"):
            keep_strict = filter_by_zig_consistency(all_specs, all_params, zig_model, scaler_X, scaler_y_p, threshold=ERROR_THRESHOLD, device=DEVICE)
            keep_relaxed = filter_by_zig_consistency_relaxed(all_specs, all_params, zig_model, scaler_X, scaler_y_p, device=DEVICE)
        _logger.info(f"Initial dataset forward-consistency (strict): {keep_strict.mean() * 100:.1f}%  ({keep_strict.sum()}/{len(keep_strict)})")
        _logger.info(f"Initial dataset forward-consistency (relaxed): {keep_relaxed.mean() * 100:.1f}%  ({keep_relaxed.sum()}/{len(keep_relaxed)})")
        log_label_quality_summary("Initial dataset labels", all_specs, all_params, zig_model, scaler_X, scaler_y_p, device=DEVICE)
        log_param_bound_feasibility("Initial dataset", all_params)
    train_subset, val_subset = distillation_dataset.split_train_val(val_frac=0.1)
    try:
        save_checkpoint(_ckpt_baseline_payload(dagger_iter=0))
        _logger.info(f"[CKPT] Baseline checkpoint saved to {CHECKPOINT_PATH}")
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to write baseline checkpoint: {e}")

train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=HARD_BUFFER_WEIGHT)
val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)
_logger.info(f"Train: {len(distillation_dataset._train_indices)}, "
             f"Val: {len(distillation_dataset._val_indices)}")


# RampedRegimeLoss imported from ctle_dagger_common (deduplicated; local copy removed 2026-09-01)

criterion = RegimeAwareLoss(
    zig_model=zig_model,
    scaler_X=scaler_X,
    eye_scale_h=eye_scale_h,
    eye_scale_w=eye_scale_w,
    eye_scale_j=eye_scale_j,
    scaler_y_p=scaler_y_p,
    alpha_spec=ALPHA_SPEC,
    beta_phys=BETA_PHYS,
    gamma_mono=GAMMA_MONO,
    alpha_invalid=ALPHA_INVALID,
    empirical_df=df,
    k_manifold=K_MANIFOLD,
    alpha_manifold=ALPHA_MANIFOLD,
)
ramped_criterion = RampedRegimeLoss(criterion, warmup_epochs=WARMUP_EPOCHS)

_log_hyperparameters()

if ckpt is None and len(df) > 0:
    run_diagnostics(teacher_labeler, df, zig_model, scaler_X, scaler_y_p, DEVICE, n=200)
elif ckpt is not None:
    _logger.info("[CKPT] Skipping run_diagnostics on resume")

if COMMON_EVAL_SPECS is None and len(df) > 0:
    with Timer("build COMMON_EVAL_SPECS shared eval set"):
        COMMON_EVAL_SPECS = sample_validation_specs(
            df, n_samples=COMMON_EVAL_SIZE,
            boundary_ratio=BOUNDARY_RATIO, seed=COMMON_EVAL_SEED,
        )
    _logger.info(f"  COMMON_EVAL_SPECS: shape={COMMON_EVAL_SPECS.shape}, "
                 f"seed={COMMON_EVAL_SEED}")
elif COMMON_EVAL_SPECS is not None:
    _logger.info(f"  COMMON_EVAL_SPECS already set (shape="
                 f"{COMMON_EVAL_SPECS.shape}); reusing")
else:
    _logger.info("  COMMON_EVAL_SPECS: empty (no data)")
    COMMON_EVAL_SPECS = np.zeros((0, 4), dtype=np.float32)


# ===========================================================================
# DAgger history
# ===========================================================================
dagger_history = {
    'iteration': [], 'failure_rate': [], 'dataset_size': [], 'train_loss': [],
    'boundary_failure_rate': [], 'interior_failure_rate': [],
}
loss_history = {
    'iteration': [],
    'train_imit': [], 'train_spec': [], 'train_phys': [], 'train_invalid': [], 'train_total': [],
    'train_manifold': [],
    'val_imit': [],   'val_spec': [],   'val_phys': [],   'val_invalid': [],   'val_total': [],
    'val_manifold': [],
}

evaluator = StudentEvaluator(
    student=None, scaler_X=scaler_X, zig_model=zig_model,
    eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
    scaler_y_p=scaler_y_p, device=DEVICE,
)

# ===========================================================================
# Main DAgger loop
# ===========================================================================
_start_iter = int(ckpt['dagger_iter']) if ckpt is not None else 0
if ckpt is not None and ckpt.get('converged', False):
    _logger.info(f"[CKPT] Resumed checkpoint has converged=True; skipping training loop")
    _start_iter = DAGGER_ITERATIONS
if _start_iter >= DAGGER_ITERATIONS:
    _logger.info(f"[CKPT] Resumed dagger_iter={_start_iter} >= DAGGER_ITERATIONS="
                 f"{DAGGER_ITERATIONS}; skipping training loop")

for dagger_iter in range(_start_iter, DAGGER_ITERATIONS):
    _logger.info(f"{'=' * 60}")
    _logger.info(f"  DAgger Iteration {dagger_iter + 1}/{DAGGER_ITERATIONS}")
    _logger.info(f"{'=' * 60}")
    _logger.info(f"Dataset size: {len(distillation_dataset)}")

    train_loader = distillation_dataset.get_loader(batch_size=BATCH_SIZE, shuffle=True, hard_weight=HARD_BUFFER_WEIGHT)
    val_loader = distillation_dataset.get_val_loader(batch_size=BATCH_SIZE)

    best_val_loss = float('inf')
    best_state = None
    best_failure_state = None

    prev_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    if dagger_iter == 0 or len(COMMON_EVAL_SPECS) == 0:
        prev_failure_rate = 1.0
        _logger.info("  Prev-iter baseline: no prior model (iteration 1) -> prev_failure_rate=100.0%")
    else:
        evaluator.student = student
        _prev_mask, _ = evaluator.identify_failures(COMMON_EVAL_SPECS, threshold=ERROR_THRESHOLD)
        prev_failure_rate = float(_prev_mask.mean())
        _logger.info(f"  Prev-iter baseline: prev_failure_rate={prev_failure_rate * 100:.2f}% "
                     f"on COMMON_EVAL_SPECS (n={len(COMMON_EVAL_SPECS)})")
    best_failure_rate = prev_failure_rate

    epoch_losses = {'total': [], 'imit': [], 'spec': [], 'phys': [], 'invalid': [], 'manifold': []}
    n_nonfinite_batches = 0
    _div_consec = 0
    diverged_at = None

    for epoch in range(EPOCHS_PER_ITER):
        ramped_criterion.set_epoch(epoch)
        student.train()
        losses = {'total': 0, 'imit': 0, 'spec': 0, 'phys': 0, 'invalid': 0, 'manifold': 0}
        for specs_batch, params_batch in train_loader:
            specs_batch = specs_batch.to(DEVICE)
            params_batch = params_batch.to(DEVICE)
            spec_targets = {
                'power': specs_batch[:, 0],
                'jitter': specs_batch[:, 1],
                'height': specs_batch[:, 2],
                'width': specs_batch[:, 3],
            }
            optimizer.zero_grad()
            logits = student(specs_batch)
            base_loss_dict = ramped_criterion.base_loss(student, spec_targets, params_batch, logits=logits)
            ramped_loss_dict = ramped_criterion(student, spec_targets, params_batch,
                                                logits=logits, loss_dict=base_loss_dict)
            total_loss = ramped_loss_dict['total']
            if not total_loss.requires_grad or not torch.isfinite(total_loss):
                losses['total'] += float(total_loss)
                n_nonfinite_batches += 1
            else:
                total_loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0))
                if math.isfinite(grad_norm):
                    optimizer.step()
                else:
                    n_nonfinite_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    if n_nonfinite_batches == 1 or n_nonfinite_batches % 50 == 0:
                        _logger.warning(
                            f"    [grad-guard] non-finite grad norm at epoch {epoch + 1}; "
                            f"skipped optimizer step ({n_nonfinite_batches} skipped this iteration)"
                        )
                losses['total'] += total_loss.item()
            for k in ('imit', 'spec', 'phys', 'invalid', 'manifold'):
                losses[k] += ramped_loss_dict[k]

        n = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] += [losses[k] / n]

        val_losses = eval_epoch(student, val_loader, ramped_criterion, DEVICE)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _logger.info(
                f"  Epoch {epoch + 1:3d}/{EPOCHS_PER_ITER}  "
                f"train={losses['total'] / n:.4f}(imit={losses['imit'] / n:.3f},"
                f"spec={losses['spec'] / n:.3f},phys={losses['phys'] / n:.3f},"
                f"invalid={losses['invalid'] / n:.3f},manifold={losses['manifold'] / n:.3f})  "
                f"val={val_losses['total']:.4f}(imit={val_losses['imit']:.3f},"
                f"spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},"
                f"invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        if (epoch + 1) % EARLYSTOP_EVAL_EVERY == 0 or epoch == EPOCHS_PER_ITER - 1:
            if len(COMMON_EVAL_SPECS) > 0:
                evaluator.student = student
                failure_mask_check, metrics_check = evaluator.identify_failures(
                    COMMON_EVAL_SPECS, threshold=ERROR_THRESHOLD
                )
                current_failure_rate = failure_mask_check.mean()
                if (epoch + 1) > EARLYSTOP_SKIP_EPOCHS:
                    if current_failure_rate < best_failure_rate - MIN_FAILURE_IMPROVEMENT:
                        best_failure_rate = current_failure_rate
                        best_failure_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
                        _logger.info(f"    earlystop@{epoch + 1:3d}: failure_rate="
                                     f"{current_failure_rate * 100:.2f}% (new best, "
                                     f"prev={prev_failure_rate * 100:.2f}%, "
                                     f"delta={(prev_failure_rate - current_failure_rate) * 100:+.2f}pts)")
                    else:
                        _logger.info(f"    earlystop@{epoch + 1:3d}: failure_rate="
                                     f"{current_failure_rate * 100:.2f}% (no accept; "
                                     f"best={best_failure_rate * 100:.2f}%, "
                                     f"min_delta={MIN_FAILURE_IMPROVEMENT * 100:.1f}%)")
                    if DIVERGENCE_ABORT and (
                            current_failure_rate >= prev_failure_rate + DIVERGENCE_MARGIN):
                        _div_consec += 1
                        if _div_consec >= DIVERGENCE_CONSEC_EVALS:
                            diverged_at = epoch + 1
                            _logger.warning(
                                f"    [divergence] failure_rate >= "
                                f"{(prev_failure_rate + DIVERGENCE_MARGIN) * 100:.1f}% for "
                                f"{DIVERGENCE_CONSEC_EVALS} consecutive evals; aborting "
                                f"iteration {dagger_iter + 1} at epoch {epoch + 1}"
                            )
                            break
                    else:
                        _div_consec = 0
                else:
                    _logger.info(f"    earlystop@{epoch + 1:3d}: failure_rate="
                                 f"{current_failure_rate * 100:.2f}% (skipped, "
                                 f"epoch {epoch + 1} <= EARLYSTOP_SKIP_EPOCHS="
                                 f"{EARLYSTOP_SKIP_EPOCHS})")
                if (epoch + 1) % EARLYSTOP_LOG_EVERY == 0 or epoch == EPOCHS_PER_ITER - 1:
                    log_failure_breakdown(metrics_check, prefix="    earlystop/")

    iter_outcome = "keep_new"
    if (best_failure_state is not None
            and best_failure_rate < prev_failure_rate - MIN_FAILURE_IMPROVEMENT):
        student.load_state_dict(best_failure_state)
        student.to(DEVICE)
        _logger.info(f"  Carry-forward: restored best-failure-rate model "
                     f"({best_failure_rate * 100:.2f}% on COMMON_EVAL_SPECS, "
                     f"prev={prev_failure_rate * 100:.2f}%, "
                     f"delta={(prev_failure_rate - best_failure_rate) * 100:+.2f}pts)")
    elif prev_state is not None:
        student.load_state_dict(prev_state)
        student.to(DEVICE)
        iter_outcome = "fallback_prev"
        _logger.warning(
            f"  Carry-forward: iteration {dagger_iter + 1} no >="
            f"{MIN_FAILURE_IMPROVEMENT * 100:.1f}% improvement "
            f"(prev={prev_failure_rate * 100:.2f}% -> "
            f"best={best_failure_rate * 100:.2f}%); kept carried-forward model"
        )
    elif best_state is not None:
        student.load_state_dict(best_state)
        student.to(DEVICE)
        iter_outcome = "fallback_valloss"
        _logger.warning(f"  Carry-forward: no prev_state and no accepted "
                        f"best_failure_state; falling back to val-loss best "
                        f"(val={best_val_loss:.4f})")
    else:
        iter_outcome = "keep_last"
        _logger.warning("  Carry-forward: NO checkpoint captured this iteration; "
                        "keeping last-epoch student")

    best_loss = float(best_val_loss)

    val_specs_arr = COMMON_EVAL_SPECS
    if len(val_specs_arr) == 0:
        failure_rate = 0.0
        boundary_failure_rate = 0.0
        interior_failure_rate = 0.0
        failed_specs_arr = np.empty((0, 4), dtype=np.float32)
        metrics = {}
    else:
        evaluator.student = student
        failure_mask, metrics = evaluator.identify_failures(val_specs_arr, threshold=ERROR_THRESHOLD)
        failure_errors = metrics['errors'].max(axis=1)
        failed_specs_arr = val_specs_arr[failure_mask]
        failure_rate = failure_mask.mean()
        if 'boundary_failure' in metrics and 'is_boundary' in metrics:
            boundary_mask_eval = metrics['is_boundary']
            interior_mask_eval = ~boundary_mask_eval
            boundary_failure_rate = metrics['boundary_failure'].sum() / max(1, boundary_mask_eval.sum())
            interior_failure_rate = metrics['interior_failure'].sum() / max(1, interior_mask_eval.sum())
        else:
            boundary_failure_rate = failure_rate
            interior_failure_rate = failure_rate
        _logger.info(f"Validation failure rate: {failure_rate * 100:.1f}%  "
                     f"({failure_mask.sum()}/{len(val_specs_arr)} specs)")
        _logger.info(f"  boundary: {boundary_failure_rate * 100:.1f}%  "
                     f"interior: {interior_failure_rate * 100:.1f}%")
        log_failure_breakdown(metrics, prefix="  val/")
        log_saturation_breakdown(metrics, prefix="  val/")
        log_param_bound_feasibility("Rolling dataset", np.array([d['params'] for d in distillation_dataset.data]))
        try:
            log_rail_probe(student, val_specs_arr)
        except Exception as e:
            _logger.warning(f"[RAIL] log_rail_probe failed: {e}")

    iter_delta = (prev_failure_rate - failure_rate) * 100
    if diverged_at is not None:
        iter_outcome += f" (divergence-abort@ep{diverged_at})"
    if n_nonfinite_batches > 0:
        iter_outcome += f" (skipped {n_nonfinite_batches} non-finite batches)"
    _logger.info(
        f"  Iter {dagger_iter + 1} final={failure_rate * 100:.2f}% "
        f"vs prev={prev_failure_rate * 100:.2f}% "
        f"(delta {iter_delta:+.2f} pts) -> {iter_outcome}"
    )

    dagger_history['iteration'].append(dagger_iter + 1)
    dagger_history['failure_rate'].append(failure_rate)
    dagger_history['boundary_failure_rate'].append(boundary_failure_rate)
    dagger_history['interior_failure_rate'].append(interior_failure_rate)
    dagger_history['dataset_size'].append(len(distillation_dataset))
    dagger_history['train_loss'].append(best_loss)

    avg_train_imit = sum(epoch_losses['imit']) / len(epoch_losses['imit']) if epoch_losses['imit'] else 0.0
    avg_train_spec = sum(epoch_losses['spec']) / len(epoch_losses['spec']) if epoch_losses['spec'] else 0.0
    avg_train_phys = sum(epoch_losses['phys']) / len(epoch_losses['phys']) if epoch_losses['phys'] else 0.0
    avg_train_invalid = sum(epoch_losses['invalid']) / len(epoch_losses['invalid']) if epoch_losses['invalid'] else 0.0
    avg_train_total = sum(epoch_losses['total']) / len(epoch_losses['total']) if epoch_losses['total'] else 0.0
    avg_train_manifold = sum(epoch_losses['manifold']) / len(epoch_losses['manifold']) if epoch_losses['manifold'] else 0.0

    loss_history['iteration'].append(dagger_iter + 1)
    loss_history['train_imit'].append(avg_train_imit)
    loss_history['train_spec'].append(avg_train_spec)
    loss_history['train_phys'].append(avg_train_phys)
    loss_history['train_invalid'].append(avg_train_invalid)
    loss_history['train_total'].append(avg_train_total)
    loss_history['train_manifold'].append(avg_train_manifold)
    loss_history['val_imit'].append(val_losses['imit'])
    loss_history['val_spec'].append(val_losses['spec'])
    loss_history['val_phys'].append(val_losses['phys'])
    loss_history['val_invalid'].append(val_losses['invalid'])
    loss_history['val_total'].append(val_losses['total'])
    loss_history['val_manifold'].append(val_losses['manifold'])

    _logger.info(f"  Iter {dagger_iter + 1} loss summary: train_total={avg_train_total:.3f} "
                 f"(imit={avg_train_imit:.3f},spec={avg_train_spec:.3f},"
                 f"phys={avg_train_phys:.3f},invalid={avg_train_invalid:.3f},"
                 f"manifold={avg_train_manifold:.3f})  "
                 f"val_total={val_losses['total']:.3f}(imit={val_losses['imit']:.3f},"
                 f"spec={val_losses['spec']:.3f},phys={val_losses['phys']:.3f},"
                 f"invalid={val_losses['invalid']:.3f},manifold={val_losses['manifold']:.3f})")

    converged = False
    skip_labeling = False
    if failure_rate < CONVERGENCE_THRESHOLD:
        _logger.info(f"CONVERGENCE MET (failure_rate={failure_rate * 100:.2f}% < "
                     f"{CONVERGENCE_THRESHOLD * 100}%)")
        converged = True
    if not converged and len(failed_specs_arr) == 0:
        _logger.info("No failures detected - skipping teacher labeling.")
        skip_labeling = True
    if not converged and not skip_labeling:
        cap = int(len(distillation_dataset) * FAILURE_CAP_RATIO)
        if len(failed_specs_arr) > cap:
            top_k_idx = np.argsort(failure_errors[failure_mask])[-cap:]
            failed_specs_arr = failed_specs_arr[top_k_idx]
            _logger.info(f"  Capped failures to top {cap} highest-error cases")
    if not converged and not skip_labeling:
        new_lr = LR_INITIAL * (0.5 ** max(0, dagger_iter - LR_DECAY_AFTER_ITER))
        new_lr = max(new_lr, 1e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr * pg.get('lr_scale', 1.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_PER_ITER, eta_min=LR_FLOOR
        )
        _logger.info(f"  LR reset to {new_lr:.2e}; cosine scheduler restarted")

    final_specs = np.empty((0, 4), dtype=np.float32)
    final_labels = np.empty((0, 7), dtype=np.float32)
    if not converged and not skip_labeling:
        _logger.info(f"  Querying PlainMLP teacher for {len(failed_specs_arr)} failed specs ...")
        _logger.info("    Single deterministic MLP forward per spec (no candidate sampling).")
        try:
            new_labels_all = teacher_labeler.label_batch(failed_specs_arr)
        except Exception as e:
            _logger.error(f"  Teacher labeling failed: {e}; skipping append")
            new_labels_all = np.zeros((len(failed_specs_arr), 7), dtype=np.float64)

        keep_valid = filter_by_zig_validity(
            new_labels_all, zig_model, scaler_X, threshold=0.5, device=DEVICE
        )
        n_valid = int(keep_valid.sum())
        n_invalid_total = int((~keep_valid).sum())
        _logger.info(f"  ZIG validity gate (0.5): {n_valid}/{len(failed_specs_arr)} passed")
        if n_invalid_total > 0:
            _logger.info(f"  k-NN fallback for {n_invalid_total} invalid specs")
            for i in np.where(~keep_valid)[0]:
                new_labels_all[i] = empirical_fallback_label(failed_specs_arr[i], df, k=3)

        final_keep_mask = keep_valid | (~keep_valid)
        final_specs = failed_specs_arr
        final_labels = new_labels_all.astype(np.float32)
        _logger.info(f"  DAgger labels: total={len(final_labels)}/{len(failed_specs_arr)}")

    if len(final_specs) > 0:
        log_label_quality_summary("  DAgger labels", final_specs, final_labels, zig_model, scaler_X, scaler_y_p, device=DEVICE)
        log_param_bound_feasibility("DAgger labels", final_labels)
        distillation_dataset.append_samples(final_specs, final_labels)
        _logger.info(f"  Added {len(final_labels)} labeled failures.  "
                     f"Dataset now: {len(distillation_dataset)}")

    try:
        save_checkpoint(_ckpt_baseline_payload(dagger_iter=dagger_iter + 1, converged=converged))
        _logger.info(f"[CKPT] Iteration {dagger_iter + 1} checkpoint saved "
                     f"(next_iter={dagger_iter + 1}, dataset={len(distillation_dataset)})")
    except Exception as e:
        _logger.warning(f"[CKPT] Failed to write iteration-end checkpoint: {e}")

    if converged:
        break

_logger.info("DAgger training complete")


# ===========================================================================
# Final plots, evaluation, and artifact save
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(dagger_history['iteration'], dagger_history['failure_rate'], 'r-o', label='overall')
if dagger_history['boundary_failure_rate']:
    axes[0].plot(dagger_history['iteration'], dagger_history['boundary_failure_rate'],
                 'b--o', label='boundary', alpha=0.6)
    axes[0].plot(dagger_history['iteration'], dagger_history['interior_failure_rate'],
                 'g--o', label='interior', alpha=0.6)
axes[0].set_xlabel('DAgger Iteration')
axes[0].set_ylabel('Failure Rate')
axes[0].set_title('Validation Failure Rate vs Iteration')
axes[0].axhline(CONVERGENCE_THRESHOLD, color='g', linestyle='--',
                label=f'Convergence={CONVERGENCE_THRESHOLD}')
axes[0].legend()
axes[1].plot(dagger_history['iteration'], dagger_history['dataset_size'], 'b-o')
axes[1].set_xlabel('DAgger Iteration')
axes[1].set_ylabel('Dataset Size')
axes[1].set_title('Rolling Dataset Size')
axes[2].plot(dagger_history['iteration'], dagger_history['train_loss'], 'm-o')
axes[2].set_xlabel('DAgger Iteration')
axes[2].set_ylabel('Best Train Loss')
axes[2].set_title('Training Loss per Iteration')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dagger_curves.png'), dpi=150)
plt.close(fig)
_logger.info("DAgger training curves saved.")

# Final evaluation on held-out test set
MODEL_NAME = 'dagger_student_kirchhoff_mlp'
if len(df) > 0:
    df_eval = df.sample(min(1000, len(df)), random_state=42)
    test_specs = df_eval[['power', 'stage_2_eye_max_height',
                           'stage_2_eye_max_width', 'stage_2_jitter']].values.astype(np.float32)
    test_specs = test_specs[:, [0, 3, 1, 2]]
    evaluator = StudentEvaluator(
        student=student, scaler_X=scaler_X, zig_model=zig_model,
        eye_scale_h=eye_scale_h, eye_scale_w=eye_scale_w, eye_scale_j=eye_scale_j,
        scaler_y_p=scaler_y_p, device=DEVICE,
    )
    test_failure_mask, test_metrics = evaluator.identify_failures(test_specs, threshold=ERROR_THRESHOLD)
    test_failure_rate = test_failure_mask.mean()
    _logger.info(f"Test failure rate: {test_failure_rate * 100:.2f}%  "
                 f"({test_failure_mask.sum()}/{len(test_specs)} specs)")
    if 'boundary_failure' in test_metrics and 'is_boundary' in test_metrics:
        boundary_mask_eval = test_metrics['is_boundary']
        interior_mask_eval = ~boundary_mask_eval
        boundary_rate = test_metrics['boundary_failure'].sum() / max(1, boundary_mask_eval.sum())
        interior_rate = test_metrics['interior_failure'].sum() / max(1, interior_mask_eval.sum())
        _logger.info(f"  boundary: {boundary_rate * 100:.2f}%  "
                     f"interior: {interior_rate * 100:.2f}%")
    _logger.info(f"  Mean errors - power: {test_metrics['errors'][:, 0].mean():.4f}, "
                 f"jitter: {test_metrics['errors'][:, 1].mean():.4f}, "
                 f"height: {test_metrics['errors'][:, 2].mean():.4f}, "
                 f"width: {test_metrics['errors'][:, 3].mean():.4f}")

    _clip_stats = student.clipping_stats()
    _logger.info(
        "KNet input clipping: %d/%d elements (%.4f%%) hit [-%.1f, %.1f] during run",
        _clip_stats['clipped_elements'], _clip_stats['input_elements'],
        100.0 * _clip_stats['clip_fraction'], student.input_rail, student.input_rail,
    )

    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.pt'))
    joblib.dump(scaler_X, os.path.join(OUTPUT_DIR, 'scaler_X.pkl'))
    _logger.info(f"Saved student + scalers to {OUTPUT_DIR}")

    student.eval()
    test_specs2 = df_eval[['power', 'stage_2_jitter',
                            'stage_2_eye_max_height', 'stage_2_eye_max_width']].values.astype(np.float32)
    test_specs_t = torch.from_numpy(test_specs2).to(DEVICE)
    pred_dict = student.predict(test_specs_t)
    pred_params = np.stack([pred_dict[name].cpu().detach().numpy()
                             for name in PARAM_COLS], axis=1)
    pred_specs_zig = test_metrics['pred_specs']
    target_specs_zig = test_metrics['target_specs']

    csv_data = {
        'power':         test_specs2[:, 0],
        'jitter':        test_specs2[:, 1],
        'height':        test_specs2[:, 2],
        'width':         test_specs2[:, 3],
        'pred_fW':       pred_params[:, 0],
        'pred_current':  pred_params[:, 1],
        'pred_ind':      pred_params[:, 2],
        'pred_Rd':       pred_params[:, 3],
        'pred_Cs':       pred_params[:, 4],
        'pred_Rs':       pred_params[:, 5],
        'pred_VDD':      pred_params[:, 6],
        'zigi_power':    pred_specs_zig[:, 0],
        'zigi_jitter':   pred_specs_zig[:, 1],
        'zigi_height':   pred_specs_zig[:, 2],
        'zigi_width':    pred_specs_zig[:, 3],
        'target_power':  target_specs_zig[:, 0],
        'target_jitter': target_specs_zig[:, 1],
        'target_height': target_specs_zig[:, 2],
        'target_width':  target_specs_zig[:, 3],
    }
    df_csv = pd.DataFrame(csv_data)
    csv_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_predictions.csv')
    df_csv.to_csv(csv_path, index=False)
    _logger.info(f"Test predictions CSV saved to {csv_path}")
else:
    _logger.info("No data — saving student state only.")
    torch.save(student.state_dict(), os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.pt'))


# ===========================================================================
# Demo predictions for specific target specs
# ===========================================================================
LABELS = ['fW', 'current', 'ind', 'Rd', 'Cs', 'Rs', 'VDD']
target_specs_demo = np.array([
    [0.01044, 37.91, 21.14, 62.43],
    [0.00709, 11.45, 33.33, 88.68],
    [0.01107, 8.79,  29.37, 91.69],
    [0.00607, 16.59, 60.29, 83.90],
    [0.00229, 29.61,  4.44, 70.82],
    [0.00235, 27.55, 20.23, 72.58],
])
_logger.info("=" * 80)
_logger.info(f"{'Target Specs':^40} | {'KNet Student Predicted Parameters':^38}")
student.eval()
with torch.no_grad():
    for spec in target_specs_demo:
        power, jitter, height, width = spec
        spec_tensor = torch.tensor([[power, jitter, height, width]], dtype=torch.float32).to(DEVICE)
        pred_dict = student.predict(spec_tensor)
        pred_values = [pred_dict[name].item() for name in LABELS]
        _logger.info(f"{power:>8.5f} {jitter:>8.3f} {height:>8.3f} {width:>8.3f} | "
                     + " ".join(f"{v:>12.4e}" for v in pred_values))
_logger.info("=" * 80)
_logger.info(f"=== DAgger (PlainMLP -> KNet) Complete ===")
_logger.info(f"Outputs saved to: {OUTPUT_DIR}")
