"""Parse results from results_many_models_realizable_knet_nb and generate an HTML comparison report."""

from __future__ import annotations

import csv
import re
from html import escape
from pathlib import Path

import torch

RESULTS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_many_models_realizable_knet_nb")
EXTRA_EPOCHS_ROOT = Path(r"C:\Users\Atharva\Downloads\results_mlp_benchmarks_extra_epochs")
OUTPUT_HTML = Path(r"C:\Users\Atharva\Documents\kirchhoffnet_realistic\comparison_report.html")

# Source-code directories to skip (not result directories).
SKIP_DIRS = {"kirchhoffnet", "kirchhoffnet_realistic"}

# Display order for problems.
PROBLEM_ORDER = ["friedman1", "friedman2", "smooth2d", "housing"]


def parse_dir_name(name: str) -> dict | None:
    """Map a result directory name to {problem, model, loss, variant}."""
    if name in SKIP_DIRS:
        return None

    # MLP pattern: {problem}_mlp_{loss}
    m = re.fullmatch(r"(.+)_mlp_(huber|mse)", name)
    if m:
        return {
            "problem": m.group(1),
            "model": "MLP",
            "loss": m.group(2),
            "variant": "",
        }

    # KNet pattern: test_{problem}_knet[_vca]
    m = re.fullmatch(r"test_(.+)_knet(_vca)?", name)
    if m:
        return {
            "problem": m.group(1),
            "model": "KNet",
            "loss": "N/A",
            "variant": "VCA" if m.group(2) else "standard",
        }

    return None


def parse_final_metrics(path: Path) -> dict:
    """Parse the MLP final_metrics.txt key:value file."""
    metrics = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metrics[key.strip()] = value.strip()
    return metrics


def parse_knet_config(path: Path) -> dict:
    """Parse config_snapshot.txt for KNet architecture/training parameters.

    The file is a mix of ``key: value`` lines (problem, OPTIM block, I/O MAPPING block)
    and a single ``preset: {<python-dict-as-string>}`` line. We extract:
      - num_stages (count of 'num_hidden' in preset)
      - num_hidden (per-stage; assume uniform)
      - num_inputs (per-stage; assume uniform)
      - epochs (planned training epochs, from OPTIM block)
      - loss (KNet's internal loss string from preset)
    """
    out: dict = {
        "num_stages": None,
        "num_hidden": None,
        "num_inputs": None,
        "epochs": None,
        "knet_loss": None,
    }
    if not path.is_file():
        return out

    text = path.read_text(encoding="utf-8")

    # Locate the OPTIM block and parse its first 'epochs: NNN' line.
    optim_match = re.search(
        r"OPTIM:\s*\n((?:[ \t]+.+\n)+)", text
    )
    if optim_match:
        block = optim_match.group(1)
        m = re.search(r"^\s*epochs:\s*(\d+)", block, re.MULTILINE)
        if m:
            out["epochs"] = int(m.group(1))

    # Extract from 'preset: {...}' using regex
    preset_match = re.search(r"preset:\s*(\{.*\})", text, re.DOTALL)
    if preset_match:
        preset_str = preset_match.group(1)

        # Stages / hidden / inputs
        num_hidden_vals = re.findall(r"'num_hidden':\s*(\d+)", preset_str)
        num_inputs_vals = re.findall(r"'num_inputs':\s*(\d+)", preset_str)
        if num_hidden_vals:
            out["num_hidden"] = int(num_hidden_vals[0])
            out["num_stages"] = len(num_hidden_vals)
        if num_inputs_vals:
            out["num_inputs"] = int(num_inputs_vals[0])

        # Loss
        loss_match = re.search(r"'loss':\s*'([^']+)'", preset_str)
        if loss_match:
            out["knet_loss"] = loss_match.group(1)

    return out


def parse_loss_history(path: Path, has_phase_col: bool) -> dict | None:
    """Parse loss_history.txt and return metrics from the row with minimum val loss.

    Returns a dict with keys: best_epoch, best_val, mse_orig, rmse_orig, mae_orig, mape_orig.
    Returns None if file cannot be parsed.
    """
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    header = lines[0].split("\t")
    # Normalize header columns
    col_idx = {name: i for i, name in enumerate(header)}

    # Required columns
    required = {"epoch", "train", "val", "mse_orig", "rmse_orig", "mae_orig", "mape_orig"}
    if not required.issubset(col_idx.keys()):
        return None

    best_val = float("inf")
    best_row = None
    best_epoch = -1

    for raw in lines[1:]:
        parts = raw.split("\t")
        if len(parts) < len(header):
            continue
        try:
            val = float(parts[col_idx["val"]])
        except ValueError:
            continue
        if val < best_val:
            best_val = val
            best_epoch = int(float(parts[col_idx["epoch"]]))
            best_row = parts

    if best_row is None:
        return None

    return {
        "best_epoch": best_epoch,
        "best_val": best_val,
        "mse_orig": float(best_row[col_idx["mse_orig"]]),
        "rmse_orig": float(best_row[col_idx["rmse_orig"]]),
        "mae_orig": float(best_row[col_idx["mae_orig"]]),
        "mape_orig": float(best_row[col_idx["mape_orig"]]),
    }


def _find_file(child: Path, base_name: str) -> Path | None:
    """Find a file in ``child`` whose stem matches ``base_name``, allowing for
    filesystem-added suffixes like `` (2)`` (e.g., ``final_metrics (2).txt``)."""
    # Exact match first.
    direct = child / base_name
    if direct.is_file():
        return direct
    # Otherwise, glob for matching stem + optional suffix.
    stem, _, ext = base_name.partition(".")
    for path in child.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name == base_name:
            return path
        # Match: "stem (N).ext" or "stem.ext"
        if name.startswith(stem + " ") or name.startswith(stem + "."):
            if name.endswith(ext):
                return path
    return None


# Trainable param counts from Kaggle training stdout for KNet directories
# that lack model.pt or final_metrics.txt entries. Source: user-provided
# training logs showing "[train] trainable params: N".
KNOWN_KNET_PARAMS: dict[str, int] = {
    "test_friedman1_knet": 5_791,
    "test_friedman1_knet_vca": 6_341,
    "test_friedman2_knet": 5_647,
    "test_friedman2_knet_vca": 6_017,
    "test_smooth2d_knet": 1_842,
    "test_smooth2d_knet_vca": 1_982,
    "test_housing_knet": 5_743,  # 6_233 (VCA) − 490 (VCA add-on)
    "test_housing_knet_vca": 6_233,
}


def collect_results() -> list[dict]:
    """Walk the results directory and collect one record per result directory.

    For both MLP and KNet runs, we treat ``loss_history.txt`` (the row with
    minimum validation loss) as the source of truth for best metrics.
    ``final_metrics.txt`` is recorded separately as ``claimed_*`` so that any
    mismatches between the two can be surfaced in the report.
    """
    records = []
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"Results directory not found: {RESULTS_ROOT}")

    # Iterate the primary root first, then the extra-epochs MLP root, tagging each
    # record so identical directory names across roots remain distinguishable.
    roots = [
        (RESULTS_ROOT, ""),
        (EXTRA_EPOCHS_ROOT, "extra_epochs"),
    ]
    for root, tag in roots:
        if not root.is_dir():
            continue
        _collect_from_root(root, tag, records)

    return records


def _collect_from_root(root: Path, tag: str, records: list[dict]) -> None:
    """Collect one record per result directory under ``root`` into ``records``."""
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        info = parse_dir_name(child.name)
        if info is None:
            continue

        rec = {
            "source_root": tag,
            "directory": child.name,
            "problem": info["problem"],
            "model": info["model"],
            "loss": info["loss"],
            "variant": info["variant"],
            "mse_orig": None,
            "rmse_orig": None,
            "mae_orig": None,
            "mape_orig": None,
            "param_count": None,
            "epochs_run": None,
            "best_epoch": None,
            "best_val": None,
            "num_stages": None,
            "num_hidden": None,
            "num_inputs": None,
            "knet_loss": None,
            "source": "",
            "has_data": True,
            # Claimed values from final_metrics.txt (may be wrong).
            "claimed_mse_orig": None,
            "claimed_rmse_orig": None,
            "claimed_mae_orig": None,
            "claimed_mape_orig": None,
            "claimed_best_epoch": None,
            "claimed_best_val": None,
            "claimed_source": "",
            # Discrepancy flag.
            "metrics_discrepancy": False,
            "epoch_discrepancy": False,
        }

        fm_path = _find_file(child, "final_metrics.txt")
        lh_path = _find_file(child, "loss_history.txt")

        # Read claimed values from final_metrics.txt if present.
        if fm_path is not None:
            fm = parse_final_metrics(fm_path)
            rec["claimed_mse_orig"] = _safe_float(fm.get("best_mse_orig"))
            rec["claimed_rmse_orig"] = _safe_float(fm.get("best_rmse_orig"))
            rec["claimed_mae_orig"] = _safe_float(fm.get("best_mae_orig"))
            rec["claimed_mape_orig"] = _safe_float(fm.get("best_mape_orig"))
            rec["claimed_best_epoch"] = _safe_int(fm.get("best_epoch"))
            rec["claimed_best_val"] = _safe_float(fm.get("best_val"))
            rec["claimed_source"] = fm_path.name

        if info["model"] == "MLP":
            rec["param_count"] = _safe_int(
                parse_final_metrics(fm_path).get("param_count")
            ) if fm_path else None
            rec["epochs_run"] = _safe_int(
                parse_final_metrics(fm_path).get("epochs_run")
            ) if fm_path else None

            # Use loss_history.txt as source of truth.
            if lh_path is not None:
                lh = parse_loss_history(lh_path, has_phase_col=False)
                if lh is not None:
                    rec["mse_orig"] = lh["mse_orig"]
                    rec["rmse_orig"] = lh["rmse_orig"]
                    rec["mae_orig"] = lh["mae_orig"]
                    rec["mape_orig"] = lh["mape_orig"]
                    rec["best_epoch"] = lh["best_epoch"]
                    rec["best_val"] = lh["best_val"]
                    rec["source"] = f"{lh_path.name} (verified, min val)"
                    _check_discrepancy(rec)
                else:
                    rec["has_data"] = False
            else:
                rec["has_data"] = False
        else:
            # KNet
            cfg_path = _find_file(child, "config_snapshot.txt")
            cfg = parse_knet_config(cfg_path) if cfg_path else parse_knet_config(Path("__missing__"))
            rec["num_stages"] = cfg["num_stages"]
            rec["num_hidden"] = cfg["num_hidden"]
            rec["num_inputs"] = cfg["num_inputs"]
            rec["knet_loss"] = cfg["knet_loss"]
            if cfg["epochs"]:
                rec["epochs_run"] = cfg["epochs"]

            # Param count: final_metrics.txt (future runs) → KNOWN_KNET_PARAMS
            # (exact trainable count from training stdout) → model.pt (state_dict
            # total, includes frozen adjacency indices).
            if fm_path is not None:
                rec["param_count"] = _safe_int(
                    parse_final_metrics(fm_path).get("param_count")
                )
            if rec["param_count"] is None:
                rec["param_count"] = KNOWN_KNET_PARAMS.get(child.name)
            if rec["param_count"] is None:
                mp = _find_file(child, "model.pt")
                if mp is not None:
                    try:
                        sd = torch.load(mp, map_location="cpu", weights_only=True)
                        rec["param_count"] = sum(p.numel() for p in sd.values())
                    except Exception:
                        pass

            # Prefer final_metrics.txt if it actually has metric fields;
            # otherwise fall back to parsing loss_history.txt.
            used_fm = False
            if fm_path is not None and rec["claimed_mse_orig"] is not None:
                rec["mse_orig"] = rec["claimed_mse_orig"]
                rec["rmse_orig"] = rec["claimed_rmse_orig"]
                rec["mae_orig"] = rec["claimed_mae_orig"]
                rec["mape_orig"] = rec["claimed_mape_orig"]
                rec["best_epoch"] = rec["claimed_best_epoch"]
                rec["best_val"] = rec["claimed_best_val"]
                rec["source"] = fm_path.name
                used_fm = True
            if not used_fm and lh_path is not None:
                lh = parse_loss_history(lh_path, has_phase_col=True)
                if lh is not None:
                    rec["mse_orig"] = lh["mse_orig"]
                    rec["rmse_orig"] = lh["rmse_orig"]
                    rec["mae_orig"] = lh["mae_orig"]
                    rec["mape_orig"] = lh["mape_orig"]
                    rec["best_epoch"] = lh["best_epoch"]
                    rec["best_val"] = lh["best_val"]
                    rec["source"] = f"{lh_path.name} (min val)"
                else:
                    rec["has_data"] = False
            elif not used_fm:
                rec["has_data"] = False

        # If we still have no metrics, mark as no data
        if (
            rec["mse_orig"] is None
            and rec["rmse_orig"] is None
            and rec["mae_orig"] is None
            and rec["mape_orig"] is None
        ):
            rec["has_data"] = False

        records.append(rec)


def _check_discrepancy(rec: dict) -> None:
    """Set rec.metrics_discrepancy / epoch_discrepancy if claimed differs from verified."""
    tol = 1e-4
    rel_tol = 1e-4

    def differs(a, b):
        if a is None or b is None:
            return False
        if abs(b) > 1.0:
            return abs(a - b) > abs(b) * rel_tol
        return abs(a - b) > tol

    metrics_off = any([
        differs(rec["mse_orig"], rec["claimed_mse_orig"]),
        differs(rec["rmse_orig"], rec["claimed_rmse_orig"]),
        differs(rec["mae_orig"], rec["claimed_mae_orig"]),
        differs(rec["mape_orig"], rec["claimed_mape_orig"]),
        differs(rec["best_val"], rec["claimed_best_val"]),
    ])
    rec["metrics_discrepancy"] = metrics_off
    rec["epoch_discrepancy"] = (
        rec["best_epoch"] is not None
        and rec["claimed_best_epoch"] is not None
        and rec["best_epoch"] != rec["claimed_best_epoch"]
    )


def _safe_float(s):
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _safe_int(s):
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def fmt_num(value, decimals=4):
    if value is None:
        return "N/A"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 0.001):
        return f"{value:.4e}"
    return f"{value:.{decimals}f}"


def fmt_pct(value):
    """MAPE is stored as a ratio (e.g., 1.58 == 158%). Display with 2 decimals + %."""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def model_label(rec: dict) -> str:
    bits = [rec["model"]]
    if rec["loss"] not in (None, "N/A") and rec["loss"]:
        bits.append(rec["loss"])
    if rec["variant"]:
        bits.append(rec["variant"])
    return " / ".join(bits)


def dir_label(rec: dict) -> str:
    """Directory name, disambiguated by source root when names collide.

    The extra-epochs MLP results share directory names with the original
    root, so records sourced there get an epoch-count suffix (e.g.
    ``friedman1_mlp_huber (1200 ep)``) to stay distinguishable.
    """
    name = rec["directory"]
    if rec.get("source_root") == "extra_epochs":
        if rec["epochs_run"] is not None:
            return f"{name} ({rec['epochs_run']} ep)"
        return f"{name} (extra epochs)"
    return name


def arch_label(rec: dict) -> str:
    """Human-readable architecture string for the HTML."""
    if rec["model"] == "MLP":
        return f"{rec['param_count']:,}" if rec["param_count"] else "N/A"
    # KNet
    parts = []
    if rec["num_stages"] and rec["num_hidden"]:
        parts.append(f"{rec['num_stages']} stages")
        parts.append(f"{rec['num_hidden']} hidden")
    if rec["num_inputs"]:
        parts.append(f"{rec['num_inputs']} inputs")
    if rec["param_count"]:
        parts.append(f"{rec['param_count']:,} params")
    return " &middot; ".join(parts) if parts else "N/A"


def best_indices(values, lower_is_better=True):
    """Return set of indices of records that have the best (min or max) value, ignoring None."""
    pairs = [(i, v) for i, v in enumerate(values) if v is not None]
    if not pairs:
        return set()
    target = min if lower_is_better else max
    best = target(v for _, v in pairs)
    return {i for i, v in pairs if v == best}


def build_html(records: list[dict]) -> str:
    # Group by problem in defined order
    grouped: dict[str, list[dict]] = {p: [] for p in PROBLEM_ORDER}
    for rec in records:
        grouped.setdefault(rec["problem"], []).append(rec)

    css = """
    body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 24px; color: #1f2937; background: #f9fafb; }
    h1 { color: #111827; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }
    h2 { color: #1e3a8a; margin-top: 32px; border-left: 4px solid #2563eb; padding-left: 10px; }
    h3 { color: #1e40af; margin-top: 24px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0; background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 14px; }
    th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid #e5e7eb; }
    th { background: #1e40af; color: white; text-align: center; font-weight: 600; }
    th.left, td.left { text-align: left; }
    tr:hover { background: #f3f4f6; }
    .best { background: #d1fae5 !important; font-weight: 700; color: #065f46; }
    .worst { background: #fee2e2 !important; color: #991b1b; }
    .na { color: #9ca3af; font-style: italic; }
    .note { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 10px 14px;
            margin: 12px 0; font-size: 13px; }
    .meta { font-size: 12px; color: #6b7280; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
             font-weight: 600; margin-right: 4px; }
    .badge-mlp { background: #dbeafe; color: #1e40af; }
    .badge-knet { background: #fce7f3; color: #9d174d; }
    .badge-vca { background: #fef3c7; color: #92400e; }
    code { background: #f3f4f6; padding: 1px 6px; border-radius: 3px; font-size: 12px; }
    """

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="UTF-8">')
    parts.append("<title>Model Comparison Report</title>")
    parts.append(f"<style>{css}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<h1>Model Comparison Report</h1>")
    parts.append(
        f"<p class='meta'>Source: <code>{escape(str(RESULTS_ROOT))}</code> &middot; "
        f"<code>{escape(str(EXTRA_EPOCHS_ROOT))}</code> &middot; "
        f"{len(records)} result directories parsed</p>"
    )

    parts.append(_render_summary_table(records))
    parts.append("<h2>Per-Problem Comparisons</h2>")
    for problem in PROBLEM_ORDER:
        rows = grouped.get(problem, [])
        if not rows:
            continue
        parts.append(_render_problem_section(problem, rows))

    parts.append(_render_huber_section(records))
    parts.append(_render_val_loss_section(records))
    parts.append(_render_verification_section(records))
    parts.append(_render_methodology_note())

    parts.append("</body></html>")
    return "\n".join(parts)


def _render_summary_table(records: list[dict]) -> str:
    """One big table: all 14 (or however many) runs side by side."""
    out = ["<h2>Summary &mdash; All Models</h2>"]
    out.append("<table>")
    out.append(
        "<thead><tr>"
        "<th class='left'>Problem</th>"
        "<th class='left'>Model</th>"
        "<th class='left'>Loss / Variant</th>"
        "<th class='left'>Directory</th>"
        "<th>MSE (orig)</th>"
        "<th>RMSE (orig)</th>"
        "<th>MAE (orig)</th>"
        "<th>MAPE</th>"
        "<th>Best Epoch</th>"
        "<th>Best Val</th>"
        "<th class='left'>Params / Architecture</th>"
        "<th>Epochs Run</th>"
        "</tr></thead><tbody>"
    )
    for rec in records:
        badge = '<span class="badge badge-mlp">MLP</span>' if rec["model"] == "MLP" \
            else '<span class="badge badge-knet">KNet</span>'
        if rec["variant"] == "VCA":
            badge += '<span class="badge badge-vca">VCA</span>'
        if rec["model"] == "MLP":
            loss_label = rec["loss"]
        else:
            loss_label = rec.get("knet_loss") or "—"
        out.append("<tr>")
        out.append(f"<td class='left'>{escape(rec['problem'])}</td>")
        out.append(f"<td class='left'>{badge}</td>")
        out.append(f"<td class='left'>{escape(loss_label)}</td>")
        out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
        out.append(_cell(rec["mse_orig"]))
        out.append(_cell(rec["rmse_orig"]))
        out.append(_cell(rec["mae_orig"]))
        out.append(f"<td>{fmt_pct(rec['mape_orig']) if rec['mape_orig'] is not None else 'N/A'}</td>")
        out.append(f"<td>{rec['best_epoch'] if rec['best_epoch'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['best_val'], 6) if rec['best_val'] is not None else 'N/A'}</td>")
        out.append(f"<td class='left meta'>{arch_label(rec)}</td>")
        out.append(f"<td>{rec['epochs_run'] if rec['epochs_run'] is not None else 'N/A'}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _cell(value):
    if value is None:
        return "<td class='na'>N/A</td>"
    return f"<td>{fmt_num(value)}</td>"


def _render_problem_section(problem: str, rows: list[dict]) -> str:
    out = [f"<h3>{escape(problem)}</h3>"]

    has_any_data = any(r["has_data"] for r in rows)
    if not has_any_data:
        out.append(
            "<div class='note'>No metric data available yet for this problem. "
            "Source directories only contain <code>config_snapshot.txt</code>.</div>"
        )

    out.append("<table>")
    out.append(
        "<thead><tr>"
        "<th class='left'>Model</th>"
        "<th class='left'>Loss / Variant</th>"
        "<th class='left'>Directory</th>"
        "<th>MSE (orig)</th>"
        "<th>RMSE (orig)</th>"
        "<th>MAE (orig)</th>"
        "<th>MAPE</th>"
        "<th class='left'>Source</th>"
        "</tr></thead><tbody>"
    )

    mse_vals = [r["mse_orig"] for r in rows]
    rmse_vals = [r["rmse_orig"] for r in rows]
    mae_vals = [r["mae_orig"] for r in rows]
    mape_vals = [r["mape_orig"] for r in rows]

    best_mse = best_indices(mse_vals)
    best_rmse = best_indices(rmse_vals)
    best_mae = best_indices(mae_vals)
    best_mape = best_indices(mape_vals)

    for i, rec in enumerate(rows):
        # Compose a clean "loss / variant" label for the row.
        if rec["model"] == "MLP":
            loss_variant = f"{rec['loss']}{(' ' + rec['variant']) if rec['variant'] else ''}"
        else:
            # KNet: show the configured internal loss instead of "N/A".
            kl = rec.get("knet_loss") or "—"
            loss_variant = f"{kl}{(' ' + rec['variant']) if rec['variant'] else ''}"

        if not rec["has_data"]:
            cls = " class='na'"
            out.append(f"<tr{cls}>")
            out.append(f"<td class='left'>{escape(rec['model'])}</td>")
            out.append(f"<td class='left'>{escape(loss_variant)}</td>")
            out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
            for _ in range(4):
                out.append("<td class='na'>N/A</td>")
            out.append(f"<td class='left meta'>{escape(rec['source']) or 'pending'}</td>")
            out.append("</tr>")
            continue

        out.append("<tr>")
        out.append(f"<td class='left'>{escape(rec['model'])}</td>")
        out.append(f"<td class='left'>{escape(loss_variant)}</td>")
        out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
        out.append(_styled_cell(rec["mse_orig"], i in best_mse))
        out.append(_styled_cell(rec["rmse_orig"], i in best_rmse))
        out.append(_styled_cell(rec["mae_orig"], i in best_mae))
        out.append(
            f"<td class='{'best' if i in best_mape else ''}'>"
            f"{fmt_pct(rec['mape_orig'])}</td>"
        )
        out.append(f"<td class='left meta'>{escape(rec['source'])}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")

    # Add an interpretation block
    if has_any_data:
        out.append(_render_problem_winner(rows, best_mse, best_rmse, best_mae, best_mape))
    return "\n".join(out)


def _styled_cell(value, is_best):
    if value is None:
        return "<td class='na'>N/A</td>"
    cls = "best" if is_best else ""
    return f"<td class='{cls}'>{fmt_num(value)}</td>"


def _render_problem_winner(rows, best_mse, best_rmse, best_mae, best_mape):
    """Print a short sentence identifying the winner per metric."""
    out = ['<div class="note">']
    metric_map = {
        "MSE": best_mse,
        "RMSE": best_rmse,
        "MAE": best_mae,
        "MAPE": best_mape,
    }
    for name, idxs in metric_map.items():
        if not idxs:
            continue
        names = ", ".join(model_label(rows[i]) for i in idxs)
        out.append(f"<strong>Best {name}</strong>: {escape(names)}<br>")
    out.append("</div>")
    return "\n".join(out)


def _render_huber_section(records: list[dict]) -> str:
    """Compare Huber-loss performance: MLP-Huber vs KNet for each problem."""
    out = ["<h2>Huber Loss Comparison</h2>"]
    out.append(
        "<p class='meta'>Comparing models that target robust (Huber-style) regression. "
        "Note: every KNet run in this batch is configured with <code>loss: huber</code> "
        "(see <code>config_snapshot.txt</code>), so the Huber-loss comparison below is "
        "essentially MLP-Huber vs KNet-on-Huber on the same problem.</p>"
    )

    # MLP-Huber + KNet (and KNet VCA) per problem
    huber_rows = [r for r in records if r["loss"] == "huber"]
    knet_rows = [r for r in records if r["model"] == "KNet"]

    problems = PROBLEM_ORDER
    problems = [p for p in problems if any(r["problem"] == p for r in huber_rows + knet_rows)]

    out.append("<table>")
    out.append(
        "<thead><tr>"
        "<th class='left'>Problem</th>"
        "<th class='left'>Model</th>"
        "<th class='left'>Variant</th>"
        "<th class='left'>Directory</th>"
        "<th>MSE (orig)</th>"
        "<th>RMSE (orig)</th>"
        "<th>MAE (orig)</th>"
        "<th>MAPE</th>"
        "<th>Best Val Loss</th>"
        "</tr></thead><tbody>"
    )

    for problem in problems:
        prob_huber = [r for r in huber_rows if r["problem"] == problem]
        prob_knet = [r for r in knet_rows if r["problem"] == problem]
        combined = prob_huber + prob_knet
        if not combined:
            continue

        # Compute best indices within this problem group
        mse_vals = [r["mse_orig"] for r in combined]
        rmse_vals = [r["rmse_orig"] for r in combined]
        mae_vals = [r["mae_orig"] for r in combined]
        mape_vals = [r["mape_orig"] for r in combined]
        val_vals = [r["best_val"] for r in combined]

        best_mse = best_indices(mse_vals)
        best_rmse = best_indices(rmse_vals)
        best_mae = best_indices(mae_vals)
        best_mape = best_indices(mape_vals)
        best_val = best_indices(val_vals)

        first = True
        for i, rec in enumerate(combined):
            row_open = "<tr>"
            if first:
                row_open = f"<tr><td class='left' rowspan='{len(combined)}'><strong>{escape(problem)}</strong></td>"
                first = False
            out.append(row_open)
            out.append(f"<td class='left'>{escape(rec['model'])}</td>")
            out.append(f"<td class='left'>{escape(rec['variant']) or '&mdash;'}</td>")
            out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
            out.append(_styled_cell(rec["mse_orig"], i in best_mse))
            out.append(_styled_cell(rec["rmse_orig"], i in best_rmse))
            out.append(_styled_cell(rec["mae_orig"], i in best_mae))
            out.append(
                f"<td class='{'best' if i in best_mape else ''}'>"
                f"{fmt_pct(rec['mape_orig']) if rec['mape_orig'] is not None else 'N/A'}</td>"
            )
            out.append(
                f"<td class='{'best' if i in best_val else ''}'>"
                f"{fmt_num(rec['best_val'], 6) if rec['best_val'] is not None else 'N/A'}</td>"
            )
            out.append("</tr>")

    out.append("</tbody></table>")
    return "\n".join(out)


def _render_val_loss_section(records: list[dict]) -> str:
    """Side-by-side comparison of validation (best-val) loss between MLP-Huber and KNet.

    Since every KNet run in this batch is configured with huber loss internally
    (per ``PRESETS[problem]['loss']``), this comparison is a clean apples-to-apples
    view of how the two architectures learn under the same objective.
    """
    out = ["<h2>Validation Loss Comparison (Same-Loss, Huber)</h2>"]
    out.append(
        "<p class='meta'>All KNet runs in this batch use <code>huber</code> loss "
        "(see <code>PRESETS[problem]['loss']</code> in <code>config.py</code> and "
        "each <code>config_snapshot.txt</code>). This section therefore compares "
        "the best validation loss achieved by the MLP-Huber and the KNet (standard + VCA) "
        "on each problem, holding the loss function fixed.</p>"
    )

    huber_rows = [r for r in records if r["loss"] == "huber"]
    knet_rows = [r for r in records if r["model"] == "KNet"]

    problems = PROBLEM_ORDER
    problems = [p for p in problems if any(r["problem"] == p for r in huber_rows + knet_rows)]

    out.append("<table>")
    out.append(
        "<thead><tr>"
        "<th class='left'>Problem</th>"
        "<th class='left'>Model</th>"
        "<th class='left'>Variant</th>"
        "<th class='left'>Directory</th>"
        "<th>Best Val Loss</th>"
        "<th>Best Epoch</th>"
        "<th>Epochs Run</th>"
        "<th>Final RMSE (orig)</th>"
        "<th>Ratio (KNet / MLP-Huber)</th>"
        "</tr></thead><tbody>"
    )

    for problem in problems:
        prob_huber = [r for r in huber_rows if r["problem"] == problem]
        prob_knet = [r for r in knet_rows if r["problem"] == problem]
        # Anchor the ratio baseline to the original (600-epoch) MLP-Huber run;
        # list remaining MLP rows (e.g. extra-epochs) first, then KNet rows.
        baseline = next((r for r in prob_huber if r["source_root"] != "extra_epochs"), None)
        other_mlp = [r for r in prob_huber if r is not baseline]
        combined = ([baseline] if baseline else []) + other_mlp + prob_knet
        if not combined:
            continue

        val_vals = [r["best_val"] for r in combined]
        best_val = best_indices(val_vals)

        # Ratio is computed against the original (600-epoch) MLP-Huber baseline.
        mlp_val = baseline["best_val"] if (baseline and baseline["best_val"] is not None) else None

        first = True
        for i, rec in enumerate(combined):
            row_open = "<tr>"
            if first:
                row_open = f"<tr><td class='left' rowspan='{len(combined)}'><strong>{escape(problem)}</strong></td>"
                first = False
            out.append(row_open)
            out.append(f"<td class='left'>{escape(rec['model'])}</td>")
            out.append(f"<td class='left'>{escape(rec['variant']) or '&mdash;'}</td>")
            out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
            out.append(
                f"<td class='{'best' if i in best_val else ''}'>"
                f"{fmt_num(rec['best_val'], 6) if rec['best_val'] is not None else 'N/A'}</td>"
            )
            out.append(
                f"<td>{rec['best_epoch'] if rec['best_epoch'] is not None else 'N/A'}</td>"
            )
            out.append(
                f"<td>{rec['epochs_run'] if rec['epochs_run'] is not None else 'N/A'}</td>"
            )
            out.append(_styled_cell(rec["rmse_orig"], False))
            # Ratio column. Show 1.000x for the baseline row itself, and otherwise
            # the ratio of the run's val loss to the baseline (KNet vs MLP-600, or
            # MLP-extra relative to the MLP-600 baseline).
            if rec is baseline:
                out.append("<td class='meta'>1.000x (baseline)</td>")
            elif (
                mlp_val is not None
                and rec["best_val"] is not None
                and mlp_val > 0
            ):
                ratio = rec["best_val"] / mlp_val
                cls = "best" if ratio < 1.0 else ""
                out.append(f"<td class='{cls}'>{ratio:.3f}x</td>")
            else:
                out.append("<td class='na'>N/A</td>")
            out.append("</tr>")

    out.append("</tbody></table>")
    out.append(
        "<p class='meta'><strong>Ratio &lt; 1.0</strong> means the run achieved a "
        "lower validation loss than the original 600-epoch MLP-Huber baseline on the "
        "same problem; <strong>ratio &gt; 1.0</strong> means it did worse. Note: training "
        "budgets differ (MLP = 600 and 1200 epochs, KNet = 800 epochs) and architectures "
        "differ (MLP ~6k params vs KNet 5 stages &times; 25 hidden with analog dynamics), "
        "so this ratio is a like-for-like loss comparison, not a compute-normalized one.</p>"
    )
    return "\n".join(out)


def _render_verification_section(records: list[dict]) -> str:
    """Show claimed vs verified (loss_history min-val row) metrics for each run."""
    out = ["<h2>Verification: Claimed vs Verified Metrics</h2>"]
    out.append(
        "<p class='meta'>Each run's <code>final_metrics.txt</code> reports a claimed "
        "<em>best epoch</em> and the metrics observed at that epoch. To double-check, "
        "we independently extract the row with the lowest <code>val</code> column from "
        "<code>loss_history.txt</code> and call that the <em>verified</em> row. All "
        "metrics shown elsewhere in this report are the verified values; this section "
        "shows the claimed values for comparison.</p>"
    )

    # Sort: any discrepancy first, then by directory name.
    def _sort_key(r):
        return (0 if (r["metrics_discrepancy"] or r["epoch_discrepancy"]) else 1, r["directory"])

    sorted_records = sorted(records, key=_sort_key)

    out.append("<table>")
    out.append(
        "<thead><tr>"
        "<th class='left'>Directory</th>"
        "<th>Status</th>"
        "<th>Claimed Epoch</th>"
        "<th>Verified Epoch</th>"
        "<th>Claimed RMSE</th>"
        "<th>Verified RMSE</th>"
        "<th>Claimed MAE</th>"
        "<th>Verified MAE</th>"
        "<th>Claimed Val</th>"
        "<th>Verified Val</th>"
        "</tr></thead><tbody>"
    )

    for rec in sorted_records:
        ok = not (rec["metrics_discrepancy"] or rec["epoch_discrepancy"])
        status = "<span class='best'>OK</span>" if ok else "<span class='worst'>MISMATCH</span>"
        out.append("<tr>")
        out.append(f"<td class='left'><code>{escape(dir_label(rec))}</code></td>")
        out.append(f"<td>{status}</td>")
        out.append(f"<td>{rec['claimed_best_epoch'] if rec['claimed_best_epoch'] is not None else 'N/A'}</td>")
        out.append(f"<td>{rec['best_epoch'] if rec['best_epoch'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['claimed_rmse_orig']) if rec['claimed_rmse_orig'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['rmse_orig']) if rec['rmse_orig'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['claimed_mae_orig']) if rec['claimed_mae_orig'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['mae_orig']) if rec['mae_orig'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['claimed_best_val'], 6) if rec['claimed_best_val'] is not None else 'N/A'}</td>")
        out.append(f"<td>{fmt_num(rec['best_val'], 6) if rec['best_val'] is not None else 'N/A'}</td>")
        out.append("</tr>")

    out.append("</tbody></table>")

    n_mismatch = sum(1 for r in records if r["metrics_discrepancy"] or r["epoch_discrepancy"])
    n_total = sum(1 for r in records if r["has_data"])
    if n_mismatch == 0:
        out.append("<div class='note'>All runs agree: claimed metrics match the min-val row in <code>loss_history.txt</code>.</div>")
    else:
        out.append(
            f"<div class='note'><strong>{n_mismatch} of {n_total}</strong> runs had a "
            f"discrepancy between <code>final_metrics.txt</code> and the actual min-val row "
            f"in <code>loss_history.txt</code>. The verified values (from loss_history.txt) "
            f"are used throughout this report. The most common pattern is "
            f"<code>final_metrics.txt</code> reporting the best epoch from an earlier "
            f"snapshot than the true minimum of the validation curve.</div>"
        )
    return "\n".join(out)


def _render_methodology_note():
    return (
        "<h2>Methodology</h2>"
        "<ul>"
        "<li><strong>Source of truth</strong>: for both MLP and KNet runs, the report uses "
        "the row with the lowest <code>val</code> in <code>loss_history.txt</code> as the "
        "<em>verified</em> best-epoch metrics. The <code>final_metrics.txt</code> values "
        "(labeled <em>claimed</em>) are kept in the Verification section for comparison.</li>"
        "<li><strong>MLP</strong> metrics in this report come from <code>loss_history.txt</code> "
        "(verified), not from <code>final_metrics.txt</code>. Several MLP directories had "
        "incorrect <code>best_epoch</code> values in <code>final_metrics.txt</code> (see the "
        "Verification section). MLP results are sourced from two directories: the original "
        "<code>results_many_models_realizable_knet_nb</code> (600 epochs) and the extra-epochs "
        "<code>results_mlp_benchmarks_extra_epochs</code> (1200 epochs). Because these share "
        "directory names, rows from the extra-epochs source are labeled with their epoch count, "
        "e.g. <code>friedman1_mlp_huber (1200 ep)</code>.</li>"
        "<li><strong>KNet</strong> directories that populate <code>final_metrics.txt</code> "
        "are read from there directly; otherwise we fall back to <code>loss_history.txt</code> "
        "(min-val row).</li>"
        "<li><strong>Loss function</strong>: every KNet run in this batch is configured "
        "with <code>huber</code> loss internally (from <code>PRESETS[problem]['loss']</code> "
        "in <code>config.py</code> and confirmed in each <code>config_snapshot.txt</code>). "
        "Therefore the <em>Huber Loss Comparison</em> and <em>Validation Loss Comparison</em> "
        "sections are apples-to-apples: same loss, same problem, different architecture.</li>"
        "<li>All reported metrics are on the <em>original</em> target scale "
        "(<code>*_orig</code> columns).</li>"
        "<li>MAPE is reported as a percentage. The source files store MAPE as a ratio "
        "(e.g., <code>1.58</code> &rarr; 158%); values &gt;100% indicate the model under-"
        "predicts on at least some targets relative to ground truth.</li>"
        "<li>Comparisons are only valid within the same problem (friedman1, friedman2, "
        "smooth2d, housing).</li>"
        "</ul>"
    )


def main():
    records = collect_results()
    print(f"Parsed {len(records)} result directories:")
    for r in records:
        status = "OK" if r["has_data"] else "NO DATA"
        arch = ""
        if r["model"] == "KNet":
            arch = f" stages={r['num_stages']} hid={r['num_hidden']} in={r['num_inputs']} knet_loss={r['knet_loss']}"
            if r["param_count"]:
                arch += f" params={r['param_count']}"
        else:
            arch = f" params={r['param_count']}"
        print(
            f"  [{status}] {r['directory']:32s} "
            f"problem={r['problem']:10s} model={r['model']:3s} "
            f"loss={r['loss']:5s} variant={r['variant']:8s}"
            f" epochs_run={r['epochs_run']}{arch}"
        )

    html = build_html(records)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nWrote {OUTPUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
