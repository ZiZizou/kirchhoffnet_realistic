"""Static contract checks for the generated Kaggle retraining notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_hw_retrain.ipynb"
GENERATOR = ROOT / "scripts" / "kaggle" / "build_notebook.py"

CLONE_CELL = '''import os, sys
from kaggle_secrets import UserSecretsClient

# Retrieve the token securely
user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GITHUB_TOKEN")
branch = "hw-refactor"  # <--- Specify your branch here ("main" or "deq")

# Define your GitHub details
username = "ZiZizou"
repo = "kirchhoffnet_realistic"

# Clone the specific branch using the -b flag
os.system(f"git clone -b {branch} https://{username}:{token}@github.com/{username}/{repo}.git")


# The exact path to your cloned repository
repo_path = "/kaggle/working/"+repo

if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
'''

CSV_FILES = (
    "dataset_log_part_1_may3.csv", "dataset_log_part_1_may4.csv",
    "dataset_log_part_2_may3.csv", "dataset_log_part_2_may4.csv",
    "dataset_log_part_3_may3.csv", "dataset_log_part_3_may4.csv",
    "dataset_log_part_4_may3.csv", "dataset_log_part_4_may4.csv",
    "dataset_log_part_5_may3.csv", "dataset_log_part_5_may4.csv",
    "dataset_log_part_6_may3.csv", "dataset_log_part_6_may4.csv",
    "dataset_log_may_7.csv", "dataset_log_march12.csv", "dataset_log_may_18.csv",
)
TEACHER_FILES = (
    "ctle_conditional_flow.pt", "flow_scaler_X.pkl", "flow_scaler_C.pkl",
    "hybrid_hurdle_ctle_model.pt", "hybrid_hurdle_scaler_X.pkl",
    "hybrid_hurdle_scaler_y_power.pkl", "hybrid_hurdle_config.pkl",
)


def test_notebook_is_clean_nbformat_four_and_clone_cell_is_preserved():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 4
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert "".join(notebook["cells"][2]["source"]) == CLONE_CELL
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_generator_and_notebook_use_fixed_inputs_without_bundle_extraction():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    generator = GENERATOR.read_text(encoding="utf-8")
    for text in (source, generator):
        assert "KNB_TEACHER_CKPT" in text
        assert "KNB_TEACHER_DIR" in text
        assert "KNB_CSV_DIR" in text
        assert "KNB_CANONICAL" in text
        assert "GITHUB_TOKEN" in text
        assert "zipfile" not in text
        assert "MANIFEST.txt" not in text
        assert "bundle_root" not in text
        assert "assets/teachers/trial_0019/dagger_student_plain.pt" not in text
    assert "/kaggle/input/models/awekill/improved-zig-nf-spline/pytorch/default/4/dagger_student_plain.pt" in source
    assert "/kaggle/input/datasets/awekill/augmented-cvae-ctle" in source
    assert '"--phase-a-mlp-teacher-ckpt", str(TEACHER_CKPT)' in source
    assert '"--teacher-dir", TEACHER_DIR' in source
    assert '"--data-dir", CSV_DIR' in source
    assert '"--zig-dir", TEACHER_DIR' in source
    for name in CSV_FILES + TEACHER_FILES:
        assert name in source
    assert "Missing required Kaggle inputs" in source
    assert "hashlib.sha256" in source
    assert "WARNING:" in source
