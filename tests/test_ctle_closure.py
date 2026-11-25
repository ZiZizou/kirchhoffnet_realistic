import csv
from types import SimpleNamespace

import numpy as np
import torch

from scripts.ctle_closure.mapping import PARAM_COLS, write_params, bounded_params


def test_logits_mapping_matches_torch_prediction_path():
    model = SimpleNamespace(
        log_lo=torch.tensor([-6.0, -3.30103, -12.0, 1.0, -15.0, 1.0, -0.22185]),
        log_hi=torch.tensor([1.0, 0.39794, 0.47712, 3.17609, -9.0, 3.17609, 0.07918]),
    )
    logits = torch.tensor([[0.0] * 7, [-2, -1, 0.5, 1, 2, 3, 4], [2, 1, -0.5, -1, -2, -3, -4]])
    predicted_logits = logits  # same raw output as model.forward(specs)
    expected = model.log_lo + (model.log_hi - model.log_lo) * torch.sigmoid(predicted_logits)
    actual = bounded_params(logits, model)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, 7)


def test_pred_csv_matches_spice_failure_check_contract(tmp_path):
    rows = np.asarray([[0.1 * i for i in range(7)], [-0.2 * i for i in range(7)]])
    path = tmp_path / "pred.csv"
    write_params(path, rows)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(PARAM_COLS)
        parsed = np.asarray([[float(row[k]) for k in PARAM_COLS] for row in reader])
    np.testing.assert_allclose(parsed, rows, rtol=1e-6, atol=1e-7)
