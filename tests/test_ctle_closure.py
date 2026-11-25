import csv

import numpy as np
import pytest

from scripts.ctle_closure.mapping import PARAM_COLS, write_params


@pytest.fixture(scope="module")
def tiny_model():
    """Cached deterministic model for the quick mapping-path unit check."""
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("log_lo", torch.tensor([-6.0, -3.30103, -12.0, 1.0, -15.0, 1.0, -0.22185]))
            self.register_buffer("log_hi", torch.tensor([1.0, 0.39794, 0.47712, 3.17609, -9.0, 3.17609, 0.07918]))

        def forward(self, specs):
            logits = torch.tensor([[0.0] * 7, [-2, -1, 0.5, 1, 2, 3, 4]], dtype=self.log_lo.dtype)
            return logits[:len(specs)]

    return TinyModel().eval()


@pytest.fixture(scope="session")
def paper_model():
    """Load the real checkpoint once for the opt-in mapping integration test."""
    import torch
    from checkpoints import load_paper_checkpoint
    return load_paper_checkpoint(dtype=torch.float64).eval()


def test_pred_csv_matches_spice_failure_check_contract(tmp_path):
    rows = np.asarray([[0.1 * i for i in range(7)], [-0.2 * i for i in range(7)]])
    path = tmp_path / "pred.csv"
    write_params(path, rows)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(PARAM_COLS)
        parsed = np.asarray([[float(row[k]) for k in PARAM_COLS] for row in reader])
    np.testing.assert_allclose(parsed, rows, rtol=1e-6, atol=1e-7)


@pytest.mark.slow
def test_logits_mapping_matches_torch_prediction_path(tiny_model):
    import torch
    from scripts.ctle_closure.mapping import bounded_params
    specs = torch.ones((2, 4), dtype=tiny_model.log_lo.dtype)
    with torch.no_grad():
        logits = tiny_model(specs)
        torch_prediction = tiny_model.log_lo + (tiny_model.log_hi - tiny_model.log_lo) * torch.sigmoid(logits)
        decoded = bounded_params(logits, tiny_model)
    torch.testing.assert_close(decoded, torch_prediction, rtol=0, atol=0)
    assert decoded.shape == (2, len(PARAM_COLS))


@pytest.mark.slow
def test_checkpoint_logits_mapping_matches_torch_prediction_path(paper_model):
    # One row limits inference cost; model loading is cached by the session fixture.
    import torch
    specs = torch.full((1, 4), 0.5, dtype=paper_model.log_lo.dtype)
    with torch.no_grad():
        logits = paper_model(specs)
        expected = paper_model.log_lo + (paper_model.log_hi - paper_model.log_lo) * torch.sigmoid(logits)
        from scripts.ctle_closure.mapping import bounded_params
        decoded = bounded_params(logits, paper_model)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert decoded.shape == (1, len(PARAM_COLS))
