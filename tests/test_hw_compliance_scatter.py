"""hw_compliance must still inject the boundary and temporal-mesh currents (W5 fix).

Regression: in ``DifferentialStage.rhs`` the 3.4 compliance branches of the
boundary family and of the legacy ota_mesh readout computed the rolled-off
currents but never scatter-added them into ``acc``, so under ``hw_compliance``
the network input (boundary OTAs) and the temporal readout mesh were
disconnected: their parameters got no gradient (found by the W5 learning
validation).  At ``x = 0`` the compliance factors are 1, so the rhs must equal
the no-compliance rhs there.
"""

import pytest
import torch

import make_golden as mg
from hw import HWConfig
from hw.constraints import HWConstraints

D = torch.float64


def _net(spec, **cons):
    hw = HWConfig(constraints=HWConstraints(**cons)) if cons else None
    with mg.f64():
        mg.seed_all(int(spec["seed"]))
        return mg._build_net(spec, hw=hw).double()


@pytest.mark.parametrize("spec_name,fams", [
    ("TOY_NET", ("boundary_cell_lib", "boundary_z_logits", "gln_rails.families.boundary")),
    ("FRIEDMAN2_NET", ("boundary_cell_lib", "boundary_z_logits",
                       "output_ode_cell_lib", "output_ode_z_logits")),
])
def test_boundary_and_mesh_get_gradients_under_compliance(spec_name, fams):
    spec = getattr(mg, spec_name)
    net = _net(spec, hw_compliance=True, hw_no_alpha=True)
    net.train()
    x = mg.make_inputs(spec)[:16]
    with mg.f64():
        mg.model_forward(net, x).pow(2).mean().backward()
    for fam in fams:
        ps = [(n, p) for n, p in net.named_parameters() if fam in n and p.requires_grad]
        assert ps, fam
        assert all(p.grad is not None for _, p in ps), [n for n, p in ps if p.grad is None]
        assert any(float(p.grad.abs().max()) > 0 for _, p in ps), fam


@pytest.mark.parametrize("spec_name", ["TOY_NET", "FRIEDMAN2_NET"])
def test_compliance_rhs_equals_plain_rhs_at_x0(spec_name):
    spec = getattr(mg, spec_name)
    # symmetric Vref: Vref = 0 at the init raw, so the sense legs (which roll
    # off at the Vref rail voltage) also see a compliance factor of 1
    plain = _net(spec, hw_no_alpha=True, hw_vref_symmetric=True)
    comp = _net(spec, hw_no_alpha=True, hw_vref_symmetric=True, hw_compliance=True)
    comp.load_state_dict(plain.state_dict())
    g = torch.Generator().manual_seed(5)
    for sp, sc in zip(plain.core.stages, comp.core.stages):
        u = torch.rand(8, int(sp.boundary_src.max()) + 1 if sp.boundary_src.numel() else 4,
                       generator=g, dtype=D) * 2 - 1
        x0 = torch.zeros(8, sp.num_nodes, dtype=D)
        with torch.no_grad():
            f_p = sp.rhs(x0, u=u)
            f_c = sc.rhs(x0, u=u)
        assert float(f_c.abs().max()) > 0
        # both paths accumulate KCL in float32 (index_add_), hence ~1e-7 relative
        torch.testing.assert_close(f_c, f_p, rtol=1e-6, atol=1e-9)


def test_audit_stiffness_slices_frozen_legs_to_n_rows():
    """hw.audit.stiffness_stats with n_rows < batch under compliance + freeze_read (W5 fix)."""
    import torch.nn as nn
    from hw.audit import run_capture, stiffness_stats

    class W(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return mg.model_forward(self.net, x)
    m = W(_net(mg.TOY_NET, hw_compliance=True, hw_no_alpha=True))
    with mg.f64():
        _, recs, _ = run_capture(m, mg.make_inputs(mg.TOY_NET)[:16])
        st = stiffness_stats(m, recs, n_rows=4)
    assert len(st) == 2 and all(r["max_dt_rho"] > 0 for r in st)
