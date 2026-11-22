import sys; sys.path.insert(0, r'C:\Users\Atharva\Documents\kirchhoffnet_realistic')
import math, torch, narma_experiment as ne, narma_gain_norm as gn

for retention in [0.97, 0.93, 0.9, 0.85, 0.8]:
    net, t_span, ns = ne._build_fabric_net(
        order=10, seed=0, freeze_read=False,
        t_span=1.0, num_steps=8, cell_library='tanh_free',
        core_refresh_interval=0, leak_constant=None,
        compile_sequence=False, hidden_dim=25,
    )
    stage = net.core.stages[0]
    dt = t_span / ns
    leak_val = -math.log(retention) / dt
    results = gn.normalize_spectral_radius(net, target_rho=0.9, knob='isat', t_span=t_span, num_steps=ns, leak_target=leak_val, max_iter=8)
    r = results[0]
    note_short = r.note[:60] if r.note else ''
    print(f'retention={retention:.2f}, leak={leak_val:.3f}: rho_before={r.rho_before:.3f} -> rho_after={r.rho_after:.3f}, scale={r.scale_applied:.2f}, note={note_short!r}')
