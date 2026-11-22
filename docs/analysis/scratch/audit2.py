import sys, importlib.util, argparse, torch, numpy as np, time, copy, json, joblib, os
import torch.nn.functional as F
sys.path.insert(0, r"C:\Users\Atharva\Documents\kirchhoffnet_realistic")
spec = importlib.util.spec_from_file_location("fm", r"fixed-mlp-distillation-kirchhoffnet.py")
fm = importlib.util.module_from_spec(spec); spec.loader.exec_module(fm)
from ctle_dagger_common import StudentEvaluator, HybridHurdleModel, _set_active_context
from cell_library import FreeTanhLibrary
torch.manual_seed(0)
d=r"C:\Users\Atharva\Downloads\trained_knet\fixed_mlp_knet\\"
b=torch.load(d+"student_bundle.pt",map_location="cpu",weights_only=False)
args=argparse.Namespace(**b["args"]); args.kn_solver_scale=1.0; args.student_kind="knet"
def build():
    m=fm.KirchhoffStudent(args,np.asarray(b["input_log_min"]),np.asarray(b["input_log_max"]))
    m.load_state_dict(b["model"]); m.eval(); return m
m=build()
p=np.load(d+"test_teacher_student_predictions.npz"); z=np.load(d+"zig_failure_comparison.npz")
specs=torch.from_numpy(p["specs"]); sv=p["student_logits"]; tl=p["teacher_logits"]
def q(t):
    t=(t.detach().cpu().numpy() if hasattr(t,"detach") else np.asarray(t)).astype(np.float64).ravel(); return "min %.4g | med %.4g | max %.4g"%(t.min(),np.median(t),t.max())
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)

# ---------- raw param stats ----------
P("## RAW PARAMETER STATS (per stage)")
for i,st in enumerate(m.net.core.stages):
    P(f"### stage {i}")
    for fam,lib,zl in [("core",st.cell_lib,st.z_logits),("boundary",st.boundary_cell_lib,st.boundary_z_logits),("readout(output_ode)",st.output_ode_cell_lib,st.output_ode_z_logits)]:
        with torch.no_grad():
            A=F.softplus(lib.a_raw); B=F.softplus(lib.b_raw); s=torch.sign(lib.s_raw)
            gm=lib.gm_min+(lib.gm_max-lib.gm_min)*torch.sigmoid(lib.gm_raw)
            isat=(lib.isat_min+(lib.isat_max-lib.isat_min)*torch.sigmoid(lib.isat_raw)).clamp_min(1e-6)
            th=lib.theta_raw; G=F.softplus(lib.g_resistive_raw); g=torch.sigmoid(zl)
            gmeff=isat*gm*A*1.01
        P(f"- {fam} (E={A.numel()}): gm {q(gm)}; Isat {q(isat)}; A {q(A)}; B {q(B)}; gm*A {q(gm*A)}; gm*B {q(gm*B)}; theta {q(th)}; |gm*theta| {q(gm*th.abs())}; G_shunt {q(G)}; gate sigma(z) {q(g)}; small-signal Isat*gm*A {q(gmeff)}; s=+1 frac {(s>0).float().mean():.2f}; gate<0.1: {(g<0.1).sum().item()}, gate>0.9: {(g>0.9).sum().item()}; gm at ceiling(>9.9): {(gm>9.9).sum().item()}, Isat >9.9: {(isat>9.9).sum().item()}, gm<0.1: {(gm<0.1).sum().item()}")
    P(f"- leak (fixed non-programmable): {st._effective_leak(num_nodes=21)[0].item():.4f}; clip softness {st.clip_softness}, clip_current {st.clip_current}, x_max {st.x_max}")
P("### interstage transfers")
for i,t in enumerate(list(m.net.core.transfers)+[m.net.post_readout_transfer]):
    P(f"- {'post_readout' if i==3 else 'transfer'+str(i)}: w1 {q(t.residual_w1)}; w2 {q(t.residual_w2)}; w3 {q(t.residual_w3)}; vth {q(t.residual_vth)}")
om=m.net.output_mapper
P("### output_mapper: gain",q(om.gain),"| bias",q(om.bias),"| tanh_gain",q(om.tanh_gain),"| relu_gain",q(om.relu_gain),"| relu_thr",q(om.relu_threshold))
P("### VCA weights: vca_W |max|",[round(st.vca_W.abs().max().item(),3) for st in m.net.core.stages],"vca_W_core |max|",[round(st.vca_W_core.abs().max().item(),3) for st in m.net.core.stages])

# ---------- patch tanh_core to capture u ----------
labels={}
for i,st in enumerate(m.net.core.stages):
    labels[id(st.cell_lib)]=f"s{i}.core"; labels[id(st.boundary_cell_lib)]=f"s{i}.bnd"; labels[id(st.output_ode_cell_lib)]=f"s{i}.rd"
rec={}
orig=FreeTanhLibrary._tanh_core
def patched(self,x_src,x_dst,gm_override=None):
    A=F.softplus(self.a_raw).unsqueeze(0);B=F.softplus(self.b_raw).unsqueeze(0);s=torch.sign(self.s_raw)
    gm=(self.gm_min+(self.gm_max-self.gm_min)*torch.sigmoid(self.gm_raw)).unsqueeze(0)
    th=self.theta_raw.unsqueeze(0)
    u=(s.unsqueeze(0)*(A*x_src-B*x_dst)+th)*gm
    r=rec.setdefault(labels.get(id(self),"?"),dict(umax=0,n=0,sat2=0,sat3=0,vmax=0,over=0,tot=0))
    r["umax"]=max(r["umax"],u.abs().max().item()); r["n"]+=u.numel(); r["sat2"]+=(u.abs()>2).sum().item(); r["sat3"]+=(u.abs()>3).sum().item()
    v=torch.cat([x_src.abs().flatten(),x_dst.abs().flatten()]); r["vmax"]=max(r["vmax"],v.max().item()); r["over"]+=(v>self.__dict__.get("_xm",4.0)).sum().item(); r["tot"]+=v.numel()
    return orig(self,x_src,x_dst,gm_override)
FreeTanhLibrary._tanh_core=patched

# ---------- forward, trajectories ----------
sp=specs[:256]
tr_hooks=[]
tr_out={}
for i,t in enumerate(list(m.net.core.transfers)+[m.net.post_readout_transfer]):
    tr_hooks.append(t.register_forward_hook(lambda mod,inp,o,i=i: tr_out.__setitem__(i,(inp[0].abs().max().item(),o.abs().max().item()))))
with torch.no_grad():
    sc=m.scale_input(sp)
    y,trajs=m.net(sc,store_trajectory=True,solver="heun")
FreeTanhLibrary._tanh_core=orig
for h in tr_hooks: h.remove()
P("\n## SCALED INPUT (to net, rail 4): min/med/max per dim",[q(sc[:,k].numpy()) for k in range(4)])
P("\n## NODE VOLTAGES over 256 test rows (x_max=4)")
for i,tj in enumerate(trajs):
    h=tj[:,:14,:].abs(); o=tj[:,14:,:].abs()
    P(f"- stage {i}: max|V| hidden {h.max():.3f} (p99 {h.flatten().quantile(0.99) if h.numel()<16e6 else float('nan'):.3f}), output-ODE nodes {o.max():.3f}; rows with any |V|>4: {(tj.abs().amax(dim=(1,2))>4).sum().item()}/256; final-state max|V| {tj[:,:,-1].abs().max():.3f}; x0 max|V| {tj[:,:,0].abs().max():.3f}")
for i,(a,bb) in tr_out.items(): P(f"- transfer{i} ({'post-readout' if i==3 else 'interstage'}): max|in| {a:.3f}  max|out| {bb:.3f}")
P("\n## DEVICE OPERATING POINT (|u|=arg of tanh, V at OTA terminals; counts over all Heun calls)")
for k in sorted(rec):
    r=rec[k]; P(f"- {k}: max|u| {r['umax']:.2f}; frac|u|>2 {r['sat2']/r['n']:.3f}; frac|u|>3 {r['sat3']/r['n']:.3f}; max|V_term| {r['vmax']:.2f}; frac V_term>4 {r['over']/r['tot']:.4f}")

# ---------- VCA gains ----------
P("\n## VCA GAIN 2*sigmoid((u@W)@v.T) over 256 rows")
with torch.no_grad():
    for i,st in enumerate(m.net.core.stages):
        for nm,W,v in [("boundary",st.vca_W,st.vca_v_boundary),("readout",st.vca_W,st.vca_v_readout),("core",st.vca_W_core,st.vca_v_core)]:
            g=2*torch.sigmoid((sc@W)@v.T)
            span=(g.amax(0)-g.amin(0))
            P(f"- stage{i} {nm}: gain {q(g.numpy())}; per-edge swing over inputs (max-min) med {span.median():.3f} max {span.max():.3f}; edges with gain<0.5 or >1.5 anywhere: {((g.amin(0)<0.5)|(g.amax(0)>1.5)).sum().item()}/{g.shape[1]}")

# ---------- failure evaluator ----------
zd=r"Y:\Documents\ASPDAC_2026\kirchhoff_redesign\mlp_distillation_target\\"
dev=torch.device("cpu")
sx=joblib.load(zd+"hybrid_hurdle_scaler_X.pkl"); sp_=joblib.load(zd+"hybrid_hurdle_scaler_y_power.pkl"); cfg=joblib.load(zd+"hybrid_hurdle_config.pkl")
zig=HybridHurdleModel(dropout=0.0,per_target=bool(cfg.get("per_target_hurdle",False))); zig.load_state_dict(torch.load(zd+"hybrid_hurdle_ctle_model.pt",map_location="cpu")); zig.eval()
for q_ in zig.parameters(): q_.requires_grad=False
_set_active_context({"DEVICE":dev,"zig_model":zig,"scaler_X":sx,"scaler_y_p":sp_,"VALIDITY_THRESHOLD":0.5,"DEGRADE_REL_THRESHOLD":0.2,"MIN_DEGRADED_DIMS":2,"ERROR_THRESHOLD":0.10})
def fail(model,sp_np):
    ev=StudentEvaluator(model,scaler_X=sx,zig_model=zig,eye_scale_h=float(cfg["eye_scale_h"]),eye_scale_w=float(cfg["eye_scale_w"]),eye_scale_j=float(cfg["eye_scale_j"]),scaler_y_p=sp_,device=dev)
    f,mt=ev.identify_failures(sp_np,threshold=float(os.environ.get("THR","0.20"))); return f
allspecs=p["specs"]; N=len(allspecs)
tmask=z["teacher_failure_mask"]; smask=z["student_failure_mask"]
def run(model,n=N):
    with torch.no_grad(): lg=model(torch.from_numpy(allspecs[:n])).numpy()
    return lg
def lp(lg): lo=b["model"]["log_lo"].numpy(); hi=b["model"]["log_hi"].numpy(); return lo+(hi-lo)/(1+np.exp(-lg))
def report(name,lg,n=N,full=True):
    mse_s=((lg-sv[:n])**2).mean(); mse_t=((lg-tl[:n])**2).mean(); mae=np.abs(lp(lg)-lp(sv[:n])).mean()
    s=f"- {name}: logit MSE vs saved-student {mse_s:.3e}; vs teacher {mse_t:.3e}; max|dlogit| {np.abs(lg-sv[:n]).max():.3e}; log10-param MAE vs saved-student {mae:.4f}"
    if full:
        f=fail(model_ref[0],allspecs[:n]) if False else None
    return s
P("\n## FAILURE RATE (ZIG forward model, test split n=%d)"%N)
P(f"- saved masks: teacher {tmask.mean():.4%}, student {smask.mean():.4%}")
t0=time.time(); lg0=run(m); P(f"- forward all {N} rows: {time.time()-t0:.1f}s")
f0=fail(m,allspecs); P(f"- rebuilt model, 1x Heun: failure {f0.mean():.4%}; identical to saved student mask on {(f0==smask).mean():.4%} rows; "+report("1x",lg0)[2:])

# ---------- solver fidelity ----------
P("\n## SOLVER FIDELITY (same stage t_span=1.75 each, total 7.0; baseline 18 steps/stage)")
res={}
for mult in [2,5,10,40]:
    mm=build(); mm.net.core.stage_steps=[18*mult]*4
    n=N if mult<=10 else 256
    t0=time.time(); lg=run(mm,n); dt=time.time()-t0
    s=report(f"{mult}x steps ({18*mult}/stage) n={n}, {dt:.0f}s",lg0[:n] if False else lg,n)
    # compare vs baseline (1x)
    dd=np.abs(lg-lg0[:n]); s+=f"; vs 1x-Heun: max|dlogit| {dd.max():.3e}, mean {dd.mean():.3e}, log10-param MAE {np.abs(lp(lg)-lp(lg0[:n])).mean():.4f}, max {np.abs(lp(lg)-lp(lg0[:n])).max():.4f}"
    if n==N:
        ff=fail(mm,allspecs); s+=f"; failure {ff.mean():.4%} (1x {f0.mean():.4%}); mask flips {(ff!=f0).sum()}/{N}"
    P(s); res[mult]=lg
# also 1x steps coarser (9/stage)
mm=build(); mm.net.core.stage_steps=[9]*4; lg=run(mm); ff=fail(mm,allspecs)
P(f"- 0.5x steps (9/stage): vs 1x max|dlogit| {np.abs(lg-lg0).max():.3e}; failure {ff.mean():.4%}")

# ---------- ablations ----------
P("\n## ABLATIONS (no retraining); GLN rails: NOT PRESENT in this checkpoint (no gln keys, stage.gln_rails=None)")
def zero(mm,which):
    with torch.no_grad():
        for st in mm.net.core.stages:
            if "core" in which: st.vca_W_core.zero_()
            if "bnd" in which: st.vca_v_boundary.zero_()
            if "rd" in which: st.vca_v_readout.zero_()
for name,which in [("VCA core gain=1",["core"]),("VCA boundary gain=1",["bnd"]),("VCA readout gain=1",["rd"]),("VCA ALL gains=1",["core","bnd","rd"])]:
    mm=build(); zero(mm,which); lg=run(mm); ff=fail(mm,allspecs)
    P(report(name,lg)+f"; failure {ff.mean():.4%} (baseline {f0.mean():.4%}, teacher {tmask.mean():.4%})")
# freeze_read False (continuous-time core)
mm=build()
for st in mm.net.core.stages: st.freeze_read=False
lg=run(mm); ff=fail(mm,allspecs)
P(report("freeze_read=False (core OTAs live, not sampled at stage entry)",lg)+f"; failure {ff.mean():.4%}")
mm=build()
for st in mm.net.core.stages: st.freeze_read=False
mm.net.core.stage_steps=[180]*4; lg=run(mm,256); P(report("freeze_read=False, 10x steps (n=256)",lg,256))
# shunt off
mm=build()
with torch.no_grad():
    for st in mm.net.core.stages:
        for lib in [st.cell_lib,st.boundary_cell_lib,st.output_ode_cell_lib]: lib.g_resistive_raw.fill_(-40.)
lg=run(mm); ff=fail(mm,allspecs); P(report("all resistive shunts G->0",lg)+f"; failure {ff.mean():.4%}")
# edge-gate prune sensitivity
mm=build()
with torch.no_grad():
    for st in mm.net.core.stages:
        for zl,lib in [(st.z_logits,0),(st.boundary_z_logits,0),(st.output_ode_z_logits,0)]:
            zl.copy_(torch.where(torch.sigmoid(zl)<0.1,torch.full_like(zl,-40.),zl))
lg=run(mm); ff=fail(mm,allspecs); P(report("edges with gate<0.1 hard-pruned",lg)+f"; failure {ff.mean():.4%}")
mm=build()
with torch.no_grad():
    for st in mm.net.core.stages:
        for zl in [st.z_logits,st.boundary_z_logits,st.output_ode_z_logits]:
            zl.copy_(torch.where(torch.sigmoid(zl)>0.5,torch.full_like(zl,40.),torch.full_like(zl,-40.)))
lg=run(mm); ff=fail(mm,allspecs); P(report("all edge gates hardened to {0,1} (>0.5->1)",lg)+f"; failure {ff.mean():.4%}")
# s sign / fp16? skip
open(r"tmp\analysis\scratch\audit_out_thr"+os.environ.get("THR","0.20")+".txt","w",encoding="utf-8").write("\n".join(out))

open(r"tmp\analysis\scratch\audit_out_thr"+os.environ.get("THR","0.20")+".txt","w",encoding="utf-8").write("\n".join(out))
