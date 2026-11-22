# Agent plan: gain bounds + learnable clip sharpness + GLN rails

**Goal.** Two additive features on top of existing KNet (shared-sense crossbar already in):

| ID | Feature | Scope | Risk |
|---|---|---|---|
| F1 | Searchable `gm_max` / `isat_max` ∈ [1, 50] + learnable clip sharpness | BO + physics | Low |
| F2 | GLN rails: B=4 tanh half-spaces → log-gm modulation on readout + boundary families | VCA/gating path | Medium |

**Non-goals.** Gilbert multipliers, peaking/gyrators, last-stage-only readout changes, Phase-A loss edits, budget/BO trial count changes.

**Constraints (from repo conventions).**
- Analog story intact: gm bounds = OTA bias range; clip sharpness = soft rail; GLN = programmable bias-current from input-conditioned rails.
- Identity-safe init wherever possible (epoch-0 forward bit-equal to baseline when new knobs are at defaults).
- Param accounting must stay exact: analytic counter + `build_net_from_config` + `--count-params-only` must agree.
- Freeze compatibility: rails are per-sample constants → compute once at stage entry; works with `freeze_read`.
- Do not edit exploit-like code; only model/train/BO paths.

**Suggested branch.** `feat/gm-bounds-gln-rails`

---

## Phase 0 — Locate and freeze baselines (agent, 30–60 min)

### 0.1 File map (confirm paths; names may vary slightly)

| Concern | Likely files |
|---|---|
| Defaults / ranges | `config.py` (`TANH_REALISTIC_*`, `PHYS`, `VCA`, `INIT`) |
| Cell params | `cell_library.py` (`FreeTanhLibrary`) |
| Clip / RHS | `differential_stage.py` |
| Net build | `kirchhoff_net.py`, `topology.py` (`build_net_from_config`) |
| BO | `kn_bayes_opt.py`, `fixed-distillation-bayes-opt.py` (if still used) |
| Harness flags | `train_script.py`, `dagger-nuance-distillation-kirchhoffnet.py`, `fixed-mlp-distillation-kirchhoffnet.py` |
| Param counter | analytic path next to `build_net_from_config` / `_knet_param_analytic` |
| Tests | `tests/` or `test_*.py` |

### 0.2 Capture baseline numbers before any edit

```bash
# Param count on a fixed arch (adjust to repo CLI)
python train_script.py --count-params-only \
  --cell-library tanh_free --num-hidden 10 --num-stages 4 \
  --small-world-k 4 --edge-repeats 2 --temporal-readout \
  --boundary-fan-out '{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}' \
  --freeze-read --leak non-programmable --vca --vca-core \
  --vca-separate-core-bus --interstage-activation residual-relu-tanh
# Record TRAINABLE_PARAMS=N0
```

Also note current defaults: `gm_min/gm_max`, `isat_min/isat_max`, clip formula constants (`0.05`, `0.02`, `x_max`).

### 0.3 Acceptance for Phase 0

- [ ] N0 recorded
- [ ] Clip formula and gm mapping quoted from code into the PR description
- [ ] Confirm shared-sense crossbar is the active readout path (flag name + default)

---

## Phase F1 — Gain bounds + learnable clip sharpness

### F1.1 Config / defaults

In `config.py` (or equivalent):

```python
# Existing style; extend ranges used by BO + cell factory
TANH_FREE_GM_MIN = 0.01
TANH_FREE_GM_MAX = 10.0          # build default (unchanged for non-BO runs)
TANH_FREE_ISAT_MIN = 1e-6        # or 0.01 — match current code
TANH_FREE_ISAT_MAX = 10.0

# NEW: BO search windows (not forced build defaults)
BO_GM_MAX_RANGE = (1.0, 50.0)
BO_ISAT_MAX_RANGE = (1.0, 50.0)

# Clip
PHYS_X_MAX_DEFAULT = 3.0
PHYS_CLIP_SCALE = 0.05           # amplitude in front of (σ−σ)
PHYS_CLIP_SHARPNESS = 0.02       # denominator; smaller = sharper
PHYS_CLIP_SHARPNESS_MIN = 1e-3
PHYS_CLIP_SHARPNESS_MAX = 0.2
# learnable: softplus(raw) mapped into [min, max] or free softplus with floor
```

**Rule:** build-time default `gm_max=10` stays so old checkpoints / Friedman recipes do not silently change. BO *suggests* higher caps.

### F1.2 Wire `gm_max` / `isat_max` through the stack

**Cell library** — `FreeTanhLibrary` already does:

```text
gm = gm_min + (gm_max - gm_min) * sigmoid(gm_raw)
```

Ensure constructor takes `gm_min, gm_max, isat_min, isat_max` and does **not** close over module-level constants only.

**Factory / stage / net:**

```python
# pseudocode — every path that builds cells
def make_cell_library(name, E, gm_max=None, isat_max=None, **kw):
    gm_max = gm_max if gm_max is not None else config.TANH_FREE_GM_MAX
    isat_max = isat_max if isat_max is not None else config.TANH_FREE_ISAT_MAX
    ...
```

Propagate from:

1. CLI `--gm-max` / `--isat-max` (may already exist — verify)
2. `build_net_from_config(cfg)` reading `cfg["gm_max"]`, `cfg["isat_max"]`
3. BO trial → cfg dict → harness subprocess argv

**Boundary + temporal/readout cell libs** must use the **same** bounds (or document if separate). Shared-sense OTAs too.

### F1.3 Learnable clip sharpness

**Current clip (from README):**

```text
clip(x) = 0.05 * (σ((x - x_max)/0.02) - σ((-x - x_max)/0.02))
```

**Target:**

```text
s = clip_sharpness   # per-stage or per-net scalar > 0
clip(x) = clip_scale * (σ((x - x_max)/s) - σ((-x - x_max)/s))
```

**Implementation sketch in `DifferentialStage` (or wherever clip lives):**

```python
class DifferentialStage(nn.Module):
    def __init__(..., x_max, clip_scale=0.05,
                 learnable_clip_sharpness=False,
                 clip_sharpness_init=0.02,
                 clip_sharpness_min=1e-3,
                 clip_sharpness_max=0.2):
        ...
        self.x_max = x_max
        self.clip_scale = clip_scale
        self.clip_sharpness_min = clip_sharpness_min
        self.clip_sharpness_max = clip_sharpness_max
        if learnable_clip_sharpness:
            # inverse-sigmoid init so mapped value == clip_sharpness_init
            t = (clip_sharpness_init - clip_sharpness_min) / (
                clip_sharpness_max - clip_sharpness_min
            )
            t = float(np.clip(t, 1e-4, 1 - 1e-4))
            raw0 = np.log(t / (1 - t))
            self.clip_sharpness_raw = nn.Parameter(
                torch.tensor(raw0, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "clip_sharpness_raw",
                torch.tensor(clip_sharpness_init, dtype=torch.float32),
            )
            self._fixed_sharpness = clip_sharpness_init

    def clip_sharpness(self):
        if isinstance(self.clip_sharpness_raw, nn.Parameter):
            t = torch.sigmoid(self.clip_sharpness_raw)
            return self.clip_sharpness_min + (
                self.clip_sharpness_max - self.clip_sharpness_min
            ) * t
        return self.clip_sharpness_raw

    def soft_clip(self, x):
        s = self.clip_sharpness()
        return self.clip_scale * (
            torch.sigmoid((x - self.x_max) / s)
            - torch.sigmoid((-x - self.x_max) / s)
        )
```

**Scope choice (pick one, document in flag):**

| Mode | Params | Recommendation |
|---|---|---|
| A. One sharpness per net | 1 | Simplest; good first cut |
| B. One per stage | `num_stages` | Preferred — stages can stiffen independently |
| C. Per-node | `N * stages` | Skip for F1 |

**Default:** `learnable_clip_sharpness=False` (fixed 0.02) so non-flag runs stay bit-identical. BO / Phase-A can enable it.

**Optimizer group:** put `clip_sharpness_raw` in **dynamic** LR group (`dyn`), not struct/mapper.

### F1.4 CLI flags

```text
--gm-max FLOAT          default 10.0
--isat-max FLOAT        default 10.0
--gm-min FLOAT          default 0.01   # if not already
--isat-min FLOAT        default ...
--learnable-clip-sharpness / --no-learnable-clip-sharpness
--clip-sharpness-init FLOAT   default 0.02
--clip-sharpness-min FLOAT    default 1e-3
--clip-sharpness-max FLOAT    default 0.2
```

### F1.5 BO dimensions (`kn_bayes_opt.py` and any fixed-distillation BO)

Add continuous log-or-linear suggests (match existing style for `x_max`):

```python
# pseudocode inside suggest()
gm_max = trial.suggest_float("gm_max", 1.0, 50.0, log=True)
isat_max = trial.suggest_float("isat_max", 1.0, 50.0, log=True)
# optional companion
learnable_clip = trial.suggest_categorical(
    "learnable_clip_sharpness", [True, False]
)
# or always-on True in CTLE Phase-A only — prefer always searchable bool
```

Pass into trial command:

```text
--gm-max {gm_max} --isat-max {isat_max} \
[--learnable-clip-sharpness]
```

**Fingerprint:** include `gm_max_range`, `isat_max_range`, `clip_sharpness_search` in sampler fingerprint so old studies refuse resume.

**Feasible arch list:** gm/isat max are **param-count neutral** (same number of tensors). No change to analytic param windows. Learnable sharpness adds 1 or `num_stages` params — update analytic counter:

```python
extra = num_stages if cfg.learnable_clip_sharpness else 0
```

### F1.6 Tests (F1)

```python
def test_gm_max_scales_cell_range():
    lib_lo = make_cell_library("tanh_free", E=4, gm_max=10.0)
    lib_hi = make_cell_library("tanh_free", E=4, gm_max=50.0)
    # with gm_raw → +∞, gm approaches gm_max
    ...

def test_clip_sharpness_init_identity():
    # fixed vs learnable at init=0.02: soft_clip(x) allclose

def test_clip_sharpness_in_param_count():
    # learnable on → +num_stages (or +1) vs off

def test_epoch0_forward_equal_when_defaults():
    # same seed, learnable off, gm_max=10: bitwise or atol=0 match
```

### F1.7 Phase F1 acceptance

- [ ] Default path: N_params == N0 (learnable off)
- [ ] Learnable on: N_params == N0 + S (S = stages or 1)
- [ ] BO can suggest gm_max=25, isat_max=25 without crash
- [ ] RHS uses `soft_clip` with live sharpness (grad flows: one backward smoke)
- [ ] Fingerprint bump documented

---

## Phase F2 — GLN rails (B=4, log-gm, readout + boundary)

### F2.1 Math (implement exactly)

For input \(u \in \mathbb{R}^{d_{in}}\) (already scaled knet specs):

```text
# Rails (shared across stages or per-stage — default SHARED)
z_b = tanh(α_b * (a_b · u + c_b))     b = 1..B
# α_b > 0 via softplus; a_b ∈ R^{din}; c_b bias

# Per gated edge e in {boundary ∪ readout_sense_or_crossbar_targets}:
log_gm_e = log(gm0_e + eps) + Σ_b W_{e,b} * z_b
gm_e = clip(exp(log_gm_e), gm_min, gm_max)
```

**What “log-gm” replaces:** the static `gm` from `gm_raw` sigmoid map becomes the **base** `gm0`; rails add a multiplicative modulation in linear gm via additive log-space.

**Identity init (mandatory):**

```text
W_{e,b} = 0  →  gm_e = gm0_e
α init small or moderate; a, c init so z≈0 or small
```

Epoch-0 forward must match no-GLN model when `W=0`.

### F2.2 Where to attach (families only)

| Family | Apply GLN? | Notes |
|---|---|---|
| Boundary OTAs | **Yes** | Primary input-conditioned path |
| Readout | **Yes** | Shared-sense OTAs **or** crossbar path — see below |
| Core hidden edges | **No** (v1) | Keeps param cost down; flag for later |
| Resistive shunt | **No** | Stability backbone — never gate shunt |

**Shared-sense + crossbar interaction.** Prefer modulating **sense OTA gm** (one per hidden node), not each crossbar weight:

```text
I_j = gm_j(u) * Isat_j * (tanh(u_j) + 0.01*u_j)   # schematic
a_dot = W_crossbar @ I_sense - leak*a - clip(a)
```

If readout is still dense OTAs in some builds, modulate those edge gms instead. Detect via flag `readout_kind in {crossbar, dense_ota, ...}`.

### F2.3 Module design

```python
class GLNRails(nn.Module):
    """Input-conditioned rails + per-edge log-gm deltas."""
    def __init__(self, in_dim, n_edges, B=4,
                 gm_min=0.01, gm_max=10.0,
                 alpha_init=1.0, w_init_std=0.0):
        super().__init__()
        self.B = B
        self.gm_min = gm_min
        self.gm_max = gm_max
        # rails
        self.a = nn.Parameter(torch.zeros(B, in_dim))      # or N(0, 0.01)
        self.c = nn.Parameter(torch.zeros(B))
        self.alpha_raw = nn.Parameter(
            torch.full((B,), inv_softplus(alpha_init))
        )
        # edge mixing: factorized to control params
        # Option A dense: W [n_edges, B]
        # Option B factorized (preferred if n_edges large):
        #   p[e,r] @ q[r,B] with r=rank_gln (e.g. 2)
        self.use_factorized = True
        self.rank = 2
        if self.use_factorized:
            self.P = nn.Parameter(torch.zeros(n_edges, self.rank))
            self.Q = nn.Parameter(torch.zeros(self.rank, B))
        else:
            self.W = nn.Parameter(torch.zeros(n_edges, B))  # identity gate

    def rails(self, u):
        # u: [Bch, din]
        alpha = F.softplus(self.alpha_raw) + 1e-6
        # z: [Bch, B]
        z = torch.tanh(alpha * (u @ self.a.T + self.c))
        return z

    def modulate_gm(self, gm0, u):
        # gm0: [E] or [Bch, E]
        z = self.rails(u)  # [Bch, B]
        if self.use_factorized:
            W = self.P @ self.Q  # [E, B]
        else:
            W = self.W
        # delta: [Bch, E]
        delta = z @ W.T
        log_gm = torch.log(gm0.clamp_min(1e-8) + 1e-8) + delta
        gm = torch.exp(log_gm).clamp(self.gm_min, self.gm_max)
        return gm
```

**Param cost estimate (report in PR):**

```text
Rails shared: B*(din + 1) + B  (a, c, alpha) ≈ 4*(4+1)+4 = 24
Per family factorized: E*rank + rank*B
Boundary E≈8: 8*2+2*4 = 24
Sense E≈h=10: 10*2+2*4 = 28
× stages if per-stage modules; prefer share rails, per-stage P/Q or share all
```

**Recommendation:** **one shared `GLNRails` core** (a,c,α) for the whole net; **separate P/Q (or W)** per family (boundary vs readout), optionally **tied across stages** for v1 to limit params. Flag:

```text
--gln-rails / --no-gln-rails
--gln-B 4
--gln-rank 2
--gln-families boundary,readout   # csv
--gln-share-rails-across-stages   # default True
--gln-alpha-init 1.0
```

### F2.4 Integration points

**A. Boundary current path** (`differential_stage` / boundary scatter):

```python
# before: gm from cell library static map
gm0, Isat, A, B, s, theta, G, gate = cell.params_for_edges(...)
if self.gln_boundary is not None:
    gm = self.gln_boundary.modulate_gm(gm0, u)
else:
    gm = gm0
u_e = gm * (s * (A * Vsrc - B * Vdst) + theta)
I_tanh = Isat * (torch.tanh(u_e) + 0.01 * u_e)
I = gate * vca * compliance * I_tanh + G * (Vsrc - Vdst)
```

**B. Readout sense path** (post-crossbar design):

```python
gm_sense = self.gln_readout.modulate_gm(gm0_sense, u)  # if enabled
# then I_sense(x, gm_sense) → crossbar
```

**C. Freeze / VCA coexistence**

- Compute `z = rails(u)` **once per stage entry** (u constant per sample).
- Cache `gm_mod` if `freeze_boundary` / freeze sense — but boundary freeze is usually off; still cache rails for speed.
- Existing VCA multiplies current; GLN modulates gm. **Both can be on:**  
  `I *= vca_gate`; `gm = gln(gm0)`. Document order: GLN inside tanh argument, VCA outside on current (current code style).

**D. Do not** apply GLN to shunt G.

### F2.5 Init and LR groups

| Param | Init | LR group |
|---|---|---|
| `a, c` | 0 or tiny N(0,0.01) | dyn or struct — prefer **dyn** |
| `alpha_raw` | softplus⁻¹(1.0) | dyn |
| `P, Q` or `W` | **0** | dyn (or struct if you treat as gates) |

Identity at W=0 is the load-bearing property — unit test it.

### F2.6 Param counter + BO

```python
def gln_param_count(in_dim, E_bound, E_read, B, rank, share_rails, stages):
    rails = B * (in_dim + 2)  # a, c, alpha — once if share
    fam = lambda E: E * rank + rank * B
    body = fam(E_bound) + fam(E_read)
    if not share_rails:
        rails *= stages
        body *= stages
    elif not share_edge_mix_across_stages:
        body *= stages
    return rails + body
```

Add to `_knet_param_analytic` / `build_net_from_config` count path.

**BO:** start with **flags fixed** (gln on, B=4, rank=2), not free dims — reduces search explosion. Optional later:

```python
use_gln = trial.suggest_categorical("gln_rails", [True, False])
```

Fingerprint must include gln schema.

### F2.7 Interaction with F1

`gm_max` from F1 is the **clamp ceiling** after `exp(log_gm0 + Wz)`. Base `gm0` still from sigmoid map into `[gm_min, gm_max]`. After modulation, clamp again to same window so BO’s gm_max remains meaningful.

```python
gm = exp(log(gm0)+delta).clamp(gm_min, gm_max)
```

### F2.8 Tests (F2)

```python
def test_gln_identity_at_zero_W():
    # W/P/Q = 0 → modulate_gm(gm0,u) == gm0 for random u

def test_gln_grad_flows():
    # loss = modulate_gm(...).sum(); backward; a.grad is not None

def test_gln_param_count_matches_modules():
    # analytic vs sum(p.numel() for p in gln.parameters())

def test_gln_does_not_touch_shunt():
    # structural: G path independent

def test_forward_match_baseline_gln_off_or_W0():
    # full KirchhoffNetWithIO smoke, atol tight
```

### F2.9 Smoke training (optional, 1 GPU-hour)

Fixed arch trial-24-like, 20 epochs, GLN on vs off, lr groups 1/4/1:

- Confirm no NaN
- Confirm `||W||` or `||P||` moves if loss requires it (log mean |delta log_gm|)

---

## Phase F3 — Wiring checklist for agent (ordered commits)

### Commit 1 — F1 plumbing only
1. Config constants + CLI
2. Cell library + `build_net_from_config` pass-through
3. Learnable clip in stage RHS
4. Param counter + tests
5. BO suggest gm_max/isat_max + fingerprint

### Commit 2 — F2 GLN module
1. `gln_rails.py` (or section in `differential_stage.py`)
2. Hook boundary + readout sense only
3. Flags default **off**
4. Analytic params + identity tests

### Commit 3 — Enable paths for CTLE / Phase-A
1. Harness argv pass-through
2. Document recommended command:

```bash
python kn_bayes_opt.py ... \
  --gm-max-search \          # or free float dims
  --learnable-clip-sharpness \
  --gln-rails --gln-B 4 --gln-rank 2 \
  --gln-families boundary,readout
```

3. Short `docs/` note: init identity, families, no shunt, freeze OK

---

## Phase F4 — Verification matrix

| Check | Command / method | Pass |
|---|---|---|
| Param parity default | `--count-params-only` vs N0 | equal |
| Param +clip | learnable on | N0+S |
| Param +GLN | gln on | N0 + analytic GLN |
| Epoch-0 equal | GLN W=0, clip init 0.02, gm_max=10 | max abs diff < 1e-6 |
| Backward | one batch train step | finite grads, no NaN |
| BO dry | 1 trial suggest + build | runs |
| Freeze_read | still freezes **core** only; GLN gm from u cached | OK |
| Crossbar | sense gm modulated; crossbar W unchanged by GLN | OK |

---

## Explicit non-goals / traps for the agent

1. **Do not** put GLN on core edges in v1.
2. **Do not** modulate resistive shunt.
3. **Do not** change default `gm_max=10` for non-BO scripts (Friedman reproducibility).
4. **Do not** budget-count GLN as free — update analytic counter or BO will accept illegal arches.
5. **Do not** use high fixed α (e.g. 50) without softplus+init; dead rails kill grads.
6. **Do not** double-apply VCA and GLN in a way that breaks identity: VCA W=0 → gate 1; GLN W=0 → gm=gm0.
7. Shared-sense path: modulate **sense** gm, not 7× duplicate — otherwise you reintroduce readout param blowup.
8. If `edge_repeats>1` on core only, confirm boundary/readout edge counts used for GLN `n_edges` match actual tensors.

---

## Minimal pseudocode: stage RHS with both features

```python
def rhs(self, x, u, frozen_core_I=None):
    acc = zeros_like(x)

    # core tanh (frozen or live)
    if frozen_core_I is not None:
        acc = acc + frozen_core_I
    else:
        acc = acc + scatter_core_tanh(x, gm=self.core_gm0)  # no GLN v1

    acc = acc + scatter_core_resistive(x)  # always; no GLN

    # boundary: GLN on gm
    gm_b = self.boundary_gm0
    if self.gln_boundary is not None:
        gm_b = self.gln_boundary.modulate_gm(gm_b, u)
    acc = acc + scatter_boundary(u, x, gm=gm_b) * vca_bound(u)

    # readout sense: GLN on gm, then crossbar
    gm_r = self.readout_sense_gm0
    if self.gln_readout is not None:
        gm_r = self.gln_readout.modulate_gm(gm_r, u)
    I_sense = sense_ota(x_hidden, gm=gm_r)
    acc_tail = crossbar(I_sense) + ...  # accumulators
    acc = write_accumulator_part(acc, acc_tail)

    s = self.clip_sharpness()  # F1 learnable or buffer
    return (acc - self.leak * x - soft_clip(x, self.x_max, s)) / C_eff
```

---

## Done definition

Agent is done when:

1. F1 merged: searchable gm/isat max + optional learnable clip sharpness; tests green; default bit-identical.
2. F2 merged: `--gln-rails` off by default; on → boundary+readout log-gm via B=4 rails; W=0 identity; param counter correct.
3. One markdown note lists flags, inits, families, and a dry-run command.
4. No full-file dumps in the PR description — diffs only.

---

## Suggested agent prompt (copy-paste)

```text
Implement F1 and F2 per docs/plan-gm-gln.md (this plan).

Order: Phase 0 baselines → Commit F1 → Commit F2 → verification matrix.
Do not implement Gilbert multipliers, peaking, or core-edge GLN.
Keep default gm_max=10 and GLN off so existing recipes stay stable.
Shared-sense crossbar is already landed: GLN modulates sense OTA gm only.
Update analytic param counts and BO fingerprints.
Add unit tests listed in F1.6 and F2.8; run them.
Do not write huge new files from scratch if a module already owns the path —
extend cell_library, differential_stage, build_net_from_config, kn_bayes_opt.
```

If the agent hits ambiguous readout API names after the crossbar refactor, it should grep `crossbar`, `sense`, `OutputAffine`, `temporal` and attach GLN at the **single sense-gm** call site, then stop and report the symbol names rather than inventing a second readout path.

---

# Grounded addendum (code-verified, 2026-09-08) — locked decisions: per-stage clip · shared GLN rails · fixed-on BO flags · train_script + kn_bayes_opt scope

## What the codebase actually says (corrects the plan above)

1. **F1 gm/isat search already exists — do not re-implement.** `train_script.py:2185-2204` already has `--gm-min/--gm-max/--isat-min/--isat-max` (None → config `TANH_REALISTIC_*` 0.01/10.0/0.01/10.0), wired at `train_script.py:3408-3409` into `make_cell_library`. `cell_library.py:515-584` factory + `FreeTanhLibrary:261-285` already accept/validate all four rails; gm map `cell_library.py:326-329`. `kn_bayes_opt.py:790-803` already searches gm_max/isat_max log-uniform [1,50]; START_POINTS pins 10.0; suggest at `kn_bayes_opt.py:1650-1653`, argv at `kn_bayes_opt.py:493-495`. F1-gm delta = none; keep build default 10.0.
2. **Clip formula (quote into PR):** `differential_stage.py:1409-1411` — `clip = σ((x−x_max)/clip_softness) − σ((−x−x_max)/clip_softness); clip_term = clip_current·clip`; defaults `config.py:302-308` (x_max 3.0, clip_current 0.05, clip_softness 0.02); stored as floats `differential_stage.py:251-254`. Gap: `build_net_from_config` (`topology.py:1853-1881`) takes `x_max/c_eff` but NOT clip rails, while `topology_to_stage` (`topology.py:1309-1315`) does — thread new sharpness through both + `prune_stage` rebuild (`topology.py:1031-1033`).
3. **Readout reality:** `--readout {linear,temporal,shared,shared-x2}` (`train_script.py:2685-2699`) → `readout_mode ota_mesh/shared_sense` (`train_script.py:3607-3623`); `--temporal-readout` is a deprecated alias (`3611-3613`). Shared-sense VCA gates sense bank only (`differential_stage.py:603-607`, `1395-1398`). GLN attaches at that single sense-gm site. `--count-params-only` exists (`train_script.py:2285-2287`, exit `3735`) — Phase-0 command is valid after swapping in `--readout shared`.
4. **Param counting:** `bo_param_sampling._knet_param_count_cached` (`bo_param_sampling.py:214-292`) builds the net exactly; docstring confirms gm/isat/x_max/lr/wd/batch/freeze never move count. Cache key has NO clip/GLN dims — must extend or feasible lists go stale. `fixed-distillation-bayes-opt.py:147-379` is a separate counter/sampler (KNet: only kn_x_max/solver_scale/wd, no gm dims) — OUT of scope, leave untouched.
5. **LR groups:** `train_script.py:1982-2001` — mapper / z_logits+VCA→struct / raw_leak,raw_drive_g→dyn / else→other. New `clip_sharpness_raw` + GLN (a,c,alpha,P,Q) → **dyn** in both `make_optimizer` and `compute_update_norms`.
6. **Freeze/VCA points:** boundary frozen `differential_stage.py:914-950` vs live; sense frozen `998-1016` vs live incl. shunt `1392-1404`; core gate hoisted `1497-1499`. GLN `z=rails(u)` once per stage entry, folded into frozen tensors when freeze flags on. Never gate shunt G or core edges.
7. **Prune inherits no-shared limit:** `prune_stage` hard-errors on shared-sense (`topology.py:790-795`), rebuilds with `vca_enabled=False` (`1043-1045`). GLN-on-shared = no-prune v1, document only.

## Phase 0 (corrected)

```bash
python train_script.py --count-params-only \
  --cell-library tanh_free --num-hidden 10 --num-stages 4 \
  --small-world-k 4 --edge-repeats 2 --readout shared \
  --boundary-fan-out '{"0":[0,4],"1":[1,5],"2":[2,6],"3":[3,7]}' \
  --freeze-read --leak non-programmable --vca --vca-core \
  --vca-separate-core-bus --interstage-activation residual-relu-tanh
# Record N0; confirm --cell-library flag name before running
```

## Commit 1 — learnable per-stage clip sharpness

1. `config.py`: `PHYS_CLIP_*` (INIT 0.02/MIN 1e-3/MAX 0.2; scale stays 0.05); document `BO_GM/ISAT_MAX_RANGE=(1,50)` as existing search.
2. `differential_stage.py`: per-stage `clip_sharpness_raw` (logit-init so mapped==0.02) + `clip_sharpness()` + `soft_clip()`; rhs uses it; off-path = fixed buffer, bit-identical.
3. `topology.py`: thread clip kwargs through `topology_to_stage` + `build_net_from_config` (cfg.get fallback) + `prune_stage` rebuild.
4. `train_script.py`: `--learnable-clip-sharpness` (default off) + init/min/max flags; dyn grouping; count reflects +S.
5. `bo_param_sampling.py` key += `learnable_clip`; `kn_bayes_opt.py` fixed flag pass-through + fingerprint += `clip_sharpness_search`, `gm_max_range`, `isat_max_range`.
6. Tests: init-identity allclose; N0 vs N0+S; epoch-0 equal; backward grad finite.

## Commit 2 — shared-rail GLN (B=4, rank=2, boundary+sense)

1. `gln_rails.py`: shared `(a,c,alpha)` core + per-stage per-family `P[E,rank]@Q[rank,B]` (all zero-init → `gm=gm0`); `modulate_gm` clamps to `[gm_min,gm_max]`.
2. `DifferentialStage`: hook boundary (frozen+live) and sense (frozen+live); `z` cached per entry; GLN inside tanh arg, VCA outside on current.
3. Flags default off: `--gln-rails --gln-B 4 --gln-rank 2 --gln-families boundary,readout --gln-share-rails-across-stages` (True). Breakdown bucket + dyn group.
4. Counter key += `(gln_on,B,rank,families)` + `gln_param_count` (rails `B·(din+2)` once + `E·rank+rank·B` per family × stages policy) == built net == `--count-params-only`.
5. Tests: W=0 identity; grad to `a`; count match; shunt untouched; full-net W0 vs off `atol<1e-6`.

## Verification matrix

Default == N0; clip == N0+S; GLN == N0+analytic; epoch-0 <1e-6; backward finite; 1-trial BO builds; freeze preserved; `crossbar_W` untouched. No core/shunt GLN, no default gm change, no fixed-BO edits. Dagger/CTLE Phase-A wiring landed 2026-09-08 (`--kn-learnable-clip-sharpness`, `--kn-gln-rails*` on the harness; Phase-A cmd forwards them; legacy dagger path still refuses).

---