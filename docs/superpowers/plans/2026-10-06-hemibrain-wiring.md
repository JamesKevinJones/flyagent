# Hemibrain Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the agent's mushroom body and compass on wiring measured from the Janelia hemibrain, behind `--wiring hemibrain`, and measure what changes against today's synthetic circuits.

**Architecture:**
- **Data:** `hemibrain.py` compiles the public CC BY archive once into a small committed `data/hemibrain_mb_cx.npz`.
- **Circuits:** a new `hemibrain_circuits.py` holds two fixed-shape, graph-capturable circuits that read that file: `HemibrainMB` (real PN→KC, real KC→MBON plasticity with dopamine-derived MBON signs) and `HemibrainCX` (one rate unit per real compass neuron).
- **Selection:** `FlyBrain(wiring=...)` chooses between today's code and these circuits at construction, behind the same `inp`/`out` vectors.

**Tech Stack:** Python 3.12, torch 2.14 (existing), numpy, stdlib `csv`/`tarfile`/`urllib`. No new dependencies (no pandas).

**Spec:** `docs/superpowers/specs/2026-10-06-hemibrain-wiring-design.md`

## Global Constraints

- **No new dependencies.** `.venv` has no pandas: use stdlib `csv` and numpy.
- **`--wiring synthetic` is the default and must not change.** Regression oracle:
  `"$PY" agent_loop.py --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7` prints `{'FORAGE': 1847, 'FLEE': 25, 'ORIENT': 124, 'IDLE': 4}`, `jumps 22`, `rewards ['banana', 'geosmin']`. Also `"$PY" fruit_fly_circuits.py` keeps passing every existing synthetic assertion.
- **Source archive:** `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz`, exactly 45,872,577 bytes, cached at `.deps/hemibrain/` (already gitignored). Runtime code never downloads; only `hemibrain.py` does.
- **Counts the file must hold:** KC 1,927; MBON 68; EPG 46, EPGt 4; PEN_a(PEN1) 20, PEN_b(PEN2) 22; Delta7 42; PEG 18. PN→KC pairs keep weight ≥ 3; the median number of PN types per KC (among KCs with input) is 5.
- **MBON sign:** `sign = (ppl1 − pam)/(pam + ppl1)`, with `pam_frac` and `ppl1_frac` alongside it; an MBON with no DAN input gets 0, 0, 0.
- **Valence:** `OUT_VALENCE = −Σ sign[m]·w0·d·kc / (Σ w0·kc + 1e-6)`. One reward gives ≥ 0.1, one punishment ≤ −0.1.
- **Compass acceptance tests:**
  1. One peak, FWHM 60–120°, after 200 ticks from random activity.
  2. Drift < 5°/s over 2,000 dark ticks.
  3. Rotation gain 0.9–1.1 at 0.02, 0.1 and 0.35 rad/tick over 200 ticks.
  4. Closed-loop heading error p50 ≤ 0.05 rad with landmark gain 0.02 and gyro bias 0.0005.
- **Attribution string, verbatim:** `Janelia FlyEM hemibrain v1.2 (Scheffer et al. 2020, eLife), CC BY 4.0`
- **Page line, verbatim:** `Wiring: Janelia hemibrain v1.2 (CC BY)`
- **AGENTS.md rule 1.** Every hemibrain `step()` must be CUDA-graph-capturable: preallocated buffers, in-place writes, `index_select` rather than 0-d indexing, no host syncs, no data-dependent Python branches.
- **Tests.** Assert-based `__main__` self-checks only.
  - Project env: `PY=.venv/Scripts/python.exe` (CPU torch).
  - CUDA env: `CPY="C:/Users/kj638/Kevin codes/ComfyUI/.venv/Scripts/python.exe"`.
- **Commits:** no Claude co-author or attribution lines. Never use the section-sign character.

## Spec clarifications (decided here)

- **Bridge-glomerulus angle map.** For glomerulus number k (1–9) on side s, `θ = ((k − 1) mod 8)·45° + (22.5° if s == "L" else 0)`. This is a stand-in for the published map (Wolff et al. 2015): a global rotation or reflection of it is absorbed by the searched angular-velocity gain sign and by the landmark. Δ7 neurons covering several glomeruli (e.g. `L1L9R8`) take the circular mean of their glomeruli's angles.
- **Path integration on hemibrain wiring.** The fan-shaped-body memory is fed with an ideal cosine bump at the *decoded* heading, not with the binned EPG activity. The binned activity's population-vector gain differs from `pv_gain` and would mis-scale the home vector. The binned activity feeds only the snapshot's `bump`.
- **Module split.** The hemibrain circuits live in a new `hemibrain_circuits.py`, so `fruit_fly_circuits.py`'s synthetic path stays byte-for-byte the same apart from the `wiring` switch.

## Review Focus

1. **A CPU-only clone runs `--wiring hemibrain` offline.** Construction must read only `data/hemibrain_mb_cx.npz`, never import `hemibrain` (the downloader) and never touch the network. Test: Task 2, `offline_load`.
2. **A truncated or wrong archive is in the cache.** `hemibrain.py` refuses it with a clear error and writes no `.npz`. Test: Task 1, `wrong_size_refused`.
3. **Sim is reset, or the page's Reset is pressed.** The wiring must survive, since `Sim.reset()` rebuilds `FlyBrain`. Test: Task 4, `reset_keeps_wiring`.
4. **Long reward streaks.** Repeated dopamine must leave valence finite and within [−1, 1], with `d` clamped. Test: Task 2, `valence_saturates`.
5. **`describe()` on hemibrain outputs.** Every worded state must stay inside the System 1 vocabulary, or the precompiled tables silently miss. Test: Task 4, `describe_vocab_holds`.

---

### Task 1: `hemibrain.py` compiles the connectome into the shipped file

**Files:**
- Create: `hemibrain.py`, `data/hemibrain_mb_cx.npz` (generated), `data/DATA_LICENSE`

**Interfaces:**
- Produces:
  - `ARCHIVE_URL: str`, `ARCHIVE_SIZE = 45_872_577`, `DATA_PATH = Path("data/hemibrain_mb_cx.npz")`.
  - `ATTRIBUTION`, the attribution string from Global Constraints.
  - `fetch(cache_dir=".deps/hemibrain") -> Path`: downloads if missing; raises `ValueError` if the size isn't `ARCHIVE_SIZE`; returns the extracted directory.
  - `derive(src_dir) -> dict[str, np.ndarray]`.
  - `main()`: fetch, derive, `np.savez_compressed(DATA_PATH, **arrays)`.
  - `selfcheck(path=DATA_PATH)`.
- npz keys, consumed by Tasks 2 and 3. All arrays are numpy; strings are `<U` arrays.

| Key | Type and shape | Meaning |
|---|---|---|
| `pn_types` | (n_pn,) str | glomerulus type of each input slot |
| `pn_kc_row`, `pn_kc_col` | int32, nnz | KC index, PN-type index |
| `pn_kc_w` | float32, nnz | summed synapse counts |
| `n_kc` | int, scalar | 1,927 |
| `kc_types` | (n_kc,) str | KC subtype |
| `kc_mbon_row`, `kc_mbon_col` | int32, nnz | MBON index, KC index |
| `kc_mbon_w0` | float32, nnz | each MBON's row sums to 1 |
| `mbon_types` | (68,) str | |
| `mbon_sign`, `mbon_pam_frac`, `mbon_ppl1_frac` | float32, (68,) | per the sign rule |
| `cx_types` | (n_cx,) str | one of `EPG, EPGt, PEN1, PEN2, Delta7, PEG` |
| `cx_side` | (n_cx,) str | `L` or `R` |
| `cx_theta` | float32, (n_cx,) | radians |
| `cx_w` | float32, (n_cx, n_cx) | `[post, pre]` synapse counts |
| `meta` | (1,) str | JSON: dataset, url, size, thresholds, attribution |

- **Selection rules:**
  - PN type: matches `^[A-Za-z0-9+_]+_(adPN|lPN|vPN|lvPN|ilPN|l2PN|ivPN)\d*$` and has a KC partner.
  - Other groups by `type` prefix: `KC`, `MBON`, `PAM`, `PPL1`.
  - CX types: exact `EPG`, `EPGt`, `PEN_a(PEN1)` → `PEN1`, `PEN_b(PEN2)` → `PEN2`, `Delta7`, `PEG`.
  - Glomeruli: from `\(PB\d+\w*\)_([LR\d]+)`. For EPG, PEN and PEG the instance suffix is one glomerulus like `L3`; for Δ7 it lists several, e.g. `L1L9R8`. Side is the first letter.

- [ ] **Step 1: Write the failing self-check**

  `selfcheck(path)` asserts the counts from Global Constraints, plus:
  - median PN types per KC == 5;
  - `mbon_sign` < 0 for MBON01, MBON02 and MBON03; > 0 for MBON11 and MBON14;
  - every `cx_theta` is finite, and the largest circular gap between sorted EPG angles is < 45°;
  - `json.loads(meta[0])["attribution"] == ATTRIBUTION`.

  `wrong_size_refused`: write a 10-byte file named like the archive into a temp cache dir, and assert `fetch(tmp)` raises `ValueError`. `main()` is never reached in this case, so no `.npz` is written.

  `python hemibrain.py --selfcheck` runs both.
- [ ] **Step 2: Run it and check it fails.** Run `"$PY" hemibrain.py --selfcheck`. Expected: `FileNotFoundError` for `data/hemibrain_mb_cx.npz`.
- [ ] **Step 3: Implement `fetch`, `derive` and `main`.**
  - `fetch` streams the archive with `urllib.request.urlopen`, checks the size, then extracts it with `tarfile`. It skips `._*` macOS entries.
  - `derive` reads the neurons CSV into a dict of `bodyId → (type, instance)`, then makes one pass over the total-connections CSV, collecting the PN→KC, KC→MBON, PAM/PPL1→MBON and CX→CX weights.
  - Write `data/DATA_LICENSE`: the CC BY 4.0 notice, the attribution string, the citation (Scheffer, L. K. et al. 2020, *eLife* 9:e57443) and the source URL.
- [ ] **Step 4: Generate the file and check it passes.**
  - Run `"$PY" hemibrain.py`, then `"$PY" hemibrain.py --selfcheck`.
  - Expected: `hemibrain self-check OK`, and the file is under 2 MB (print its size).
- [ ] **Step 5: Commit.** `git add hemibrain.py data/hemibrain_mb_cx.npz data/DATA_LICENSE`, then `git commit -m "hemibrain.py: compile MB and CX wiring from the hemibrain v1.2 connectome"`.

### Task 2: Real mushroom body (`HemibrainMB`)

**Files:**
- Create: `hemibrain_circuits.py`
- Modify: `fruit_fly_circuits.py`:
  - `FlyBrain.__init__` gains `wiring="synthetic"`;
  - `_step`'s MB block delegates to `HemibrainMB` when the wiring is `hemibrain`;
  - `_state` and `vram_mb` include its tensors;
  - `demo` adds `demo_hemibrain(device)`.

**Interfaces:**
- Consumes (Task 1): npz keys `pn_types`, `pn_kc_*`, `n_kc`, `kc_mbon_*`, `mbon_*`.
- Produces:
  - `load_wiring(path="data/hemibrain_mb_cx.npz") -> dict`: numpy arrays, loaded with `np.load(..., allow_pickle=False)`.
  - `class HemibrainMB(data, device, kc_sparsity=0.05, lr=<calibrated>)`:
    - attributes: `n_pn`, `n_kc`, `k_active`, `kc` (n_kc), `kc_familiarity`, `d` (nnz of KC→MBON);
    - `step(pn: Tensor[n_pn], dopamine: Tensor[0-d], o: Tensor[N_OUT]) -> None` writes `OUT_KC_ACTIVE`, `OUT_NOVELTY` and `OUT_VALENCE`, and updates `kc`, `kc_familiarity` and `d` in place;
    - `state() -> list[Tensor]`.
  - `FlyBrain(wiring="hemibrain")`: `n_pn` and `n_kc` come from the file, and the constructor arguments for them are ignored. Odor tags, recall and goal buffers are unchanged and read `self.kc`; `self.kc` *is* `self.mb.kc` (the same tensor).
- **Algorithm** (the spec's Section 2, restated as tensor ops):
  - `drive = W_pn_kc_csr @ pn`, then the same top-k threshold as synthetic.
  - `kc_g = kc.index_select(0, kc_mbon_col)`.
  - `d.add_(lr * (relu(dopamine) * kc_g * pam_frac[row] + relu(-dopamine) * kc_g * ppl1_frac[row])).clamp_(0, 1)`. Here `pam_frac[row]` and `ppl1_frac[row]` are gathered once at init.
  - `valence = -(sign_row * w0 * d * kc_g).sum() / ((w0 * kc_g).sum() + 1e-6)`.

- [ ] **Step 1: Write the failing test.** Add `demo_hemibrain(device)` to `fruit_fly_circuits.py`, called from `__main__` for `cpu`, and for `cuda` when it's available, after `capture()`:

```python
b = FlyBrain(device=device, wiring="hemibrain"); b.capture()
assert b.n_kc == 1927 and "hemibrain" not in sys.modules            # offline_load: runtime never imports the downloader
odor_a = torch.rand(b.n_pn, generator=torch.Generator().manual_seed(1))
b.inp.zero_(); b.inp[IN_ODOR:] = odor_a; b.tick()
assert abs(float(b.out[OUT_KC_ACTIVE]) - b.mb.k_active) <= 2 and float(b.out[OUT_NOVELTY]) > 0.9
assert abs(float(b.out[OUT_VALENCE])) < 1e-6                           # untrained odor reads 0
b.remember_odor(3); b.inp[IN_DOPAMINE] = 1.0; b.tick(); b.inp[IN_DOPAMINE] = 0.0; b.tick()
assert float(b.out[OUT_NOVELTY]) < 0.05 and int(b.out[OUT_ODOR_ID]) == 3
assert float(b.out[OUT_VALENCE]) >= 0.1, float(b.out[OUT_VALENCE])    # one reward
p = FlyBrain(device=device, wiring="hemibrain"); p.inp[IN_ODOR:] = odor_a; p.tick()
p.inp[IN_DOPAMINE] = -1.0; p.tick(); p.inp[IN_DOPAMINE] = 0.0; p.tick()
assert float(p.out[OUT_VALENCE]) <= -0.1, float(p.out[OUT_VALENCE])   # one punishment
for _ in range(100):                                                   # valence_saturates
    b.inp[IN_DOPAMINE] = 1.0; b.tick()
v = float(b.out[OUT_VALENCE]); assert math.isfinite(v) and -1.0 <= v <= 1.0, v
```

- [ ] **Step 2: Run it and check it fails.** Run `"$PY" fruit_fly_circuits.py`. Expected: `TypeError: ... unexpected keyword argument 'wiring'`.
- [ ] **Step 3: Implement `load_wiring`, `HemibrainMB` and `FlyBrain(wiring=...)`.**
  - **Calibrating `lr`.** Pick the smallest value from {0.05, 0.1, 0.2, 0.5, 1.0} that passes both one-pairing thresholds, and record it as a named constant with a one-line comment.
  - **Unchanged.** `FlyBrain`'s CX on hemibrain stays the synthetic ring in this task.
  - **Unknown `wiring` value.** Raise `ValueError`.
- [ ] **Step 4: Run the checks and confirm they pass.**
  - `"$PY" fruit_fly_circuits.py`: both demos pass.
  - `"$CPY" fruit_fly_circuits.py`: the CUDA graph also captures the hemibrain MB.
  - The regression oracle (Global Constraints): unchanged.
- [ ] **Step 5: Commit.** `git commit -m "Real mushroom body: hemibrain PN->KC and KC->MBON with dopamine-derived valence"` (`git add hemibrain_circuits.py fruit_fly_circuits.py`).

### Task 3: Real compass (`HemibrainCX`), gain search, acceptance tests

**Files:**
- Modify: `hemibrain_circuits.py` (add `HemibrainCX` and `cx_acceptance`)
- Modify: `fruit_fly_circuits.py` (hemibrain CX path, plus its demo asserts)
- Create: `eval_connectome.py` (`--tune-cx` only in this task)

**Interfaces:**
- Consumes (Task 1): `cx_types`, `cx_side`, `cx_theta`, `cx_w`.
- Produces:
  - `class HemibrainCX(data, device, gains: dict[str, float], av_gain: float, landmark_gain_scale: float, substeps: int, tau_ticks: float)`.
    - `gains` keys: `"EPG>Delta7", "Delta7>all", "EPG>PEN", "PEN>EPG", "EPG>PEG", "PEG>EPG", "EPG>EPG"`. Pairs not listed get gain 0.
    - `step(angvel, landmark_heading, landmark_gain, o) -> Tensor[0-d] heading`: runs `substeps` Euler updates; writes `OUT_HEADING`; returns the heading tensor.
    - `bump16() -> Tensor[16]`: EPG activity binned into 16 sectors, normalised to sum to 1.
    - `state() -> list[Tensor]`.
  - `CX_DEFAULTS: dict`, the tuned constructor arguments, set at the end of this task.
  - `cx_acceptance(make_cx) -> dict[str, float]`. `make_cx` is a zero-argument factory. Keys:
    - `fwhm_deg`, `n_peaks` (test 1);
    - `drift_deg_per_s` (test 2);
    - `gain_0.02`, `gain_0.1`, `gain_0.35` (test 3);
    - `passed: bool` (tests 1–3 all within the Global Constraints bounds).
  - On hemibrain wiring, `FlyBrain` uses `HemibrainCX(**CX_DEFAULTS)`. `self.bump` is overwritten with `bump16()` each tick, for display. The fan-shaped body is fed `relu(cos(theta16 − heading))` normalised (see Spec clarifications).
- **Dynamics:** `r += (1/(tau_ticks·substeps))·(−r + relu(W·r + I))` per substep. Each row of `W` is the gained, signed counts divided by that row's total count.
- **Inputs:**
  - Angular velocity: `I_pen = av_gain · angvel · (+1 for side R, −1 for side L)` on PEN1 and PEN2.
  - Landmark: `I_epg = landmark_gain · landmark_gain_scale · cos(θ − landmark_heading)` on EPG and EPGt.
- **Initial state:** a cosine bump at θ = 0 on the EPGs.

- [ ] **Step 1: Write the failing test.** In `demo_hemibrain`, add:

```python
res = cx_acceptance(lambda: HemibrainCX(load_wiring(), device, **CX_DEFAULTS))
assert res["passed"], res
# 4. closed loop with landmark: same protocol as demo() section 2, on a hemibrain brain
assert drift_with_landmark <= 0.05, drift_with_landmark
```

  `drift_with_landmark` is computed exactly as `demo()` section 2 computes `drift[0.02]` (2,000 ticks of `0.1·sin(i/29)` turning, gyro bias 0.0005, landmark gain 0.02, final heading error), on a fresh `FlyBrain(device=device, wiring="hemibrain")`. Also import `CX_DEFAULTS` and `cx_acceptance`.
- [ ] **Step 2: Run it and check it fails.** Run `"$PY" fruit_fly_circuits.py`. Expected: `ImportError` for `CX_DEFAULTS`.
- [ ] **Step 3: Implement `HemibrainCX`, `cx_acceptance` and `eval_connectome.py --tune-cx`.**
  - **Grid:**
    - each listed gain in {0, 0.5, 1, 2};
    - `av_gain` in {−8, −4, −2, 2, 4, 8};
    - `substeps` in {2, 4, 8};
    - `tau_ticks` in {0.5, 1, 2};
    - `landmark_gain_scale` in {1, 5, 20}, used only for test 4.
  - **Search order.** Coordinate descent: sweep one parameter at a time, three rounds, starting from all gains 1. That's about 300 evaluations, not the full product.
  - **Output.** It prints the best passing configuration (or the closest one, with each test's value), and writes `CX_DEFAULTS` as a Python literal for pasting.
- [ ] **Step 4: Run the search and set `CX_DEFAULTS`.**
  - Run `"$PY" eval_connectome.py --tune-cx > <workspace>/tune.log`.
  - **If a configuration passes:** paste it as `CX_DEFAULTS`, with a comment naming the search ranges and the date.
  - **If none passes tests 1–3:** take the fallback in Step 4b.
- [ ] **Step 4b (fallback only): derived 16-wedge kernel.**
  - Add `derived_ring_kernel(data) -> Tensor[16,16]`: group EPGs into 16 sectors by θ, then compute the effective EPG→EPG weight as direct + via Δ7 (sign −1) + via PEN, using `cx_w` products.
  - On hemibrain wiring, `FlyBrain` then keeps the synthetic ring, with `w_ring = derived_ring_kernel(data)`, normalised to zero mean.
  - Replace the demo's acceptance asserts with the synthetic demo's sections 1–2 on this brain.
  - Ledger `Ruling: per-neuron CX failed <test>: <value>`, and save `tune.log` for README 3g.
- [ ] **Step 5: Run the checks and confirm they pass.**
  - `"$PY" fruit_fly_circuits.py` and `"$CPY" fruit_fly_circuits.py`: both pass, with the hemibrain CX captured in the CUDA graph.
  - The regression oracle: unchanged.
- [ ] **Step 6: Commit.** `git commit -m "Real compass: one rate unit per hemibrain CX neuron, tuned gains"` (or `"... derived 16-wedge kernel (per-neuron failed <test>)"` if the fallback was used).

### Task 4: `--wiring` through Sim, CLI, server and page

**Files:**
- Modify: `agent_loop.py`:
  - `Sim.__init__` gains `wiring="synthetic"`;
  - `reset()` builds `FlyBrain(..., wiring=self.wiring)`;
  - the snapshot gains `"wiring"`;
  - `main()` gains `--wiring`;
  - `run()` prints it;
  - `selfcheck` adds hemibrain blocks.
- Modify: `serve.py` (`--wiring`), `eval_goals.py` (`--wiring`), `web/index.html` (the wiring line)

**Interfaces:**
- Consumes (Tasks 2–3): `FlyBrain(wiring=...)`.
- Consumes (system1_engine): `VOCAB`, `BASE_KEYS`.
- Produces:
  - `Sim(device="cpu", n_kc=2000, backends=("rules",), system1_cpus=(), gyro_bias=0.0005, landmark_gain=0.02, wall=False, tables_dir="tables", wiring="synthetic")`;
  - snapshot key `"wiring"`.

- [ ] **Step 1: Write the failing test.** Append to `agent_loop.py`'s `selfcheck()`, importing `BASE_KEYS` and `VOCAB` from `system1_engine`:

```python
sim = Sim(wiring="hemibrain")                                         # hemibrain_reaches_and_flees
assert sim.brain.n_kc == 1927
fled, seen = False, set()
for t in range(2000):
    if t == 1000:
        sim.launch_predator()
    snap = sim.step()
    fled |= 1000 <= t < 1030 and snap["behaviour"] == "FLEE"
    seen.add(tuple(describe(sim.brain.out, sim.world.loom, sim.names, sim.goal)[k] for k in BASE_KEYS))
assert "banana" in sim.world.rewarded and fled, (sim.world.rewarded, fled)
assert all(v in VOCAB[k] for st in seen for k, v in zip(BASE_KEYS, st))  # describe_vocab_holds
sim.reset()                                                            # reset_keeps_wiring
assert sim.brain.n_kc == 1927 and sim.step()["wiring"] == "hemibrain"
sim.close()
```

- [ ] **Step 2: Run it and check it fails.** Run `"$PY" agent_loop.py --selfcheck`. Expected: `TypeError: ... unexpected keyword argument 'wiring'`.
- [ ] **Step 3: Implement.**
  - Add `--wiring` (choices `synthetic`, `hemibrain`; default `synthetic`) to `agent_loop.py`, `serve.py` and `eval_goals.py`.
  - In `web/index.html`, add `<p class="jump" id="wiring" hidden>Wiring: Janelia hemibrain v1.2 (CC BY)</p>` after `#s1`. Set `hidden = snap.wiring !== "hemibrain"` only when that changes.
- [ ] **Step 4: Run the checks and confirm they pass.**
  - `"$PY" agent_loop.py --selfcheck`, `"$PY" serve.py --selfcheck`, `"$PY" eval_goals.py --selfcheck`: all OK.
  - The regression oracle: unchanged.
  - In the Browser pane: `serve.py --wiring hemibrain --port 8767` shows the wiring line and a moving compass, and `serve.py` (synthetic) hides the line. Take a screenshot.
- [ ] **Step 5: Commit.** `git commit -m "--wiring synthetic|hemibrain through Sim, CLI, server and page"`.

### Task 5: Measure, document

**Files:**
- Modify: `eval_connectome.py` (the default mode prints the measurements)
- Modify: `README.md` (new `### 3g. Real wiring from the hemibrain connectome` after 3f, the Try-it line, and a data credit next to the self-checks); `AGENTS.md` (layout lines for `hemibrain.py`, `hemibrain_circuits.py`, `data/`); `docs/DECISIONS.md`; `docs/VERIFY.md`; `docs/STATE.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `python eval_connectome.py [--runs 3]`. For both wirings it prints:
  1. **`odor overlap`:** the mean Jaccard overlap of KC codes over 50 random odor pairs (seeded), and over 50 pairs differing in one glomerulus (one input slot replaced with a fresh random value).
  2. **`learning`:** valence after one reward, after one punishment, and on a one-glomerulus-different odor after one reward on the original.
  3. **`compass`:** heading drift in darkness over 2,000 ticks (°/s); error after 2,000 ticks with gyro bias 0.0005 and no landmark; and the same with landmark gain 0.02.
  4. **`closed loop`:** `Sim` for 2,000 ticks with the predator at 1,000: behaviour counts, rewards and jumps. Then the CLI-paced tick p50/p99 and overruns, CPU and CUDA, via `agent_loop.run` over `--runs` runs.

- [ ] **Step 1: Implement the measurements, then smoke-test them.** Run `"$PY" eval_connectome.py --runs 1`. Expected: all four blocks print for both wirings, with no exceptions.
- [ ] **Step 2: Measure.** Run `"$PY" eval_connectome.py --runs 3` and `"$CPY" eval_connectome.py --runs 3`, and save both outputs to `docs/eval-connectome-2026-10-06.txt`.
- [ ] **Step 3: Write the docs.**
  - **README 3g:**
    - an A/B table for each block;
    - the tuned `CX_DEFAULTS` (or the fallback, with the failed test and its value);
    - the dataset, the selection and the MBON sign rule in one paragraph;
    - the Known limitations from the spec.
  - **DECISIONS:** the derived shipped file, the MBON sign rule, the per-neuron compass (or the fallback), and that the default stays synthetic unless the numbers favour hemibrain. Change a default only if 3g shows hemibrain at least as good on every block, and record that call.
  - **VERIFY:** the `hemibrain.py --selfcheck`, `fruit_fly_circuits.py` and `eval_connectome.py` commands, with their expected values.
  - **STATE:** what changed, and the next step.
- [ ] **Step 4: Run the full suite one last time.** All six existing self-checks plus `hemibrain.py --selfcheck` must exit 0, and the regression oracle must hold.
- [ ] **Step 5: Commit.** `git add eval_connectome.py README.md AGENTS.md docs`, then `git commit -m "Hemibrain wiring: A/B measurements, README 3g, docs"`.
