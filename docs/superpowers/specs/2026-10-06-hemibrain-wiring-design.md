# Real wiring from the hemibrain connectome: design

**Date:** 2026-10-06 · **Status:** approved in conversation, awaiting written-spec review
**Sub-project 3 of 3.** Order: (1) free-text goals (merged) → (2) precompiled System 1 (merged) → (3) real wiring.

## Purpose and success

Portfolio piece. Replace the hand-built mushroom body (MB) and central complex (CX) with wiring measured from a real fly
brain. Keep the agent, the 15 ms tick and every interface, then measure what changes. Success means:

1. `--wiring hemibrain` runs the same agent on Janelia hemibrain v1.2 wiring:
   - 1,927 real Kenyon cells (KCs) fed by real projection neurons (PNs);
   - 68 real mushroom-body output neurons (MBONs) whose learning and valence come from the connectome;
   - a compass made of about 200 real CX neurons, one rate unit each.
2. A clone runs it offline from a small committed file. `python hemibrain.py` rebuilds that file from the public
   archive.
3. README section 3g reports A/B numbers (`synthetic` vs `hemibrain`) for odor discrimination, learning, compass
   drift, closed-loop behaviour and tick timing, failures included.
4. Nothing regresses:
   - `--wiring synthetic` is the default and keeps today's code path, so the regression oracle is unchanged;
   - System 1 tables stay valid;
   - every self-check passes.

## Decisions (from the brainstorm)

| Decision | Chosen | Rejected because |
|---|---|---|
| What "full connectome" means here | Real wiring inside the existing agent | A whole-brain spiking model has an unmeasurable behaviour claim (which neurons to read out is open research) |
| Dataset | Janelia hemibrain v1.2: CC BY 4.0, public 45.9 MB archive, no login | FlyWire is CC BY-NC 4.0; neuPrint needs a personal token |
| Circuits | MB and CX, MB first | MB only leaves the compass hand-built; CX only skips the near-certain win |
| CX model | One rate unit per real neuron; derived 16-wedge kernel as the named fallback | Starting with the derived kernel gives a weaker claim |
| MB output | Real KC→MBON synapses; MBON valence sign derived from dopamine-neuron (DAN) input | Today's single readout makes "real MB" half true; a typed-in compartment table is literature, not measurement |
| Shipping | Compile once into a small derived `.npz`, committed with attribution | Parsing 3.5M rows at startup slows every start and forces a 46 MB download on every clone |

## Scope

**In:**
- `hemibrain.py` (download, extract, derive, self-check);
- `data/hemibrain_mb_cx.npz` and `data/DATA_LICENSE`;
- `FlyBrain(wiring=...)` with the real MB and the per-neuron CX;
- the CX gain search and its acceptance tests;
- the `--wiring` flag everywhere a `Sim` is built;
- the page attribution line;
- `eval_connectome.py`;
- README 3g, DECISIONS, VERIFY, STATE.

**Out (YAGNI):**
- the 257 ring neurons and PFL3 modelled per neuron (the landmark cue and goal heading drive keep their abstractions);
- fan-shaped-body path integration per neuron;
- DAN dynamics (dopamine stays a world signal);
- both hemispheres;
- FlyWire and the ventral nerve cord connectome (MANC);
- spiking models;
- making `hemibrain` the default before it's measured.

## 1. `hemibrain.py` and the shipped file

`python hemibrain.py` writes `data/hemibrain_mb_cx.npz`. It uses stdlib `csv`, `tarfile` and `urllib`, plus numpy;
pandas is not a dependency.

**Source.** It downloads `https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz` into
`.deps/hemibrain/` (gitignored) once, and refuses the archive unless its size is 45,872,577 bytes. It reads
`traced-neurons.csv` (`bodyId,type,instance`) and `traced-total-connections.csv` (`bodyId_pre,bodyId_post,weight`).

**Selection and derivation:**
- **PNs:**
  - neurons whose `type` is an olfactory PN type (glomerulus + `PN` suffix, e.g. `DA1_lPN`, `VP1m+VP2_lvPN2`) and that
    synapse onto a KC;
  - grouped by type, so every PN of a glomerulus shares one input slot (the receptor neurons aren't traced);
  - `n_pn` = the number of such types (about 90, exact value stored).
- **KCs:**
  - all neurons whose `type` starts with `KC`: 1,927;
  - `w_pn_kc[kc, pn_type]` = summed synapse counts, keeping pairs with at least 3 synapses;
  - KCs with no PN input (about 170) stay in and simply never fire.
- **MBONs:**
  - all neurons whose `type` starts with `MBON`: 68;
  - `w_kc_mbon[mbon, kc]` = synapse counts (no threshold), normalised so each MBON's row sums to 1;
  - per MBON, `pam` and `ppl1` = summed synapses from neurons typed `PAM*` and `PPL1*`;
  - `pam_frac = pam/(pam+ppl1)`, `ppl1_frac = ppl1/(pam+ppl1)`, `sign = (ppl1 − pam)/(pam + ppl1)`;
  - an MBON with no DAN input gets fractions 0 and sign 0.
- **CX:**
  - all neurons typed `EPG`, `EPGt`, `PEN_a(PEN1)`, `PEN_b(PEN2)`, `Delta7`, `PEG` (about 200);
  - `w_cx[post, pre]` = synapse counts among them;
  - per neuron: type index, side (`L`/`R` from the instance name), protocerebral-bridge glomerulus (parsed from
    names like `EPG(PB08)_L3`);
  - compass angle θ from the published bridge-to-ellipsoid-body map (Wolff et al. 2015; Hulse et al. 2021). That
    map is a small table in `hemibrain.py`, labelled *published* in a comment;
  - Δ7 neurons span several glomeruli (e.g. `L1L9R8`), so their θ is the circular mean of the angles they cover.
- **Provenance:** dataset `hemibrain:v1.2`, source URL, archive size, the selection thresholds, and the attribution
  string "Janelia FlyEM hemibrain v1.2 (Scheffer et al. 2020, eLife), CC BY 4.0".

**Shipping.** The file is a few hundred KB, saved with `np.savez_compressed`. It's committed with
`data/DATA_LICENSE` (CC BY 4.0 notice and citation) and a README credit.

**Self-check.** `python hemibrain.py --selfcheck` reads the committed file, with no download, and checks:
- KC 1,927; MBON 68; EPG 46 + EPGt 4; PEN1 20, PEN2 22; Δ7 42; PEG 18;
- the median number of PN types per KC is 5 (among KCs with input);
- signs: MBON01, MBON02 and MBON03 < 0 (avoidance); MBON11 and MBON14 > 0 (approach);
- every CX neuron has a finite θ, and EPG angles cover the circle (largest gap under 45°).

## 2. Mushroom body on real wiring

`FlyBrain(wiring="hemibrain")` loads the file, and `n_pn`/`n_kc` come from it. `World` builds odor patterns from
`brain.n_pn`, so the sources become random patterns over real glomeruli with no other change.

- **Input layer.**
  - The antennal-lobe normalisation is unchanged.
  - `w_pn_kc` is the real CSR matrix of synapse counts.
  - APL is still k-winners-take-all at 5%. The connectome's single APL inhibits every KC, which is the global
    inhibition the k-WTA rule stands for.
  - Novelty and familiarity are unchanged.
- **KC→MBON plasticity** (Aso et al. 2014: dopamine depresses the active KCs' synapses in its compartment). Each
  pair keeps its fixed weight `w0` (from the file) and a learned depression `d ∈ [0, 1]`, stored alongside `w0` in
  the same sparsity pattern:
  - reward (`IN_DOPAMINE > 0`): `d += lr · dopamine · kc · pam_frac[m]`;
  - punishment (`IN_DOPAMINE < 0`): `d += lr · |dopamine| · kc · ppl1_frac[m]`;
  - `d` is clamped to `[0, 1]`.
- **Valence,** measured against the naive network so an untrained odor reads 0:
  `OUT_VALENCE = −Σ_m sign[m] · Σ_k w0[m,k]·d[m,k]·kc[k] / (Σ_m Σ_k w0[m,k]·kc[k] + ε)`.
  Reward depresses avoidance MBONs (sign < 0), which gives positive valence; punishment depresses approach MBONs,
  which gives negative valence.
- **Calibration.** `lr` is set so that one reward contact gives `OUT_VALENCE ≥ 0.1` on the rewarded odor, and one
  punishment gives `≤ −0.1`. That keeps `describe()`'s ±0.02 thresholds reading "rewarded" and "punished".
- **Graph capture.** Every operation is fixed-shape sparse or elementwise work, in place, so AGENTS.md rule 1
  holds.

## 3. Compass: one rate unit per real neuron

- **Weights.** `W[i,j] = w_cx[i,j] · sign[type_j] · gain[type_j, type_i]`.
  - `sign` is −1 for Δ7 and +1 for the others.
  - `gain` is a small set of type-pair constants: EPG→Δ7, Δ7→all, EPG→PEN, PEN→EPG, EPG↔PEG, EPG→EPG.
  - Each row is normalised by its total synaptic input.
- **Dynamics.** `r ← r + (dt/τ)·(−r + relu(W·r + I))`, run for a fixed number of substeps per tick inside the
  captured graph.
- **Inputs:**
  - `IN_ANGVEL` excites PENs on one bridge side and inhibits the other, scaled by a gain. The connectome's
    one-wedge PEN→EPG offset moves the bump.
  - `IN_LANDMARK_*` drives EPGs by `landmark_gain · cos(θ_i − landmark_heading)`.
- **Readout.**
  - `OUT_HEADING` is the population vector of EPG activity at the EPGs' angles.
  - `bump` (16 values for the page) is EPG activity binned into 16 sectors.
  - Fan-shaped-body path integration takes the decoded heading, as now.
- **Gain search.** `python eval_connectome.py --tune-cx` grid-searches the type-pair gains, the angular-velocity
  gain, the landmark gain and the substep count, and prints the best set. Those values go into
  `fruit_fly_circuits.py` as named constants, with a comment giving the search ranges.
- **Acceptance tests** (in `fruit_fly_circuits.py`'s self-check for `wiring="hemibrain"`):
  1. **Forms:** from random activity, after 200 ticks, EPG activity has one peak 60–120° wide (full width at half
     maximum).
  2. **Holds:** in darkness with zero angular velocity, the decoded heading drifts less than 5°/s over 2,000 ticks.
  3. **Rotation gain:** for constant angular velocities of 0.02, 0.1 and 0.35 rad/tick, the decoded rotation over
     200 ticks is 0.9–1.1× the input.
  4. **Closed loop:** with the landmark on and the default gyro bias, heading error p50 is at most 0.05 rad (2×
     synthetic's 0.025).
- **Fallback.** If no searched configuration passes tests 1–3, the compass uses wedge-to-wedge weights derived
  from the same connectome inside today's 16-wedge ring:
  - group EPGs by wedge;
  - compute effective EPG→EPG weights through the Δ7 and PEN paths.
  
  The spec's success criteria then apply to that compass. README 3g reports which per-neuron test failed and by
  how much.

## 4. Integration, testing, documentation

- **Switch.** `--wiring synthetic|hemibrain` on `agent_loop.py`, `serve.py`, `eval_goals.py` and
  `eval_connectome.py`; `Sim(wiring=...)` passes it to `FlyBrain`. The default is `synthetic`, and the regression
  oracle (`{'FORAGE': 1847, 'FLEE': 25, 'ORIENT': 124, 'IDLE': 4}`, `jumps 22`) is pinned to it.
- **System 1.** `describe()` emits the same words, so the precompiled tables stay valid, and their hash rightly
  ignores wiring.
- **Page.** When wiring is `hemibrain`, one line under the System 1 line reads
  "Wiring: Janelia hemibrain v1.2 (CC BY)". The snapshot gains `"wiring": "synthetic" | "hemibrain"`.
- **Self-checks:**
  - `hemibrain.py --selfcheck` (Section 1);
  - `fruit_fly_circuits.py`: both wirings, on the CPU, and under CUDA-graph capture when CUDA exists;
  - MB: a new odor reads novel, then familiar on repeat; one reward gives valence ≥ 0.1; one punishment gives
    ≤ −0.1;
  - CX: the four acceptance tests (or the fallback's equivalents);
  - `agent_loop.py --selfcheck`: one `wiring="hemibrain"` run where the fly reaches the banana within 2,000 ticks
    and FLEE latches inside the loom.
- **`eval_connectome.py`** prints, for both wirings:
  - **odor discrimination:** mean KC-code overlap for 50 random odor pairs, and for pairs differing in one
    glomerulus;
  - **learning:** valence after one reward and one punishment, and the valence a similar odor (one glomerulus
    changed) picks up;
  - **compass:** drift in darkness, error with gyro bias and no landmark, and error with the landmark;
  - **closed loop:** behaviour counts, rewards, jumps, and tick p50/p99 on CPU and CUDA (3 runs each).
- **Docs:**
  - README 3g with those numbers and the tuned gains;
  - a README data credit;
  - `docs/DECISIONS.md`: the derived shipped file, the MBON sign rule, the per-neuron compass and its fallback;
  - `docs/VERIFY.md`: the new commands;
  - `docs/STATE.md`.

## Known limitations

- **One hemisphere.** Hemibrain covers one side, so this is one mushroom body, and the compass neurons are those
  traced there.
- **Synapse counts as weights.** Weights are synapse counts with type-level signs and gains. Real synaptic
  strengths, neurotransmitter receptors and neuromodulation are not modelled.
- **The bridge-to-ellipsoid-body angle map is published data,** not derived from this export.
- **Uniform dopamine input.** Dopamine is one world signal, split by each MBON's PAM/PPL1 fraction; the 322 DANs
  aren't simulated.
