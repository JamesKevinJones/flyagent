# flyagent

A hybrid agent: a *Drosophila* circuit model (central complex, mushroom body, ventral nerve
cord) runs at a fixed 15 ms tick, and a typed "System 1" decision model (Laya locally, or
TypeSafe Jev hosted) picks the behaviour asynchronously.

```bash
python fruit_fly_circuits.py          # circuit self-checks, CPU + CUDA
python system1_engine.py              # state encoding + rules + HTTP client self-checks
python agent_loop.py --tick-cpus 2,3 --system1-cpus 4-7    # defaults: System 1 = rules, circuits on CPU (see 3b)
```

All numbers below were **measured on this laptop** (i5-13420H, RTX 4050 Laptop 6 GB, 16 GB
single-channel DDR5, Windows 11 native, Python 3.12, torch 2.14+cu130, driver 617.14),
except where a row says *published* or *estimate*. Native Linux and WSL2 were not measured,
because torch isn't installed in WSL here.

---

## 1. Tradeoff matrix: Laya on the RTX 4050 vs hosted Jev

Laya is a 421M-parameter model: a ModernBERT-large encoder (395M) plus a decision head.
The "synthetic" rows run a random-weight ModernBERT-large with the same shape. Latency
doesn't depend on weight values, so those rows measure Laya's GPU cost without the 808 MB
download.

| Option | Latency p50 / p99 | VRAM | Notes |
|---|---|---|---|
| **Real Laya, Laya's default load** (FP32 weights, bf16 autocast), Windows | 53.5 / 55.7 ms | **1.6 GB resident, 2.3 GB peak** | Over the 1.5 GB budget. The prompt is **278 tokens**, not 128, because question definitions are part of the input. |
| **Real Laya, bf16-resident weights** (what `laya_backend` does now) | 32–57 ms warm; 80–110 ms in other runs | **831 MB resident, 1,221 MB peak** | Same choice as FP32 on 191 of 194 states; max probability difference 0.0066. The `fast`/`compile` paths need TileLang/Triton, which aren't on Windows, so this is eager. |
| Same, first call after ≥ 5 s of GPU idle | **300–356 ms** | | The dGPU drops to P8 (210 MHz) and has to wake up. See section 3b. |
| Encoder only (random weights), FP16, eager, 128 tok | 64–84 / 74–121 ms | ~0.8 GB | Limited by kernel-launch overhead: flat from 64 to 256 tokens. |
| Encoder only, FP16, CUDA graph, 128 tok | 10.7 / 15.6 ms | 0.8 GB (peak 795 MB) | What `laya.load(..., fast=True)` does on Linux (TileLang + per-shape graphs). |
| **Encoder only, FP16, CUDA graph, 256 tok** | **22.4 / 27.3 ms** | 0.8 GB | The realistic graph-captured figure for Laya's ~278-token prompt. |
| Same, sustained 90 s | p50 11.1 → 12.7 ms, throughput 86 → 73 inf/s (−15%) | | GPU at **88–89 °C within 15 s**, drawing only 30–39 W. |
| Laya INT8 (TensorRT / ORT) | *estimate* 5–8 ms | ~0.45 GB | Not measured (no TensorRT here). Weight reads alone take 4.1 ms in FP16 at 192 GB/s vs 2.1 ms in INT8, and the measured 10.7 ms isn't at that floor yet. Re-fit calibration afterwards: INT8 shifts probabilities. |
| Laya on a T4 | 32.8–39.5 ms | | *published* (model card), stock path. |
| Jev via Cloudflare Workers AI | 70–500 ms end to end, p50 ~76–276 ms | 0 | *published*. The nearest Cloudflare edge to this laptop is in India. TCP connect took **20–68 ms**, and a fresh HTTPS request took **57–300 ms** before any inference (measured). Where the GPU behind the edge runs isn't documented. |

### What the numbers decide

- **Neither option fits inside a 15 ms tick synchronously.** Graph-captured Laya fills almost
  the whole tick on its own, and a single Jev round trip is 4–30 ticks. So System 1 runs **off the
  tick**, in its own process. Its answers are latched into the VNC when they arrive, and the
  escape reflex never waits for them. This is how the fly works too: the giant fiber
  takes milliseconds, and deliberation is much slower.
- **Self-hosted Laya is faster than Jev**: 22 ms graph-captured or 32–110 ms eager, against 70–276 ms. With bf16
  weights it fits the 1.5 GB budget, so INT8 isn't needed for memory. **But on this task neither beats
  the rule table**, which is correct on 100% of the policy checks in 10 µs. Laya, after fixing the question
  wording, flees on 99.2% of threats, but with P(FLEE) of only 0.32–0.48 (section 3b).
- **Heat is the limit, not power.** The 60–75 W TGP never comes into play: the GPU throttles at 89 °C
  while drawing about 35 W. The fix is to need less GPU time. The loop asks System 1 only when the
  worded state changes (about 9 calls/s, not 67), and caches answers, since the state space is finite.
  In the closed-loop run, System 1 p50 was 0 ms (cache hit) and p99 22 ms.
- **PCIe is irrelevant.** Each tick moves 256 bytes to the GPU and 48 bytes back, in two
  memcpys captured inside the CUDA graph. Laptop x8 links don't matter at this size.
- **Laya earns its place only once the state carries free text a rule table can't read**, such as
  operator instructions or task goals. For six categorical fields, the policy *is* a lookup table.

### VRAM budget (6 GB)

| Item | Budget | Measured |
|---|---|---|
| Laya, bf16 weights + activations | ≤ 1.5 GB | 831 MB resident, 1,221 MB peak (FP32 default: 1.6 / 2.3 GB). The cap is *enforced* with `set_per_process_memory_fraction`, applied after the bf16 conversion, so overruns become an OOM, which falls back to the next backend. |
| Fly circuits | ≤ 1.5 GB | 24.8 MB (2k KCs), 35.5 MB (50k KCs) peak |
| Contexts, display, headroom | ≥ 2.5 GB | ~4.4 GB left. Per-context overhead isn't measurable under Windows WDDM; check with `nvidia-smi` on Linux. |

The circuit budget is about 40× bigger than needed. It only matters if you load a whole
connectome (FlyWire: ~140k neurons, ~50M synapses: at most ~300 MB as FP16-value + int32-index CSR, and less once synapses are merged into weighted edges).

---

## 2. Pipeline

```
 every 15 ms tick: ONE CUDA graph (fruit_fly_circuits.py)                        async, own process
┌────────────────────────────────────────────────────────────────────────┐      (system1_engine.py)
│ pinned inp[64] ──memcpy──►                                             │
│  AL   odor[50] / mean ────────► PN                                     │
│  MB   PN ─CSR(2000×50, fan-in 6)─► KC drive ─APL top-k 5%─► KC code    │
│        ├─ novelty   = 1 − overlap(KC, familiarity trace)               │
│        ├─ valence   = MBON(KC), dopamine-gated plasticity              │
│        └─ odor id   = argmax overlap(KC, remembered tags)              │
│  CX   angvel ─P-EN shift─► EB ring (16 wedges) ─cos recurrence─► bump  │
│        ▲ landmark heading × gain (ring neurons) pulls the bump back    │   describe(): words, not numbers
│        ├─ heading   = phase of bump                                    │   ┌──────────────────────────┐
│        └─ FB memory += speed × bump ─► home vector                     │──►│ threat: approaching      │
│  VNC  P(behaviour)·motor programs ─► fwd, turn, tripod CPG             │   │ odor: banana, familiar   │
│        giant fiber: loom > θ·(1 − ½·P(jump)) ─► JUMP                   │   │ odor_memory: rewarded    │
│ ──memcpy──► pinned out[12]                                             │   │ home: behind-left, far   │
└────────────────────────────────────────────────────────────────────────┘   └────────────┬─────────────┘
        ▲            only when the worded state changes, and not cached ──────────────────┘
        │                                  ▼
        │      rules (default) | Laya (local, ≤1.5 GB) | Jev / laya-serve | any LLM key
        │      Choice: FORAGE/FLEE/ORIENT/IDLE  Score: urgency 0..3  Noul: prime jump
        └──────── latched into inp[6:12] on the next tick after arrival
```

**Turning circuit state into System 1 input** (`describe()` in `system1_engine.py`). Laya
and Jev are text models, and both document weakness at arithmetic and comparisons. So nothing
numeric crosses over:

- **Ring-attractor phase is never sent as an angle.** Heading only means something relative to a goal, so
  the bump phase and the FB home vector are combined into one of 8 egocentric words
  (`home: behind-left`) plus `near`/`far`.
- **The KC code is never sent as indices.** 100 active KC indices are meaningless tokens to a language
  model. What the code *means* is sent instead: identity (`odor: banana`, by overlap with
  remembered tags, or `unknown`), familiarity (`new`/`familiar`, from the novelty trace), and
  learned value (`rewarded`/`punished`/`neutral`, from the dopamine-trained MBON).
- **Binning makes the state change rarely.** That gates System 1 calls (about 9/s instead of 67/s),
  and it makes the state space finite, so answers are cached.

On the way back, Choice probabilities *blend* the four motor programs instead of hard-switching. A
70/30 FORAGE/ORIENT answer steers between the two. Score scales speed. Noul doesn't trigger the
jump; it *lowers the giant fiber's threshold*. In the closed-loop run, a primed fly jumped at loom 0.40
instead of 0.70, about 90 ms earlier.

---

## 3. Measured closed-loop results

`python agent_loop.py --backends synthetic,rules --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7`
(the "synthetic" backend is Laya-shaped GPU load, so the loop is measured under contention):

| Circuits on | Tick compute p50 / p99 | Tick period p50 / p99 | Overruns > 16 ms |
|---|---|---|---|
| CUDA, 2k KCs | 0.99 / 2.26 ms | 15.00 / **15.26** ms | 15 / 1999 (0.75%) |
| CPU, 2k KCs | 1.27 / 2.28 ms | 15.00 / 15.53 ms | 9 / 1999 |
| CUDA, 50k KCs | 1.25 / 5.74 ms | 15.00 / 15.67 ms | 15 / 1999 |

**p99 meets 15 ms; worst case doesn't** (max 27–39 ms, roughly once every 130 ticks). That is
Windows preempting a desktop process. Python on a general-purpose OS can't guarantee hard
real-time; for a hard guarantee, use native Linux with `isolcpus` + `SCHED_FIFO` (below), and
measure again.

### 3b. Real Laya vs the rule table

`PYTHONPATH=.deps python eval_system1.py` scores both on **all 1,938 states** `describe()` can emit.
Model: `convaiinnovations/laya-typed-decisions`. Correctness is judged only where the intended
policy is unambiguous:

| Check | Rules | Laya, behavioural wording | Laya, explicit criteria (now default) |
|---|---|---|---|
| Threat → FLEE (n=1292) | 100% | **0.0%** | 99.2% |
| Imminent → P(jump) > 0.5 (n=646) | 100% | 0.2% | 62.8% |
| No threat → P(jump) < 0.5 (n=646) | 100% | 100% | 97.5% |
| Rewarded odor, safe → FORAGE (n=204) | 100% | 70.6% | 26.0% |
| Punished odor, safe → not FORAGE (n=204) | 100% | 38.2% | 32.8% |
| Mean P(FLEE): none / approaching / imminent | 0.05 / 0.85 / 0.85 | 0.15 / 0.11 / 0.16 | 0.17 / 0.48 / **0.32** |
| Choice agreement with rules | | 16.7% | 78.8% |

- **Wording matters more than anything else.** "FLEE: run away from a threat" → 0%. "FLEE: threat
  is approaching or imminent" → 99.2%. Naming the state field works; describing the behaviour doesn't.
- **It handles one-field rules, not two-field ones.** Rules that need two fields at once (safe *and*
  rewarded → FORAGE) got *worse* with explicit criteria. It also inverts urgency: "imminent"
  gets a lower P(FLEE) than "approaching".
- **The argmax looks better than the behaviour.** The VNC blends motor programs by probability, so a
  correct argmax at P(FLEE) 0.32 still means a fly running away at about a third of rules' commitment.

**Closed loop** (`agent_loop.py --backends laya,rules`, 2,000 ticks):

| | Rules | Laya, circuits on CUDA | Laya, circuits on CPU |
|---|---|---|---|
| Ticks FORAGE / FLEE / ORIENT / IDLE | 1845 / 25 / 129 / 1 | 12 / 4 / 1981 / 3 | 11 / 6 / 1981 / 2 |
| Tick period p99, overruns | 15.26 ms, 15 | 18.05 ms, 33 | **15.000 ms, 6** |
| System 1 answer delay p99 / max | n/a | 45 ms / not recorded | 60 ms / **300 ms** |

- **The fly orients almost the whole time.** In the foraging state, ORIENT edges out FORAGE 0.31 to 0.29, and
  that near-tie decides 99% of the run.
- **The cache makes threat responses slow.** Most decisions are cache hits, so the GPU idles, and
  after 5 s idle it sits at P8 (210 MHz). A new state, such as the predator appearing, then waits about 300 ms
  for the GPU to wake: FLEE latched 20 ticks after the threat. Measured in isolation, the first call is
  306–356 ms after a 5–15 s idle and 33 ms after a 0.5 s idle.
- **Eager Laya and the circuit graph contend for the GPU.** Two contexts time-slice on Windows,
  so with eager Laya, run the circuits on the CPU: 6 overruns instead of 33.

**Verdict (adopted: `--backends` now defaults to `rules`).** Keep `rules` as the primary System 1 for this categorical state. Fine-tuning Laya on
(state → behaviour) pairs would mean distilling a 1,938-entry table into a 421M-parameter model. If
Laya stays, (a) precompute its answers for every state at startup (60–85 s batched; save to disk),
so runtime is a lookup with no GPU wake-ups, or (b) keep the GPU warm with a small kernel every
~0.5 s. Option (a) only works while the state stays finite.

### 3c. Landmark correction of the compass

The ring takes a visual landmark: inputs `IN_LANDMARK_HEADING` and `IN_LANDMARK_GAIN`, standing in for
ring neurons. The cue is a cosine bump at the heading the landmark implies, added before the recurrence.
Each tick it pulls the phase about gain/(1+gain) of the way toward the landmark, so a gyro bias *b* leaves
a steady error of about *b*·(1+gain)/gain instead of growing without limit. The simulated world now has a
biased gyro (`--gyro-bias`, default 0.0005 rad/tick ≈ 2°/s) and a landmark (`--landmark-gain`, default 0.02):

| 2,000 ticks, gyro bias 0.0005 rad/tick | Heading error p50 / max | FORAGE / FLEE / ORIENT / IDLE ticks |
|---|---|---|
| Landmark, gain 0.02 | 0.025 / 0.025 rad (theory 0.0255) | 1847 / 25 / 124 / 4 |
| No landmark | 0.500 / 1.000 rad | 1144 / 25 / 228 / **603** |

Without the landmark, path integration runs on the drifting heading. The home vector goes wrong, and the
fly idles 30% of the run because it thinks it is home. Raise the gain for a reliable landmark (faster
correction, smaller error); lower it for a noisy one (noise passes through in proportion to the gain).

### 3d. Bring your own LLM key (any provider)

The `llm` backend talks to any OpenAI-compatible chat API, so any provider's key works. It's
configured only through environment variables; keys never go in files or flags:

| Provider | Set | Tested here |
|---|---|---|
| **Gemini** | `GEMINI_API_KEY` (base URL and model default to Gemini's OpenAI endpoint and `gemini-3.8-flash`; `LLM_MODEL` overrides) | Self-check only (no key) |
| OpenAI | `LLM_BASE_URL=https://api.openai.com/v1`, `LLM_API_KEY`, `LLM_MODEL` | No |
| Anthropic | `LLM_BASE_URL=https://api.anthropic.com/v1`, `LLM_API_KEY`, `LLM_MODEL` | No |
| Groq / OpenRouter | `LLM_BASE_URL=https://api.groq.com/openai/v1` or `https://openrouter.ai/api/v1`, `LLM_API_KEY`, `LLM_MODEL` | No |
| **Ollama (local, no key)** | `LLM_BASE_URL=http://127.0.0.1:11434/v1`, `LLM_MODEL=<model>` | **Yes** |

```bash
export GEMINI_API_KEY=...                  # PowerShell: $env:GEMINI_API_KEY = "..."
python agent_loop.py --backends llm,rules
python eval_system1.py llm --n 120         # score it against the rule table first
```

`LLM_TIMEOUT` (default 5 s) caps each call. On a timeout, an error, or a reply that isn't the expected
JSON, that decision falls back to `rules`. The reply is treated as untrusted input: probabilities are
clamped and renormalised, and anything else raises.

**Measured with Ollama `qwen3:4b-instruct-2507-q4_K_M`** (120 random states):

| Check | Rules | Laya | Qwen3-4B (LLM) |
|---|---|---|---|
| Threat → FLEE | 100% | 99.2% | 95.3% |
| Imminent → P(jump) > 0.5 | 100% | 62.8% | 100% |
| Rewarded odor, safe → FORAGE | 100% | 26.0% | 83.3% |
| Punished odor, safe → not FORAGE | 100% | 32.8% | 100% |
| Mean P(FLEE) when imminent | 0.85 | 0.32 | 0.94 |
| Time per decision | ~10 µs | 37–110 ms | **~1.7 s** (3–5 s in the loop) |

A general LLM reads the field-named criteria far better than Laya, but in the closed loop it's too
slow to steer the fly. Only 40 decisions arrived in 30 s, and the 450 ms predator loom ended before
any FLEE answer did. The giant-fiber reflex handled the escape (16 jumps), and tick timing held
(p99 16.1 ms). For this state space, `rules` stays the default. An LLM backend fits slower
deliberation: route choice, or reading free-text instructions.

---

## 4. Tuning guide for the Nitro V 15

### CPU placement (i5-13420H: 4 P-cores = logical 0–7, 4 E-cores = 8–11; Intel hybrid CPUs list P-core threads first)

| Thread | CPUs | Why |
|---|---|---|
| Tick loop | `--tick-cpus 2,3` (one P-core, both hyperthreads) | Avoid CPU 0, where most interrupt handling lands. Leave the sibling hyperthread idle, or the two contend for the same core. |
| System 1 process | `--system1-cpus 4-7` | Laya's tokenisation and decoding are pure Python; a separate *process* keeps them off the tick's GIL. |
| Logging, network, disk | E-cores 8–11 | `taskset -c 8-11 <cmd>` on Linux, or `psutil.Process(pid).cpu_affinity([8,9,10,11])`. |

- **Native Linux:** `p_cores()` reads `/sys/devices/cpu_core/cpus` automatically. For hard timing,
  boot with `isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3` and run with `CAP_SYS_NICE`
  (`sudo setcap cap_sys_nice+ep $(readlink -f $(which python))`), so `SCHED_FIFO` succeeds.
- **WSL2 can't pin to P-cores.** It exposes 6 untyped vCPUs, which Hyper-V moves between P- and
  E-cores, and `taskset` inside WSL only picks vCPUs. If the 15 ms tick matters, use native
  Linux or Windows native.
- **Windows native** (what was measured): pass `--tick-cpus`. The loop raises itself to
  `HIGH_PRIORITY_CLASS`, which needs no admin. Set the power plan to *Best performance*.

### Timers

- **Don't use `asyncio` for a 15 ms deadline on Windows.** Measured: `asyncio.sleep(13.5 ms)` overshoots by
  up to 13 ms at p99, even with `timeBeginPeriod(1)`. `time.sleep` (a high-resolution waitable
  timer since Python 3.11) overshoots by 2.7 ms at p99. The loop sleeps until 1.5 ms before the
  deadline, then spin-waits. Swapping asyncio for this took overruns from 492 to 35 per 1,500 ticks.
  System 1 is still asynchronous, as a process-pool future.
- `gc.freeze()` after setup, and `torch.set_num_threads(1)` in the tick process.

### GPU

- **Stop the GPU idling between ticks.** A paced CUDA tick costs 2.2 ms against 0.2 ms
  back-to-back, because the GPU drops into a low-power state during the 13 ms gap. With System 1
  keeping it busy, the tick fell to 0.99 ms. Two fixes: on Windows, set NVIDIA Control Panel → Power
  management mode → *Prefer maximum performance* for python.exe. On Linux, lock clocks with
  `sudo nvidia-smi -lgc 1500,1500` (some GeForce laptop SKUs refuse this), a fixed clock below the throttle point, so latency stays
  stable instead of swinging between 1350 and 2130 MHz.
- **The dGPU sleeps when System 1 is idle.** After about 5 s with no work it drops to P8, and the next Laya call
  costs 300+ ms instead of about 50 ms. *Prefer maximum performance* (above) also prevents this.
- **Laptop cooling:** the GPU already sat at 72–79 °C *idle*. Raise the rear of the laptop, and use
  NitroSense's fan max mode for long runs. Expect sustained throughput about 15% below the
  first-15-second numbers.

### PyTorch

- **CUDA graphs:** the whole circuit tick, including both pinned-memory copies, is one
  `torch.cuda.CUDAGraph`. Replay plus sync takes 0.2 ms back-to-back. The mushroom-body step alone
  measured 0.05 ms graphed vs 0.55 ms eager, and its p99 went from 10 ms to 2.5 ms. `torch.sparse_csr`
  SpMM and `topk` capture fine on torch 2.14.
- **`torch.compile(mode="reduce-overhead")`** is CUDA graphs plus Triton kernel fusion. Triton
  isn't installed in the Windows torch build here, so the code captures graphs directly, which works
  everywhere. On Linux, `laya.load(..., compile=True)` / `fast=True` handle Laya.
  For the circuits, fusing about 60 tiny kernels would save microseconds inside a 15 ms tick, so it isn't worth it.
- **Pinned memory:** `inp`/`out` are pinned, and the copies are captured in the graph, so each tick is
  one launch and one event sync. True zero-copy (the GPU reading mapped host memory directly) isn't
  exposed by PyTorch, and with about 300 bytes per tick it would save nothing.
- **Circuits on CPU or GPU?** At 2k KCs, put them on the CPU (now the default; `--device cuda` to override). Against graph-captured load the two
  tie (period p99 15.5 vs 15.3 ms), but against *eager* Laya the GPU tick suffers from context
  time-slicing (18.05 ms vs 15.000 ms p99). The GPU pulls ahead from about 50k KCs, or with a full connectome.

---

## Not done, and what it takes

- **Laya is installed project-locally** (`pip install --no-deps --target .deps laya==0.3.27`;
  weights in the Hugging Face cache). Run anything that loads it with `PYTHONPATH=.deps`.
- **Laya on Linux with `fast=True`** (TileLang + CUDA graphs) hasn't been measured; that would show
  whether the 22 ms graph-captured figure holds for the real model.
- **Jev live:** needs a Cloudflare account and API token.
  `JEV_URL=https://api.cloudflare.com/client/v4/accounts/<id>/ai/run`, `JEV_API_KEY=<token>`.
- **Native Linux / WSL2 timings** haven't been measured.
