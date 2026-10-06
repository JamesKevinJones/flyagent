# Decisions

Append-only. Newest at the top.

---

## 2026-10-06: Real wiring from the hemibrain connectome, behind `--wiring hemibrain`

**Context.** Sub-project 3 (spec: docs/superpowers/specs/2026-10-06-hemibrain-wiring-design.md). The goal was to
replace the hand-built mushroom body and compass with measured wiring, then measure what changes.
**Decision.**
- **Data.** `hemibrain.py` compiles Janelia hemibrain v1.2 (CC BY 4.0, a public 45.9 MB archive) once into
  `data/hemibrain_mb_cx.npz`, which is 102 KB and committed with `data/DATA_LICENSE`. Runtime reads only that file.
- **Mushroom body.** 63 real glomerulus inputs, 1,927 real KCs, and the real KC→MBON synapses onto 68 MBONs.
  - Each MBON's valence sign is `(PPL1 − PAM)/(PPL1 + PAM)` of its measured dopamine input (Aso et al. 2014:
    reward through PAM depresses avoidance MBONs).
  - On the right, fully traced side, this agrees with the literature for MBON01–03 (avoidance) and MBON11 and 14
    (approach).
- **Compass map.** The bridge-to-compass angle map was chosen from the data: of the candidate maps, it's the one
  under which the real PEN→EPG wiring shifts the bump by one wedge in opposite directions per side.
- **Compass model.** One rate unit per real neuron forms a bump and holds it, but fails the rotation-gain test: the
  bump moves at most about 60°/s. So the agent uses the spec's fallback, today's ring with its kernel derived from
  the same wiring. That ring misses rotation gain only at the slowest speed (0.88 vs 0.9).
**Why not FlyWire, or a whole-brain spiking model.** FlyWire is CC BY-NC. A whole-brain model's behaviour claim
would rest on guessing which neurons to read out.
**Consequences.**
- `--wiring synthetic` stays the default, and the regression oracle is unchanged.
- System 1 tables stay valid, because `describe()` emits the same words.
- Under hemibrain wiring the compass pins to wedges, so slow turns under-rotate about 12% and a tiny gyro bias never
  accumulates (README 3g).

---

## 2026-10-06: Model System 1s are precompiled into tables; the tick only looks them up

**Context.** A model answering a new state live took about 300 ms (Laya, GPU waking from P8) to 5 s (local
Qwen3-4B), and neither latched FLEE inside the 450 ms loom. `describe()` can only emit 1,938 base states
(spec: docs/superpowers/specs/2026-10-06-precompiled-system1-design.md).
**Decision.**
- The first model named in `--backends` owns a table, `tables/<backend>-<slug>-<hash8>.jsonl`. It's filled
  in the background by the old worker process, threat states first, and saved as it goes.
- `Sim` looks each state up in a dict in its own process; a miss is answered by rules.
- A complete table starts no worker and loads no model.
- Under any goal, non-threat states are decided by rules, because the models were never asked about goals.
- The measured Laya and Qwen3-4B tables are committed (`git add -f`); `tables/` is otherwise gitignored.
**Why not keep the model live with a cache.** The cache only helps on a second visit, and the first visit
is the one the predator punishes. Looking up over IPC would also cost about a millisecond, not microseconds.
**Why one table per run, not per loaded backend.** The file has to be known before the model loads, or a
complete table couldn't skip the model.
**Consequences.**
- The "System 1 runs in its own process" entry below now describes the filler only.
- A model's policy shows only on threats and with no goal set.
- A change to `describe()`'s vocabulary needs the old tables deleted by hand.

---

## 2026-10-05: One generic `llm` backend for every provider key

**Context.** The repo goes public, and anyone cloning it should be able to plug in their own key
(Gemini, OpenAI, Anthropic, Groq, OpenRouter, local Ollama) without installing Laya or opening a Cloudflare account.
**Decision.** One backend over the OpenAI-compatible `/chat/completions` API, using the stdlib `http.client`
and the same persistent-connection helper as the Jev client. It's configured only through env vars, with a
`GEMINI_API_KEY`-only shortcut.
**Why not one SDK per provider.** That means five dependencies and five code paths for one JSON POST. Every
listed provider exposes the OpenAI-compatible endpoint.
**Consequences.** Provider-specific features (native structured-output schemas, thinking controls) aren't
used. The prompt is generated from `QUESTIONS`, so Laya and the LLM always see the same criteria.

---

## 2026-10-05: Goals are interpreted once and executed by rules and circuit buffers

**Context.** Free-text goals for a portfolio demo (spec: docs/superpowers/specs/2026-10-05-free-text-goals-design.md).
**Decision.**
- A built-in parser handles the world's vocabulary, with an optional LLM for leftover words.
- The goal is compiled into odor signs, a heading drive and a home sign.
- Goal-aware rules decide every tick.
- A stdlib server streams over SSE to one static page, local only.
**Why not an LLM per decision.** It's measured at 3–5 s per decision with a local model, and Laya can't
read multi-field rules. A goal changes rarely, so interpreting it once costs nothing at tick time.
**Why not a public link.** A JavaScript port would be a second, unmeasured simulator. A hosted server
costs money and exposes an LLM key.
**Consequences.**
- Goals are limited to the vocabulary.
- The LLM's answer replaces the parser's, and with a 4B model that hurts paraphrases (README 3e).

---

## 2026-10-05: Circuits default to the CPU

**Context.** With `rules` as System 1 nothing keeps the GPU busy, and the idle GPU wakes up on every
15 ms tick. Back-to-back runs at 2k KCs: CPU had 5 and 7 overruns per 2,000 ticks (p99 15.000 ms);
CUDA had 12 and 27 (p99 up to 17.7 ms).
**Decision.** `agent_loop.py --device` defaults to `cpu`.
**Why not CUDA.** At this size the CUDA graph saves microseconds of compute and costs milliseconds of wake-up.
**Consequences.** Use `--device cuda` for about 50k KCs or more, a full connectome, or whenever Laya keeps the GPU warm anyway.

---

## 2026-10-05: System 1 defaults to the rule table, not Laya

**Context.** README section 3b: over all 1,938 states, rules pass 100% of policy checks in about 10 µs.
Laya (bf16, explicit criteria) gets 99.2% threat→FLEE but at P(FLEE) 0.32–0.48, 26% on
safe+rewarded→FORAGE, and in the closed loop it orients for 99% of ticks, with a 300 ms GPU-wake stall
on new states.
**Decision.** `agent_loop.py --backends` defaults to `rules`. `laya`, `http` (Jev) and `synthetic`
stay available as opt-in chains.
**Why not Laya first, rules as fallback.** The fallback only fires on errors, not on wrong answers,
so Laya-first means Laya's worse decisions drive the fly.
**Consequences.** Revisit when the state gains free text (goals, operator instructions) that a
table can't read. That is where Laya can win.

---

## 2026-10-05: Laya questions name state fields; Laya weights stored in bf16

**Context.** Measured over all 1,938 states, behavioural criteria ("FLEE: run away from a threat")
gave FLEE on 0% of threatened states. Laya's default load uses 1.6 GB of VRAM (FP32 weights), over the
1.5 GB budget.
**Decision.** Criteria name the fields they depend on ("FLEE: threat is approaching or imminent"):
99.2% FLEE. `laya_backend` casts the model to bf16 when it runs eager: 831 MB resident, 37 ms vs
54 ms, same choice on 191 of 194 states.
**Why not keep the descriptive wording.** It reads nicer and fails the safety check outright.
**Consequences.** The criteria now restate the rule table, so Laya is checked against the rules'
own logic. Two-field rules still fail (safe and rewarded → FORAGE: 26%).

---

## 2026-10-05: The tick is a deadline loop, not asyncio

**Context.** The spec asked for an asyncio event loop with a 15 ms tick or better.
**Decision.** `time.sleep` until 1.5 ms before the deadline, then spin. System 1 stays asynchronous
as a `ProcessPoolExecutor` future.
**Why not asyncio.** Measured on this laptop: `asyncio.sleep(13.5 ms)` overshoots by up to 13 ms at
p99, against 2.7 ms for `time.sleep`. With asyncio, 492 of 1,499 ticks overran; after the switch, 35 did.
**Consequences.** If asyncio I/O is ever needed, run it in another thread or process, not around the tick.

## 2026-10-05: System 1 runs off the tick, in its own process, and only on state change

**Context.** Graph-captured Laya takes 10.7 ms p50 at 128 tokens, and Jev 70–500 ms.
Neither fits inside a 15 ms tick alongside anything else.
**Decision.** Answers are latched when they arrive. Calls happen only when `describe()`'s words change,
and answers are cached per state (deterministic models, finite state space). The giant-fiber
reflex lives in the circuits; Noul only lowers its threshold.
**Why not a thread.** Laya's tokenisation is pure Python and would take the GIL from the tick.
**Consequences.** Behaviour timing varies between runs, because it depends on System-1 latency.

## 2026-10-05: Small circuits are dense; only PN->KC is sparse (CSR)

**Context.** The spec asked for sparse tensors.
**Decision.** The 16x16 ring kernel stays dense. PN->KC (fan-in 6 of 50) is `torch.sparse_csr`,
which captures into CUDA graphs with `topk` on torch 2.14.
**Why not sparse everywhere.** A sparse 16x16 matrix is slower and harder to read than a dense one.
**Consequences.** A full connectome would go in as another CSR matrix and inherit the same graph capture.
