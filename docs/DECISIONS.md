# Decisions

Append-only. Newest at the top.

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
