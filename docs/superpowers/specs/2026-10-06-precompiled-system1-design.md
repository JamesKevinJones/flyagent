# Precompiled System 1 tables: design

**Date:** 2026-10-06 · **Status:** approved in conversation, awaiting written-spec review
**Sub-project 2 of 3.** Order: (1) free-text goals (merged) → (2) fast System 1 on novel states → (3) full connectome.

## Purpose and success

Portfolio piece, latency engineering. Today a model System 1 answers a state it hasn't seen in about 300 ms
(Laya, GPU waking from P8) or 1.7–5 s (local Qwen3-4B through `llm`), and neither latched FLEE inside the
450 ms loom (README 3b, 3d). The worded state space is finite: 1,938 base states. So every model answer can
be compiled ahead of time and looked up in microseconds. Success means:

1. With a complete table, a model backend decides in about the time rules take (about 10 µs, measured), and
   the README shows before/after numbers from this machine.
2. Table answers equal live answers: 100% for Laya on a 100-state sample; for the LLM, whatever we measure.
3. With the table, a predator launch latches FLEE as fast as with rules.
4. The tick never waits on a model, including while a table is still filling.
5. Nothing regresses. `--backends rules` stays the default and behaves exactly as today, and every existing
   self-check passes.

## Decisions (from the brainstorm)

| Decision | Chosen | Rejected because |
|---|---|---|
| Goal of sub-project 2 | Latency engineering: make model backends usable live | Instant-then-model hybrid keeps a slow path; fine-tuning is a different project |
| Which backends | Any backend in the chain (`laya`, `llm`, `http`, `synthetic`) through the existing `decide(state)` interface | Hard-wiring Laya is no less code and leaves the LLM slow |
| How the table fills | In the background from the worker process, saved as it goes; misses answered by rules | A compile-only step is a chore to remember; it comes free as "fill until done" |
| Fill order | Threat states first, then the rest | Nearest-states-first only helps the 16-minute LLM fill, and threat-first already covers the loom case |
| Lookup location | In the sim process, a dict | Over IPC the lookup would cost about a millisecond, not microseconds |

## Scope

**In:**
- table key, file, loader and validation
- the background filler and the `--compile` CLI
- `Sim` lookup with goal precedence
- snapshot, page status line and CLI stats
- `serve.py --backends`
- self-checks
- `eval_system1.py --latency` and README section 3f
- committed Laya and Qwen3-4B tables

**Out (YAGNI):**
- compiling Laya with `predict_batch` (only if the per-state fill takes longer than about 3 minutes)
- goal-aware model questions
- nearest-states fill order
- table sharing or download
- per-goal tables (the models don't read goal fields)

## 1. Architecture

```
sim process (15 ms tick)                          worker process (existing ProcessPoolExecutor(1))
  step(): state changed?                            compile_chunk(states) -> [(state, Decision) | failed]
    goal precedence -> rules, or                      runs the chain's first loaded model backend
    table[base(state)] hit -> model answer
    miss -> rules answer        (all via _Done, applied next tick, like rules today)
  step(): filler future done? -> merge into dict, append to file, submit next chunk (never blocks)
```

- **Base state:** the 6 non-goal fields of `describe()` (`threat`, `odor`, `odor_familiarity`,
  `odor_memory`, `home`, `moving`). Models never read the goal fields, so one table serves every goal.
- **Live path:** no IPC, no model call. The decision cost is a dict lookup, or the rules function on a miss.
- **Filler:** `compile_chunk` takes about 32 states, runs the first model backend in the chain that loaded,
  and returns an answer or a failure per state. Only the filler ever calls the model, so backends need no
  thread safety.
- **Failures:** a state that fails stays missing (rules keep covering it). The filler never stores fallback
  answers, the same rule `worker_decide` follows today. After 5 consecutive chunks with no successful answer,
  `Sim` stops submitting and reports `status: "stalled"`.
- **No model loaded:** if every model backend in the chain fails to load, `status` is `"off"`, and rules
  decide.
- **Complete table:** if the loaded file already holds all 1,938 states, `Sim` starts no worker process. No
  model loads, no VRAM is used, and startup is instant.
- **`--backends rules`:** unchanged. No pool, no table, `status: "off"`.
- **Removed:** the live model path. `worker_decide`, its `_cache` and the per-state chain call go;
  `worker_init` stays as the filler's initializer. `eval_system1.py` calls backends directly and is
  unaffected.

### Goal precedence

The models were never asked about goals, so for a non-threat state under a goal only rules know what to do.
The order, per worded-state change:

1. `threat != "none"`: the table (the model's own FLEE, compiled first), else rules.
2. Any goal field set (`goal_seek`, `goal_avoid`, `goal_heading` not `"none"`, or `goal_rest == "yes"`):
   rules.
3. Otherwise: the table, else rules.

This makes the page behave the same under every backend, with the model showing through on threats and on the
default goal.

## 2. The table file

- **Path:** `tables/<backend>-<slug>-<hash8>.jsonl`, where:
  - `slug` is a filename-safe form of `LAYA_MODEL` or `LLM_MODEL`;
  - `slug` is `jev` for `http` (never the URL, which can carry an account id);
  - `hash8` is the first 8 hex digits of a sha256 over the backend name, the full model id (`LAYA_MODEL`,
    `LLM_BASE_URL` + `LLM_MODEL`, or `JEV_URL`) and the prompt that backend sends (`QUESTIONS`, or
    `LLM_PROMPT` for `llm`).
  
  Changing a question, the prompt or the model therefore starts a fresh table and never mixes two policies.
  Each backend exposes this identity as `fn.table_id = (backend, slug, hash8)`.
- **Format:** append-only JSON Lines, one line per state, flushed after each chunk:
  ```json
  {"state": {"threat": "none", "odor": "banana", ...}, "probs": [0.8, 0.05, 0.1, 0.05], "urgency": 0.33, "p_jump": 0.05, "ms": 41.2}
  ```
  `ms` keeps the original model latency, so the README can quote it next to the lookup time.
- **Loader:** the file is editable, so it is input. A line is skipped, and the skips counted and printed once,
  if any of these hold:
  - it doesn't parse;
  - its `state` keys aren't exactly the base fields;
  - a value is outside `describe()`'s vocabulary;
  - `probs` is not 4 finite numbers in 0..1 summing to 1 within 1e-3;
  - `urgency` or `p_jump` is outside 0..1.
  
  For duplicates, the last line wins. A truncated last line (crash mid-write) is just a skipped line.
- **Enumerating states:** `all_states()` moves from `eval_system1.py` into `system1_engine.py`. `fill_order()`
  yields `all_states()` with the 1,292 approaching/imminent states first.
- **Git:** `tables/` is gitignored. The two measured tables (Laya, Qwen3-4B) are force-added (`git add -f`)
  so a clone gets Laya's real policy on a CPU-only laptop. Hosted-key tables are never committed by accident,
  because nothing under `tables/` is tracked unless added by hand.

## 3. What you see

- **Snapshot keys:**
  - `table`: `{"backend": "laya", "filled": 1212, "total": 1938, "status": "filling"}`, where `status` is one of
    `filling`, `complete`, `stalled` or `off`;
  - `decided_by`: `"laya table"` or `"rules"`, for the decision currently applied.
- **Page:** one status line under the behaviour bars: "System 1: laya table 1,212 / 1,938, rules cover the
  rest", or "System 1: laya table complete", or "System 1: rules". Read the design-engineering skill before
  touching `web/index.html`.
- **`serve.py --backends`:** default `rules`, passed through to `Sim`.
- **`agent_loop.py` stats:** table hits and misses, lookup p50 in µs, and the table line.
- **`python system1_engine.py --compile <backends>`:** runs the filler in-process until the table is complete
  or stalled, printing progress, threat-coverage time and total time. This builds the committed tables.

## 4. Testing and measurement

**Self-checks (assert-based `__main__`, project rule 5):**
- **`system1_engine.py`:**
  - `all_states()` yields 1,938 states whose keys equal `describe()`'s base fields.
  - `fill_order()` puts all 1,292 threat states first.
  - The loader, given a temp file with one good line, a bad-probs line, an unknown-word line, a duplicate
    and a truncated last line, keeps exactly the valid answers, with the last duplicate winning.
  - `compile_chunk` with an in-process stub backend that fails on some states leaves those missing and
    stores no rules answer.
  - The table hash changes when `QUESTIONS` changes.
- **`agent_loop.py --selfcheck`, using a temp `tables_dir`:**
  - A complete table means no pool, and decisions come from the table.
  - A partial table means a miss reports `decided_by == "rules"`.
  - A rest goal gives IDLE from rules even where the table has an answer.
  - A threat state under a rest goal is still answered by the table.

  Pool wiring is covered by the real Laya run below, not a unit test, since a stub can't cross a Windows
  spawn.

**Measurements (README section 3f, this machine, Laya then local Qwen3-4B):**
1. **Compile cost:** time to cover the threat states, and to fill all 1,938.
2. **Decision latency:** live model p50/p99 before, table lookup p50/p99 after, plus tick p50/p99 while the
   filler runs, to show it doesn't disturb the tick.
3. **Loom latch:** ticks from loom onset to FLEE latched, over 20 predator launches, for live model (old
   numbers), table and rules.
4. **Same answers:** 100 sampled states asked live against the table.

The code for these is `eval_system1.py --latency`; no new script.

**Docs:**
- `docs/DECISIONS.md`: in-process lookup, goal precedence, committed tables.
- `docs/VERIFY.md`: new checks.
- `docs/STATE.md`: updated at the end.

## Known limitations

- Under a goal, non-threat decisions are always rules, so a model's policy shows only on threats and the
  default goal.
- A table is only as current as its hash inputs. A change to `describe()`'s vocabulary needs a manual delete
  of the old tables (the loader would skip every line anyway).
- LLM answers at temperature 0 may still vary run to run; the table freezes one sample.
