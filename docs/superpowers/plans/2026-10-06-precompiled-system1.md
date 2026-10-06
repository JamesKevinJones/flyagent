# Precompiled System 1 Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make any model System 1 backend decide live in microseconds by compiling its answers for all 1,938 worded states into a saved table, filled in the background with rules covering the gaps.

**Architecture:**
- **Live path:** `Sim` holds the table as a dict in the sim process. Each worded-state change is a goal-precedence check, then a lookup, with rules covering a miss. There is no IPC and no model call on the tick.
- **Filler:** the existing single-worker `ProcessPoolExecutor` runs `compile_chunk` over threat states first, and `Sim` merges finished chunks without blocking and appends them to `tables/<backend>-<slug>-<hash8>.jsonl`.
- **Complete table:** if the table is already complete, no worker starts and no model loads.

**Tech Stack:** Python 3.12, torch 2.14 (existing), stdlib only (`hashlib`, `json`, `pathlib`, `re`, `tempfile`). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-precompiled-system1-design.md`

## Global Constraints

- **No new dependencies.** The page stays one static file.
- **Unchanged default:** `--backends rules` keeps exactly today's behaviour. Regression oracle:
  `"$PY" agent_loop.py --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7` must print `{'FORAGE': 1847, 'FLEE': 25, 'ORIENT': 124, 'IDLE': 4}`, `jumps 22`, `rewards ['banana', 'geosmin']`.
- **The tick never waits on a model.** Nothing in `Sim.step()` calls `.result()` on a future that is not `done()`.
- **Values:**
  - 1,938 base states, of which 1,292 are threat states (approaching/imminent) and come first;
  - chunk size 32;
  - the filler is stalled after 5 consecutive chunks with no successful answer;
  - probabilities must sum to 1 within 1e-3.
- **Base fields, in this order:** `threat, odor, odor_familiarity, odor_memory, home, moving`.
- **Goal precedence:**
  1. A threat state: the table, else rules.
  2. Any goal field set: rules.
  3. Otherwise: the table, else rules.
- **Keys:** keys stay in env only. The `http` backend's slug is `jev`; a URL never appears in a filename.
- **Tests:** one assert-based `__main__` self-check per module, no framework.
  - Project env: `PY=.venv/Scripts/python.exe` (CPU torch, no Laya).
  - CUDA env: `CPY="C:/Users/kj638/Kevin codes/ComfyUI/.venv/Scripts/python.exe"`, with `PYTHONPATH=.deps` for Laya.
- **Commits:** plain messages. No Claude co-author or attribution lines.
- **Text:** never use the section-sign character anywhere.

## Spec clarification (decided here)

**One table per run.** The table belongs to the first non-`rules` name in `--backends`. The spec's "chain's first loaded model backend" would leave the file unknown until the model loads, which defeats "complete table → no worker". If that backend fails to load, the status is `off` and rules decide; any later model names in the chain are ignored. The spec's `fn.table_id` attribute therefore becomes a module function, `table_path(name)`, which needs no model load.

## Review Focus

1. **A CPU-only clone runs `--backends laya` against a committed complete table.** It must never import `laya`, and it decides from the table. Test: Task 3, `complete_table_no_pool`, run in `.venv` (no Laya).
2. **The model backend can't load** (no CUDA, no key). The status becomes `off` within a few seconds, rules decide and ticks keep coming. Test: Task 3, `unavailable_model_is_off`.
3. **A model returns a bad answer** (NaN probs from an LLM). It must never enter the table or the file. Test: Task 2, `bad_answer_not_stored`.
4. **The server is stopped mid-fill and started again.** The file is reloaded and only missing states are asked. Test: Task 2, `resume_fills_only_missing`.
5. **`describe()` emits a state the table doesn't know** (a future vocabulary word). The lookup misses and rules answer, with no exception. Test: Task 1, `unknown_state_uses_rules`.

---

### Task 1: State vocabulary, table file and lookup (`system1_engine.py`)

**Files:**
- Modify: `system1_engine.py` (new section after `describe()`, plus the `__main__` self-check)
- Modify: `eval_system1.py` (import `THREATS`, `ODORS`, `HOMES`, `all_states` from `system1_engine`; delete its local copies)
- Modify: `.gitignore` (add `tables/`)

**Interfaces:**
- Produces:
  - `BASE_KEYS: tuple[str, ...]`, the 6 base fields in the order above.
  - `THREATS`, `ODORS`, `HOMES`, moved verbatim from `eval_system1.py`.
  - `VOCAB: dict[str, set[str]]`, the allowed values per base field.
  - `N_STATES = 1938`.
  - `all_states() -> Iterator[dict]`, as today, and `fill_order() -> list[dict]`, threat states first, otherwise in `all_states()` order.
  - `base_key(state: dict) -> tuple`, the values of `BASE_KEYS` in order.
  - `table_path(name: str, tables_dir="tables") -> pathlib.Path | None`:
    - returns None for `rules`, and for `llm` when `llm_config()` is None;
    - otherwise returns `Path(tables_dir) / f"{name}-{slug}-{hash8}.jsonl"`;
    - the slug and hash inputs are in the table below.
  - `valid_answer(state: dict, probs, urgency, p_jump) -> bool`.
  - `load_table(path, name: str) -> dict[tuple, Decision]`. A missing file returns `{}`. Entries get `backend=f"{name} table"` and keep the file's `ms`.
  - `append_table(path, answers: list[tuple[dict, Decision]]) -> None`. Creates the parent directory; writes one JSON line per answer, as `{"state": {base fields}, "probs": [...], "urgency": u, "p_jump": p, "ms": m}`; flushes once per call.
  - `lookup(state: dict, table: dict, rules) -> Decision`, which applies the goal precedence in Global Constraints. `state` is a full `describe()` dict; `rules` is a `rules_backend()` function.

| name | model id (hashed in full) | slug | prompt hashed |
|---|---|---|---|
| `laya` | `LAYA_MODEL` or `convaiinnovations/laya-typed-decisions` | model id | `QUESTIONS` |
| `llm` | `LLM_BASE_URL` + `LLM_MODEL` from `llm_config()` | `LLM_MODEL` | `LLM_PROMPT` |
| `http` | `JEV_URL` or `""` | `jev` | `QUESTIONS` |
| `synthetic` | `synthetic` | `synthetic` | `QUESTIONS` |

- slug: `re.sub(r"[^A-Za-z0-9._-]+", "-", text)`.
- hash8: `sha256("\n".join([name, model_id, json.dumps(prompt, sort_keys=True)]))`, first 8 hex digits.

- [ ] **Step 1: Write the failing self-check additions** in `system1_engine.py`'s `__main__`, using a `tempfile.TemporaryDirectory()`:

```python
states = list(all_states())
assert len(states) == N_STATES == 1938 and all(tuple(s) == BASE_KEYS for s in states)
assert tuple(k for k in describe(out, 0.0, {0: "banana"}) if not k.startswith("goal_")) == BASE_KEYS
order = fill_order()
assert len(order) == 1938 and all(s["threat"] != "none" for s in order[:1292]) \
    and all(s["threat"] == "none" for s in order[1292:])
# load_table_skips_bad_lines: good, bad probs, unknown word, duplicate (last wins), truncated tail
good = states[0]
p = Path(tmp) / "t.jsonl"
append_table(p, [(good, Decision((0.7, 0.1, 0.1, 0.1), 0.3, 0.1, "laya", 40.0))])
with open(p, "a", encoding="utf-8") as f:
    f.write(json.dumps({"state": states[1], "probs": [0.9, 0.9, 0, 0], "urgency": 0, "p_jump": 0, "ms": 1}) + "\n")
    f.write(json.dumps({"state": {**states[2], "odor": "durian"}, "probs": [1, 0, 0, 0], "urgency": 0, "p_jump": 0, "ms": 1}) + "\n")
    f.write(json.dumps({"state": good, "probs": [0.1, 0.1, 0.1, 0.7], "urgency": 0, "p_jump": 0, "ms": 2}) + "\n")
    f.write('{"state": {"threat": "no')
t = load_table(p, "laya")
assert list(t) == [base_key(good)] and t[base_key(good)].probs == (0.1, 0.1, 0.1, 0.7)
assert t[base_key(good)].backend == "laya table"
# goal precedence
rules = rules_backend()
calm = {**s, "goal_rest": "yes"}                             # s: the describe() state above (no threat)
t2 = {base_key(s): Decision((0.25,) * 4, 0.5, 0.5, "laya table", 9.0)}
assert lookup(calm, t2, rules).backend == "rules" and lookup(calm, t2, rules).probs == (0, 0, 0, 1)
assert lookup(s, t2, rules).backend == "laya table"
threat = {**s, "threat": "imminent", "goal_rest": "yes"}
t2[base_key(threat)] = Decision((0.1, 0.7, 0.1, 0.1), 1.0, 0.9, "laya table", 9.0)
assert lookup(threat, t2, rules).backend == "laya table"
assert lookup({**s, "odor": "durian"}, t2, rules).backend == "rules"     # unknown_state_uses_rules
# table identity
assert table_path("rules") is None and table_path("laya").name.startswith("laya-convaiinnovations-laya-typed-decisions-")
before = table_path("laya"); QUESTIONS["jump"]["instructions"] += " "
assert table_path("laya") != before; QUESTIONS["jump"]["instructions"] = QUESTIONS["jump"]["instructions"][:-1]
```

- [ ] **Step 2: Run it and check it fails.** Run `"$PY" system1_engine.py`. Expected: `NameError: name 'all_states' is not defined`.
- [ ] **Step 3: Implement the Interfaces above.**
  - `valid_answer` checks: the state's keys equal `BASE_KEYS` exactly; every value is in `VOCAB`; `probs` is 4 finite floats in 0..1 with `abs(sum - 1) <= 1e-3`; `urgency` and `p_jump` are finite and in 0..1.
  - `load_table` catches `json.JSONDecodeError`, `KeyError`, `TypeError` and `ValueError` per line. If any lines were skipped, it prints `[system1] <path>: skipped N invalid lines` once.
  - Move the vocabulary constants and `all_states` out of `eval_system1.py`, and import them back there.
  - Add `tables/` to `.gitignore`.
- [ ] **Step 4: Run it and check it passes.** Run `"$PY" system1_engine.py` and `"$PY" eval_system1.py --help`. Expected: the self-check OK line, and the help text with no ImportError.
- [ ] **Step 5: Commit.** `git add system1_engine.py eval_system1.py .gitignore`, then `git commit -m "System 1 tables: state vocabulary, table file, goal-aware lookup"`.

### Task 2: The filler and `--compile` (`system1_engine.py`)

**Files:**
- Modify: `system1_engine.py`

**Interfaces:**
- Consumes (Task 1): `fill_order`, `base_key`, `valid_answer`, `table_path`, `load_table`, `append_table`, `N_STATES`, and `BACKENDS`.
- Produces:
  - `CHUNK = 32`, `STALL_CHUNKS = 5`.
  - `filler_init(name: str) -> None`, the worker initializer. It builds `BACKENDS[name]()` into a module global; on any exception it prints `[system1] {name} unavailable: ...` and stores None.
  - `compile_chunk(states: list[dict], decide=None) -> list[tuple[dict, Decision | None]] | None`:
    - `decide` defaults to the module global;
    - returns None when no backend is loaded;
    - returns `(state, None)` for a state whose call raised or whose answer fails `valid_answer`;
    - never calls rules.
  - `missing(table: dict) -> list[dict]`, which is `fill_order()` minus the keys already in `table`.
  - `compile_table(name: str, tables_dir="tables", decide=None) -> dict`:
    - in-process, chunk by chunk, until complete or stalled;
    - appends each chunk's successes to the file;
    - prints progress, the time at which threat states were covered, and the total time;
    - returns the table.
  - CLI: `python system1_engine.py --compile <name>` calls `compile_table(name)`. Plain `python system1_engine.py` still runs the self-check.

- [ ] **Step 1: Write the failing self-check additions** (temp dir; the stub is an in-process function):

```python
calls = []
def stub(state):
    calls.append(base_key(state))
    if state["home"] == "behind, far":
        raise TimeoutError("stub")
    if state["home"] == "left, near":
        return Decision((float("nan"),) * 4, 0.0, 0.0, "stub", 1.0)          # bad_answer_not_stored
    return rules_backend()(state)._replace(backend="stub", ms=5.0)
res = compile_chunk(fill_order()[:32], decide=stub)
assert len(res) == 32 and all(d is None or d.backend == "stub" for _, d in res)
tbl = compile_table("synthetic", tmp, decide=stub)                       # writes tables/synthetic-...jsonl
bad = [s for s in all_states() if s["home"] in ("behind, far", "left, near")]
assert len(tbl) == N_STATES - len(bad) and all(base_key(s) not in tbl for s in bad)
assert len(load_table(table_path("synthetic", tmp), "synthetic")) == len(tbl)
calls.clear()                                                            # resume_fills_only_missing
compile_table("synthetic", tmp, decide=stub)
assert set(calls) == {base_key(s) for s in bad} and len(calls) == len(bad)
calls.clear()                                                            # stall after 5 empty chunks
def broken(state):
    calls.append(1)
    raise TimeoutError("down")
assert compile_table("synthetic", tmp + "/stall", decide=broken) == {} and len(calls) == STALL_CHUNKS * CHUNK
```

- [ ] **Step 2: Run it and check it fails.** Run `"$PY" system1_engine.py`. Expected: `NameError: name 'compile_chunk' is not defined`.
- [ ] **Step 3: Implement the Interfaces above.** `compile_table` loads the table, loops over `missing(table)` in chunks of `CHUNK`, appends the successes, and keeps a no-success streak that stops the loop at `STALL_CHUNKS`. With `decide=None` it first calls `filler_init(name)`, and if that leaves nothing loaded it prints the reason and returns.
- [ ] **Step 4: Run it and check it passes.** Run `"$PY" system1_engine.py`. Expected: the OK line.
- [ ] **Step 5: Commit.** `git commit -am "System 1 tables: background filler and --compile"`.

### Task 3: `Sim` decides by table (`agent_loop.py`)

**Files:**
- Modify: `agent_loop.py`: `_system1_process`, `Sim.__init__`, `reset`, `_submit`, `step`, `close`, `run`, `main`, `selfcheck`
- Modify: `system1_engine.py`: delete `worker_decide`, `_cache`, `_chain` and `worker_init`, now unused; update the module docstring's last paragraph to describe the filler

**Interfaces:**
- Consumes (Tasks 1–2): `table_path`, `load_table`, `append_table`, `lookup`, `missing`, `compile_chunk`, `filler_init`, `N_STATES`, `CHUNK`, `STALL_CHUNKS` and `rules_backend`.
- Produces:
  - `Sim(device="cpu", n_kc=2000, backends=("rules",), system1_cpus=(), gyro_bias=0.0005, landmark_gain=0.02, wall=False, tables_dir="tables")`.
  - Attributes:
    - `table: dict`
    - `table_status: str`, one of `filling | complete | stalled | off`
    - `model: str | None`, the first non-rules name
    - `lookup_us: list[float]`
  - New snapshot keys:
    - `"table": {"backend": model or "rules", "filled": len(table), "total": 1938, "status": table_status}`;
    - `"decided_by"`: the `backend` of the decision currently applied, `"none"` before the first.
  - CLI `--tables-dir` (default `tables`).

Behaviour:
- **`__init__`:**
  - `self.rules = rules_backend()`; `model` is the first name in `backends` other than `rules`.
  - If `model` is None, or `table_path(model)` is None, the status is `off`.
  - Otherwise, load the table. If it's complete, the status is `complete` and no pool starts. If not, the status is `filling`, the pool starts with initializer `_system1_process(model, cpus)` (pin, then `filler_init`), and `Sim` submits the first chunk of `missing(table)`.
  - Drop the old warm-up `.result()` call.
- **`_submit(state)`:** returns `_Done(lookup(state, self.table, self.rules))` and appends the elapsed µs to `lookup_us`. The rest of `step()`'s pending/latch logic is unchanged.
- **`step()` fill poll,** before the decision code. If the fill future is `done()`:
  - a None result, or a raised exception, means the status becomes `off`;
  - otherwise, merge the successes into `table`, append them to the file, and update the no-success streak;
  - then the status becomes `complete` if the table is full, or `stalled` at `STALL_CHUNKS`; otherwise submit the next chunk.
- **`reset()`:** keeps the table, the filler and its status. Only the world and the brain reset.
- **`run()`:** replace the `system1` stats line with:

```
system1        decisions N by {...}  lookup p50 X p99 Y us  sent->latched p50 .. ms  table laya 1938/1938 complete
```

- [ ] **Step 1: Write the failing self-check additions** in `selfcheck()`, in a `tempfile.TemporaryDirectory()`:

```python
from system1_engine import Decision, N_STATES, all_states, append_table, table_path
full = [(s, Decision((0.25, 0.25, 0.25, 0.25), 0.5, 0.5, "laya", 40.0)) for s in all_states()]
append_table(table_path("laya", tmp), full)
sim = Sim(backends=("laya",), tables_dir=tmp)                           # complete_table_no_pool
assert sim.pool is None and sim.table_status == "complete" and "laya" not in sys.modules
sim.step(); snap = sim.step()
assert snap["decided_by"] == "laya table" and snap["table"]["filled"] == N_STATES, snap["table"]
sim.set_goal(parse("rest")[0]); sim.step(); snap = sim.step()
assert snap["decided_by"] == "rules" and snap["behaviour"] == "IDLE"   # goal precedence
sim.close()
# unavailable_model_is_off: .venv has no laya and no CUDA, so the filler's backend fails to load
sim = Sim(backends=("laya",), tables_dir=tmp + "/empty")
t0 = time.perf_counter()
while sim.table_status == "filling" and time.perf_counter() - t0 < 60:
    sim.step()
assert sim.table_status == "off" and sim.step()["decided_by"] == "rules", sim.table_status
sim.close()
```

Keep every existing selfcheck block unchanged. They run `Sim()` with rules and are the regression oracle.

- [ ] **Step 2: Run it and check it fails.** Run `"$PY" agent_loop.py --selfcheck`. Expected: `TypeError: ... unexpected keyword argument 'tables_dir'`.
- [ ] **Step 3: Implement the behaviour above, and delete the dead worker code** in `system1_engine.py`.
- [ ] **Step 4: Run the checks and confirm they pass.**
  - `"$PY" agent_loop.py --selfcheck` and `"$PY" system1_engine.py`: both print OK.
  - The regression oracle in Global Constraints: exact counts.
  - `"$PY" serve.py --selfcheck`: OK. It calls `Sim(wall=True)` and must be unaffected.
- [ ] **Step 5: Commit.** `git commit -am "Sim decides System 1 by precompiled table, rules on a miss"`.

### Task 4: Page and server (`serve.py`, `web/index.html`)

**Files:**
- Modify: `serve.py` `main()`: add `--backends` (default `rules`) and `--tables-dir`, passed to `Sim`, and print the table line at startup.
- Modify: `web/index.html`: add one status line under `#jump` in the Brain section.

**Interfaces:**
- Consumes (Task 3): snapshot `table` and `decided_by`.

- [ ] **Step 1: Read the design-engineering skill first** (`C:\Users\kj638\.ai-skills\design-engineering\skills\design-engineering\SKILL.md`, the typography and UI-copy references only).
- [ ] **Step 2: Add the line.** Add `<p class="s1" id="s1" aria-live="off">System 1: rules</p>` after `#jump`, styled like `.jump` (same size and colour tokens; no new colours). Render it from each snapshot:
  - `table.status == "off"` and `backend == "rules"`: `System 1: rules`
  - `filling`: `System 1: {backend} table {filled} / 1,938, rules cover the rest`
  - `complete`: `System 1: {backend} table complete`
  - `stalled`: `System 1: {backend} table stalled at {filled} / 1,938, rules cover the rest`
  - `off` with a model backend: `System 1: {backend} unavailable, rules`
  
  Format numbers with `toLocaleString("en-US")`. Write to the DOM only when the text changes.
- [ ] **Step 3: Verify in the browser pane.**
  - Run `preview_start {name: "flyagent"}` (the default `rules`). The line reads `System 1: rules`, and the console shows no errors.
  - Then run once with a complete table: `"$PY" serve.py --backends laya --tables-dir <tmp with the Task 3 full table> --port 8766`, opened via the Browser pane. The line reads `System 1: laya table complete`. Take a screenshot.
  - Run `"$PY" serve.py --selfcheck`: OK.
- [ ] **Step 4: Commit.** `git commit -am "Page shows which System 1 decides and how full its table is"`.

### Task 5: `eval_system1.py --latency`

**Files:**
- Modify: `eval_system1.py`

**Interfaces:**
- Consumes: `table_path`, `load_table`, `lookup`, `rules_backend`, `BACKENDS`, `fill_order`, `base_key` (Tasks 1–2), and `Sim` (Task 3).
- Produces: `python eval_system1.py <backend> --latency [--tables-dir tables]`. It requires a complete table and prints four blocks:
  1. **`live`:** p50/p99 ms of `BACKENDS[backend]()` over 100 states sampled with `random.Random(0)` (warm, back-to-back). Laya uses its per-state backend here, not `laya_decisions`.
  2. **`lookup`:** p50/p99 µs of `lookup` over the same 100 states, under `DEFAULT_GOAL` (100 repeats each, `time.perf_counter_ns`).
  3. **`same answers`:** the fraction of the 100 whose argmax choice matches the table, and the max |ΔP|.
  4. **`loom latch`:** 20 headless runs. Each run builds `Sim(backends=(backend,), tables_dir=...)`, steps 200 ticks, then calls `launch_predator(bearing=k * 2π / 20)`, and counts the ticks until `snapshot["behaviour"] == "FLEE"` (capped at 30, which counts as "not within the loom"). It reports p50/max ticks and ms (× 15), and the same for `Sim()` with rules.

- [ ] **Step 1: Add the mode, with a `--skip-live` flag** that skips blocks 1 and 3, for CPU-only checks.
- [ ] **Step 2: Smoke-test it in `.venv`.**
  - First write a full table of rules answers to a temp dir: `append_table(table_path("laya", tmp), [(s, rules(s)._replace(backend="laya")) for s in all_states()])`.
  - Then run `"$PY" eval_system1.py laya --latency --skip-live --tables-dir <tmp>`.
  - Expected: blocks 2 and 4 print. The lookup p50 is under 50 µs. The table loom-latch ticks equal rules' exactly, because the table *is* rules' answers. Any difference means the lookup path is wrong.
- [ ] **Step 3: Commit.** `git commit -am "eval_system1 --latency: live vs table vs rules"`.

### Task 6: Build the tables, measure, document

**Files:**
- Create (force-added): `tables/laya-convaiinnovations-laya-typed-decisions-<hash8>.jsonl` and `tables/llm-qwen3-4b-instruct-2507-q4_K_M-<hash8>.jsonl`
- Modify: `README.md`: new `### 3f. Precompiled System 1` after 3e; update the "Not done" list if an item is resolved
- Modify: `docs/DECISIONS.md`, `docs/VERIFY.md`, `docs/STATE.md`

- [ ] **Step 1: Compile Laya.** Run `PYTHONPATH=.deps "$CPY" system1_engine.py --compile laya` and record the threat-coverage time and the total time.
- [ ] **Step 2: Compile Qwen3-4B.** First warm Ollama with one request, then run `LLM_TIMEOUT=60 LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M "$CPY" system1_engine.py --compile llm` (about 16 min; run it in the background). Record both times.
- [ ] **Step 3: Measure.**
  - Run `eval_system1.py laya --latency` (CUDA env with `.deps`) and `eval_system1.py llm --latency` with the Ollama env.
  - Then the filler's effect on the tick: `PYTHONPATH=.deps "$CPY" agent_loop.py --backends laya,rules --tables-dir <empty tmp> --ticks 4000 --tick-cpus 2,3 --system1-cpus 4-7`, run 3 times, reporting the tick period p50/p99 and overruns.
  - Then the CPU-only check: `"$PY" agent_loop.py --backends laya --ticks 2000` with the committed table (expect `table laya 1938/1938 complete`, no Laya import).
- [ ] **Step 4: Write the docs.**
  - **README 3f:** a table with rows live model / table / rules and columns decision p50, p99, loom latch, same answers, compile time, with every number from Step 3. One paragraph on why: a finite worded state space, and models that ignore the goal fields. One line on the goal precedence, and the CPU-only-clone point.
  - **`docs/DECISIONS.md`:** in-process lookup; goal precedence (any goal field → rules); one table per run; committed tables.
  - **`docs/VERIFY.md`:** the Task 1–5 commands, plus the regression oracle.
  - **`docs/STATE.md`:** what changed, and the next step (sub-project 3).
- [ ] **Step 5: Commit.** `git add -f tables/*.jsonl`, `git add README.md docs`, then `git commit -m "Precompiled Laya and Qwen3-4B tables; README 3f measurements"`.
