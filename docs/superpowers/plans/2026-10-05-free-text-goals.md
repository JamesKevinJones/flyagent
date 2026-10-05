# Free-text Goals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a person type a goal in plain English on a local web page and watch the fly pursue it, with the interpretation shown and measured.

**Architecture:**
- **Interpretation:** a parser (with an optional LLM on top) turns text into a small `Goal` once.
- **Compilation:** `Sim.set_goal` turns the `Goal` into brain buffers (odor signs, heading drive, home sign) and goal fields that the rule table reads.
- **Serving:** a stdlib server runs the same `Sim` in a 15 ms thread and streams snapshots to one static page over SSE.

**Tech Stack:** Python 3.12, torch 2.14 (existing), stdlib `http.server` / `threading` / `json`; vanilla JS and canvas; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-05-free-text-goals-design.md`

## Global Constraints

- **No new dependencies.** Server is stdlib only; the page is one static file with no build step.
- **Unchanged behaviour:** 15 ms tick, escape reflex, `--device cpu` and `--backends rules` defaults, and every existing self-check. The empty goal reproduces today's behaviour exactly.
- **Regression oracle:** the CLI default run must still print `{'FORAGE': 1847, 'FLEE': 25, 'ORIENT': 124, 'IDLE': 4}`, `jumps 22` and `rewards ['banana', 'geosmin']`:
  `"$PY" agent_loop.py --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7`
- **Vocabulary:**
  - seek ⊆ {banana, home}; avoid ⊆ {geosmin, home}
  - heading ∈ {north, north-east, east, south-east, south, south-west, west, north-west}, with north = π/2 rad in the landmark frame (east = 0, counterclockwise)
  - rest: bool
- **Limits:** goal text at most 200 chars (400 otherwise); request body at most 2 KB (413 otherwise); bind `127.0.0.1` only, default port 8765.
- **Values:** heading weight 0.4 when a heading is set; odor-match threshold 0.6; arena wall ±60 body lengths, off in the CLI; SSE snapshot every other tick.
- **Keys:** LLM keys come from env only (`GEMINI_API_KEY`, or `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`) and never reach the browser.
- **Tests:** one runnable `__main__` self-check per module (`assert`-based, no framework). Run with `PY="C:/Users/kj638/Kevin codes/ComfyUI/.venv/Scripts/python.exe"`.
- **Commits:** no Claude co-author lines.

## Review Focus

1. **Goal changed while a System 1 decision is in flight.** The stale answer for the old goal must be discarded, not latched. Test: Task 4, `stale_decision_discarded`.
2. **Presets clicked quickly.** Out-of-order interpretations must never overwrite a newer goal; the last submitted wins. Test: Task 5, `last_goal_wins`.
3. **Messy casing, punctuation and compass spellings.** "FIND the BANANA!!", "north east", "North-East" and "NE" must parse the same as the clean forms. Test: Task 1 cases.
4. **The browser tab closes mid-stream.** The server must not raise or stall the sim, and ticks keep advancing. Test: Task 5, `client_disconnect`.
5. **A heading goal pointing into the wall.** "head north" for 3,000 ticks: the fly stays inside ±60, positions stay finite, and nothing becomes NaN. Test: Task 4, `wall_holds`.

---

### Task 1: Goal model, parser, and interpretation (`goals.py`)

**Files:**
- Create: `goals.py`
- Modify: `system1_engine.py`, splitting `llm_backend()`'s env resolution out into `llm_config()`

**Interfaces:**
- Produces, in `goals.py`:
  - `class Goal(NamedTuple)` with fields `seek: tuple, avoid: tuple, heading: str|None, rest: bool, text: str, source: str` (field order exactly as written)
  - `DEFAULT_GOAL = Goal((), (), None, False, "", "default")`
  - `HEADINGS: dict[str, float]`, the 8 compass words → radians
  - `validate(goal: Goal) -> str|None`, returning an error message or None
  - `parse(text: str) -> tuple[Goal, tuple[str, ...]]`, returning the goal (`source="parser"`) and the ignored content words
  - `class Interpretation(NamedTuple)` with fields `goal: Goal, ignored: tuple, ms: float, note: str|None`
  - `interpret(text: str, llm=None) -> Interpretation`, where `llm` is a callable `(text: str) -> str` returning the raw model reply, or None
  - `make_llm() -> callable|None`, built from env via `system1_engine.llm_config()`; None when unconfigured
- Produces, in `system1_engine.py`: `llm_config() -> tuple[str, str|None, str, float]|None`, returning (url, key, model, timeout). `llm_backend()` uses it unchanged.
- **Import rule:** `goals.py` imports `system1_engine` only inside `make_llm()`, because `system1_engine` imports `goals` (Task 3).

- [ ] **Step 1: Write the failing self-check.** Add `if __name__ == "__main__":` to `goals.py` with these assertions (`g(...)` builds a Goal with `text`/`source` ignored when comparing; compare `(seek, avoid, heading, rest)`):

```python
cases = [
    ("find the banana", (("banana",), (), None, False)),
    ("FIND the BANANA!!", (("banana",), (), None, False)),
    ("avoid the smell and head north", ((), ("geosmin",), "north", False)),
    ("don't go near the mould, go home", (("home",), ("geosmin",), None, False)),
    ("head north east", ((), (), "north-east", False)),
    ("go North-East", ((), (), "north-east", False)),
    ("go up", ((), (), "north", False)),
    ("rest", ((), (), None, True)),
    ("rest and find food", (("banana",), (), None, False)),         # rest dropped, reported
    ("find the banana but avoid the banana", ((), (), None, False)),  # conflict dropped, reported
    ("stay away from home", ((), ("home",), None, False)),
    ("", ((), (), None, False)),
]
```

  Plus:
  - `parse("rest and find food")[1] == ("rest",)`
  - `parse("find the banana but avoid the banana")[1] == ("banana",)`
  - `parse("I'm starving")[1] == ("starving",)`, with stop-words ("I'm", "the", "a", "and") never reported

  `interpret` checks use a fake `llm`, a plain function:
  - Fully parsed text never calls `llm`; `source == "parser"`.
  - Reply `'{"seek":["banana"],"avoid":["geosmin"],"heading":null,"rest":false}'` for "I'm starving but that stink is gross" gives `source == "llm"` and `ignored == ()`.
  - Reply `'{"seek":["pizza"]}'` falls back to the parser result, with `note` containing "invalid".
  - A reply that raises `TimeoutError` falls back, with `note` containing "TimeoutError".
  - Reply `'not json'` falls back, with `note` containing "json".
  - `interpret("x"*201)` raises `ValueError`.
  - `interpret("").goal == DEFAULT_GOAL` and `interpret("   ").goal == DEFAULT_GOAL`, with source "default".

- [ ] **Step 2: Run it and confirm it fails.** Run `"$PY" goals.py`; expect `NameError`/`ImportError`.
- [ ] **Step 3: Implement `goals.py`** with the interfaces above.
  - Normalise to lowercase and replace punctuation except apostrophes with spaces.
  - Join "north east" / "north-east" / "ne" (and the other 7) into the canonical hyphenated word before tokenising.
  - Split clauses on "and", "but" and ",". The most recent verb in a clause sets the mode for the nouns that follow.
  - Synonyms and verbs exactly as spec section 1.
  - The LLM prompt lists the schema and allowed values. Parse the reply between the first `{` and the last `}`; reject unknown keys or values and any `validate()` error.
  - `make_llm()` wraps `system1_engine._poster`, posting to `/chat/completions` with `response_format={"type":"json_object"}`, and returns the message content.
- [ ] **Step 4: Refactor `llm_config()`** out of `llm_backend()` with no behaviour change. `"$PY" system1_engine.py` still prints `self-check OK`.
- [ ] **Step 5: Run `"$PY" goals.py`**; expect `goals self-check OK`.
- [ ] **Step 6: Commit** `goals.py` and `system1_engine.py`: "Add goal model, parser and LLM interpretation".

### Task 2: Goal buffers in the brain (`fruit_fly_circuits.py`)

**Files:**
- Modify: `fruit_fly_circuits.py` (constructor buffers, `_step` VNC section, `demo()`)

**Interfaces:**
- Produces: `FlyBrain.set_goal(odor_signs: list[float], goal_heading: float, heading_weight: float, home_sign: float) -> None`. `odor_signs` has length `n_odor_tags` (32). It is a host call outside the graph that writes persistent device tensors in place, so a captured graph sees new values without re-capture.
- **Defaults at construction:** signs all +1, heading 0, weight 0, home_sign +1. That's identical to today.

- [ ] **Step 1: Add failing checks to `demo()`**, after the existing ones, in a fresh `FlyBrain` per check, on both CPU and the CUDA graph:
  - `odor_sign_flips_turn`: remember odor A as id 3, present odor A, and set `IN_P_BEHAVIOUR` to FORAGE=1 and `IN_ODOR_LR = 0.5`. With default signs `OUT_TURN > 0`; after `set_goal(signs with [3] = -1, 0, 0, 1)`, `OUT_TURN < 0`.
  - `heading_drive_north`: heading starts at 0 (east). After `set_goal([1]*32, math.pi/2, 0.4, 1)` with p(FORAGE)=1 and no odor, `OUT_TURN > 0` (turn left toward north). With `math.pi*3/2` (south) it's < 0.
  - `home_sign_flips_orient`: with p(ORIENT)=1 and a nonzero home vector, `OUT_TURN` changes sign when `home_sign=-1`.
  - The existing checks pass unchanged. Run them first in the new `demo()`, before any `set_goal` call.
- [ ] **Step 2: Run it and confirm it fails.** Run `"$PY" fruit_fly_circuits.py`; expect `AttributeError: set_goal`.
- [ ] **Step 3: Implement.** Buffers `self.odor_sign` (n_odor_tags) and `self.goal_vec` (3: heading, weight, home_sign) on the device; `set_goal` uses `copy_`. In `_step`:
  - the FORAGE turn becomes `torch.where(match_val > 0.6, self.odor_sign[best_idx], 1.0) * x[IN_ODOR_LR] + w * torch.sin(goal_heading - heading)`
  - the ORIENT turn becomes `home_sign * home_turn`
  - Capture never writes these buffers, so `_state()` stays unchanged.
- [ ] **Step 4: Run `"$PY" fruit_fly_circuits.py`**; expect both `[cpu]` and `[cuda] self-check OK`.
- [ ] **Step 5: Run the regression oracle** (Global Constraints) and confirm the same numbers.
- [ ] **Step 6: Commit**: "Add goal buffers: odor signs, heading drive, home sign".

### Task 3: Goal-aware state and rules (`system1_engine.py`)

**Files:**
- Modify: `system1_engine.py` (`describe`, `rules_backend`, self-check)

**Interfaces:**
- Consumes: `goals.Goal`, `goals.DEFAULT_GOAL`
- Produces: `describe(out, loom, odor_names, goal=DEFAULT_GOAL) -> dict`. It adds these keys:
  - `goal_seek`: `"none"` or the items joined with `"+"` in vocabulary order (`banana+home`)
  - `goal_avoid`: same format as `goal_seek`
  - `goal_heading`: the word or `"none"`
  - `goal_rest`: `"yes"`/`"no"`
- `rules_backend` must accept states **without** goal keys (`eval_system1.py` states) via `state.get(key, "none"/"no")`.

- [ ] **Step 1: Add failing self-check assertions.** Build states with `describe(out, loom, names, goal)` and decide with `rules_backend()`. Argmax must be:
  - FLEE for loom 0.6 with each of: rest, seek banana, seek home, heading north
  - IDLE for rest with no threat
  - ORIENT for seek home
  - ORIENT for avoid home when `home` is "here" or "…, near"
  - FORAGE for avoid home when home is far
  - FORAGE for seek banana
  - FORAGE for heading north
  - today's choice for `DEFAULT_GOAL`, against the current assertions
  - unchanged results for a state dict with no goal keys
- [ ] **Step 2: Run `"$PY" system1_engine.py`**; expect failure (unexpected keyword `goal`).
- [ ] **Step 3: Implement** the priority from spec section 2. The existing probability tuples for FLEE/IDLE/ORIENT/FORAGE are reused unchanged.
- [ ] **Step 4: Run `"$PY" system1_engine.py`**; expect `self-check OK`. Then run the regression oracle; the numbers are unchanged.
- [ ] **Step 5: Commit**: "Add goal fields to the worded state and goal-aware rules".

### Task 4: `Sim` refactor, goal compilation, wall, predator on demand (`agent_loop.py`)

**Files:**
- Modify: `agent_loop.py`

**Interfaces:**
- Consumes: `FlyBrain.set_goal` (Task 2), `describe(..., goal)` (Task 3), `goals.Goal`/`DEFAULT_GOAL`
- Produces: `class Sim` with:
  - `Sim(device="cpu", n_kc=2000, backends=("rules",), system1_cpus=(), gyro_bias=0.0005, landmark_gain=0.02, wall=False)`
  - `.step() -> dict`, a snapshot with keys `t, x, y, heading, bump (16 floats), probs (dict BEHAVIOURS→float), behaviour, jump, odor, home {x,y}, loom, tick_p50, tick_p99`
  - `.set_goal(goal: Goal) -> None`
  - `.launch_predator(bearing: float|None = None) -> None`
  - `.reset() -> None`
  - `.close() -> None`
  - `.goal`, `.tick`
- `run(args)` (the CLI) builds on `Sim`. It keeps its flags and printed stats, and launches the scripted predator at `ticks//2` with the same seeded bearing as today.
- `World` gains `wall: bool` (±60, reflecting heading at the edge) and `launch_predator(tick, bearing)`. The loom becomes a function of ticks since launch, not of `n_ticks`.
- **Goal compilation in `Sim.set_goal`:**
  - signs: if the goal has seek or avoid, a remembered odor gets +1 if in seek, −1 if in avoid, else 0; with neither, all +1
  - heading: weight 0.4 when set, else 0
  - home_sign: −1 if "home" in avoid, else +1
  - the goal is stored for `describe()`
- **Stale decisions:** a pending System 1 result is applied only if its state's goal fields equal the current goal fields; otherwise it's discarded and re-asked.

- [ ] **Step 1: Write a failing self-check**, a new `if __name__ == "__main__" and "--selfcheck" in sys.argv` block, so the CLI's argparse path stays untouched:
  - `stale_decision_discarded`: set a goal, step until a decision is pending, switch to `rest`, step 40 ticks, and assert the latched argmax is IDLE.
  - `wall_holds`: `Sim(wall=True)`, `set_goal(parse("head north")[0])`, 3,000 steps; assert all |x|,|y| ≤ 60 and finite.
  - `seek_banana_reaches`: 2,000 steps with "find the banana" from reset; `"banana" in sim.world.rewarded`.
  - `default_matches_cli`: 2,000 `Sim` steps with the scripted predator give the same behaviour tick counts as the regression oracle.
- [ ] **Step 2: Run `"$PY" agent_loop.py --selfcheck`**; expect `NameError: Sim`.
- [ ] **Step 3: Implement `Sim`, the `World` changes and the CLI-on-`Sim` refactor.** Keep the deadline/priority/pinning code in `run()`; `Sim.step()` holds only one tick of work.
- [ ] **Step 4: Run `"$PY" agent_loop.py --selfcheck`**, then the regression oracle; both must pass with identical oracle numbers.
- [ ] **Step 5: Commit**: "Extract Sim, compile goals, add arena wall and on-demand predator".

### Task 5: Local server (`serve.py`)

**Files:**
- Create: `serve.py`

**Interfaces:**
- Consumes: `Sim` (Task 4), `goals.interpret`/`make_llm` (Task 1)
- Produces:
  - `make_server(sim, port=8765, llm=None) -> ThreadingHTTPServer` on `127.0.0.1`. `main()` passes `goals.make_llm()`; tests inject a fake.
  - `SimRunner(sim)`, a thread that steps every 15 ms with the same deadline loop as the CLI. It has `.latest` (snapshot) and `.cond` (threading.Condition, notified every other tick), plus `.start()`/`.stop()`.
  - `main()` with `--port`, `--device`
- **Endpoints**, exactly as spec section 3:

  | Endpoint | Behaviour |
  |---|---|
  | `GET /` | serves `web/index.html` |
  | `GET /stream` | `text/event-stream` lines `data: <json>\n\n` |
  | `POST /goal` | `{text}` → 200 `{goal: {...}, source, ignored, ms, note}` |
  | `POST /predator`, `POST /reset` | 204 |
  | any other path | 404 |

- **Goal ordering:** each `POST /goal` takes a sequence number. When an interpretation finishes, its goal is applied only if no later request has been submitted; the response still returns its own interpretation.

- [ ] **Step 1: Write a failing self-check** (`"$PY" serve.py --selfcheck`): a server on port 0 with a CPU `Sim`, and the client using `http.client`:
  - `GET /` → 200, with `<html` in the body
  - `GET /stream` → 3 `data:` events parse as JSON with the snapshot keys
  - `POST /goal {"text":"find the banana"}` → 200, `source == "parser"`, `goal.seek == ["banana"]`
  - a 3 KB body → 413; text of 201 chars → 400; invalid JSON → 400; `GET /../goals.py` → 404
  - `last_goal_wins`: make the server with a fake `llm` that sleeps 1 s and returns `{"seek":[],"avoid":[],"heading":null,"rest":true}`. Send two `POST /goal` in parallel threads: first "rest, I'm sleepy" (not fully parsed, so the slow LLM runs), then 0.1 s later "go home" (parser, instant). After both return, `sim.goal.seek == ("home",)`.
  - `client_disconnect`: open `/stream`, read 1 event, close the socket, sleep 0.3 s; `runner` tick count still increases and the server log has no traceback
- [ ] **Step 2: Run it and confirm it fails** (no `serve.py`).
- [ ] **Step 3: Implement `serve.py`.** Write a placeholder `web/index.html` containing `<html><body>flyagent</body></html>` so `GET /` passes; Task 6 replaces it. Static serving maps only `/` to that file.
- [ ] **Step 4: Run `"$PY" serve.py --selfcheck`**; expect `serve self-check OK`.
- [ ] **Step 5: Commit** `serve.py` and `web/index.html`: "Add local server with SSE stream and goal endpoint".

### Task 6: The page (`web/index.html`)

**Files:**
- Modify: `web/index.html`
- Create: `.claude/launch.json` with entry `flyagent` → a `.cmd` wrapper `web/serve.cmd` that runs `serve.py`, port 8765. Per memory, `preview_start` breaks on paths with spaces, so the wrapper is required.

**Interfaces:**
- Consumes: the endpoints and snapshot keys from Task 5. Nothing consumes the page.
- **Design inputs:** load the `design-engineering` skill before writing. Palette: #0d0f0c / #ece8dc / #f0a03c / #e2513b. Font: Cascadia Code, copied from `brag-output/composition/assets/fonts/CascadiaCode.ttf` to `web/CascadiaCode.ttf` (OFL). `serve.py` must also serve `/CascadiaCode.ttf`, its one extra static route.
- **Layout and controls**, as spec section 3:
  - arena canvas (odor halos, home, fly marker and trail, loom ring, wall)
  - goal box with 4 preset chips and an interpretation card (structured goal, source badge, ignored words, ms)
  - compass (16 wedges from `bump`), behaviour bars, jump indicator, footer tick p50/p99
  - Predator and Reset buttons
  - a disconnected banner when the EventSource errors

- [ ] **Step 1: Write the page.** Accessibility basics: labelled input, buttons are `<button>`, visible focus, `aria-live="polite"` on the interpretation card.
- [ ] **Step 2: Verify in the built-in browser**: `preview_start {name:"flyagent"}`.
  - `read_console_messages` shows no errors.
  - Typing "avoid the smell and head north" shows `avoid: geosmin, heading: north, source: parser`.
  - The trail turns north within about 5 s.
  - Predator shows the loom ring, FLEE dominates and the jump indicator flashes.
  - Reset clears the trail.
  - Mobile width (375 px): no horizontal scroll; the panels stack.
- [ ] **Step 3: Screenshot the working page**, for the user and for Task 8.
- [ ] **Step 4: Commit** `web/` and `.claude/launch.json`: "Add the interactive goal page".

### Task 7: Measurements (`eval_goals.py`)

**Files:**
- Create: `eval_goals.py`

**Interfaces:**
- Consumes: `goals.parse`/`interpret`/`make_llm`, `Sim`
- Produces: `PHRASES`, a list of 30 `(text, (seek, avoid, heading, rest))`:
  - 10 plain (all parser-solvable)
  - 10 paraphrased ("grab some fruit", "get back to the nest")
  - 10 messy ("I'm starving but that mouldy stink is gross")
  - Labels fixed in the file.
- **Functions:**
  - `interpretation_accuracy(llm) -> dict`: exact-match rate for parser-only and parser+LLM, and LLM ms p50/p95
  - `follow(goal_text, ticks=2000, seed=0) -> dict`: the spec section 4 metric per preset, plus the no-goal baseline
  - `serving_cost() -> dict`: tick p50/p99 for a headless `SimRunner` vs `SimRunner` + server + one SSE client, 2,000 ticks each
- CLI: `"$PY" eval_goals.py` prints all three tables; `--no-llm` skips the LLM column.

- [ ] **Step 1: Write the self-check** (`--selfcheck`): the parser-only accuracy on the 10 plain phrases is 100% (they are written to be parser-solvable), `follow("find the banana")["reached"] is True`, and `follow("rest")["idle_share"] > 0.9`.
- [ ] **Step 2: Run it and confirm it fails** (no file). Implement it, then run `--selfcheck`; expect OK.
- [ ] **Step 3: Run the full eval** with `LLM_TIMEOUT=60 LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M "$PY" eval_goals.py` and save the output to `docs/eval-goals-2026-10-05.txt` for Task 8.
- [ ] **Step 4: Commit** `eval_goals.py` and the output: "Add goal interpretation and goal-following evaluation".

### Task 8: Docs and README

**Files:**
- Modify: `README.md` (a "Try it" quick start under the video; new section 3e with Task 7's measured tables, failures included; Known limitations from the spec)
- Modify: `docs/VERIFY.md` (the self-check commands for goals/serve/agent_loop and the eval command)
- Modify: `docs/STATE.md`, `docs/DECISIONS.md` (decisions: compile-once goals, local-only, parser+LLM, SSE over WebSocket)
- Modify: `AGENTS.md` layout (new files)

- [ ] **Step 1: Write the docs.** Every number in 3e is copied from `docs/eval-goals-2026-10-05.txt`; none is typed by hand.
- [ ] **Step 2: Run every command in `docs/VERIFY.md`**; all pass. The regression oracle numbers are unchanged.
- [ ] **Step 3: Commit**: "Document free-text goals: quick start, measurements, decisions".
- [ ] **Step 4: README clip, after the user approves.** Record the page (Claude in Chrome GIF recorder, or a short Hyperframes edit), upload it inline as with the brag video (needs the user's go-ahead for the Chrome upload), commit the README URL, and run the pre-push security review before any push.
