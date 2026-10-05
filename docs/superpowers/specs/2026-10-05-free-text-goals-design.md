# Free-text goals with a local interactive page: design

**Date:** 2026-10-05 · **Status:** approved in conversation, awaiting written-spec review
**Sub-project 1 of 3.** Order: (1) free-text goals → (2) fast System 1 on novel states → (3) full connectome.
Sub-project 2 builds on this design's "compile on goal change" step; sub-project 3 is independent.

## Purpose and success

Portfolio piece. A visitor types a goal in plain English and watches the fly pursue it. Success means:

1. A cold clone works with no model installed: `python serve.py`, open the page, type a goal, see the fly act on it.
2. The page shows *how* the goal was understood: the structured goal, which interpreter produced it, and any ignored words.
3. The README reports measured interpretation accuracy and goal-following results, failures included, in the style of section 3b.
4. Nothing regresses. The 15 ms tick, the escape reflex, the CPU default and every existing self-check are unchanged, and the default (empty) goal reproduces today's behaviour exactly.

## Decisions (from the brainstorm)

| Decision | Chosen | Rejected because |
|---|---|---|
| Demo surface | Interactive local web page, plus a README clip | Terminal-only is weak for a portfolio; a recorded video alone can't be tried |
| Hosting | Local only, after cloning | A JS port means a second, unmeasured simulator; a hosted server costs money and exposes an LLM key |
| Interpreter | Built-in parser, with an LLM on top when configured | LLM-required breaks cold clones; parser-only fails on the first unexpected phrasing |
| Execution | Compile the goal once, then execute with fast rules | Per-decision LLM is 3–5 s late, Laya can't read multi-field rules, and a hybrid doubles the failure modes |

## Scope

**In:**
- the goal vocabulary below
- parser and LLM interpretation
- compiling goals into the brain and the rules
- the local server and page
- headless goal evaluation
- README section 3e
- a README clip

**Out (YAGNI):**
- multi-step plans ("food, then home")
- numeric distances
- new world objects
- a public link or hosting
- goal awareness in Laya, Jev or the LLM System 1 backends (goals are a rules feature)
- a separate left/right comparison per odor (see Known limitations)

## 1. Goal and interpretation: new `goals.py`

```python
class Goal(NamedTuple):
    seek: tuple        # subset of ("banana", "home")
    avoid: tuple       # subset of ("geosmin", "home")
    heading: str|None  # one of: north, north-east, east, south-east, south, south-west, west, north-west
    rest: bool         # stay put (IDLE) unless threatened
    text: str          # what the user typed (stripped, at most 200 chars)
    source: str        # "parser" | "llm" | "default"
```

- **`DEFAULT_GOAL`**: everything empty, `source="default"`; reproduces today's behaviour.
- **Validation:** a goal is invalid if seek and avoid share an item, or if `rest` is combined with seek or heading. Invalid LLM output is rejected; the parser never produces invalid goals.

**Parser** (`parse(text) -> (Goal, ignored_words)`), with no model:
- **Nouns, via a synonym table:**
  - food / fruit / banana / sugar / eat → `banana`
  - smell / stink / mould / mold / earthy / geosmin → `geosmin`
  - home / nest / start / base → `home`
  - the 8 compass words, plus up/top → north, down/bottom → south, left → west, right → east (map-relative)
- **Verbs:**
  - Seek: find, go to, get, seek, look for, eat, reach.
  - Avoid: avoid, stay away from, keep away from, don't go near, away from, not.
  - Rest: rest, stay, stop, wait, sit, idle.
- **How verbs attach:** a verb applies to the noun phrases that follow it until the next verb or clause boundary ("and", "but", a comma).
- **Leftover words:** content words not in any table are returned as `ignored_words`. Stop-words are not reported.
- **Conflicts, so the parser never returns an invalid goal:**
  - An item that ends up in both seek and avoid is dropped from both and reported in `ignored_words`.
  - If rest appears together with any seek or heading, rest is dropped and "rest" is reported in `ignored_words`. Moving goals win.

**Order of interpretation** (`interpret(text, llm=None) -> Interpretation(goal, ignored, ms)`):
1. Run the parser. If `ignored` is empty, return the parser's goal, `source="parser"`.
2. Otherwise, if an LLM is configured (the existing `llm_backend` env config), ask it for JSON matching the schema, with the allowed values listed in the prompt.
3. Validate the reply against the allowed values and the rules above. If it's valid, return it with `source="llm"` and empty `ignored`.
4. On a timeout, an HTTP error, malformed JSON, or an out-of-vocabulary or invalid goal, return the parser's partial goal and its `ignored` words, with `source="parser"` and the reason in a `note` field.

## 2. Execution

**Brain (`fruit_fly_circuits.py`).** A new `FlyBrain.set_goal(odor_signs, goal_heading_rad, heading_weight, home_sign)` host call, outside the CUDA graph, writes persistent device buffers that the graph reads every tick:
- **`odor_sign[32]`:** per remembered odor tag; +1 seek, −1 avoid, 0 neutral. The FORAGE turn becomes `sign[OUT_ODOR_ID] × IN_ODOR_LR` when the odor match is above 0.6. Below that, it's `default_odor_sign × IN_ODOR_LR`, where `default_odor_sign` is +1, as today.
- **Heading drive:** `heading_weight × sin(goal_heading − heading)`, added to the FORAGE turn (PFL3-style steering on the compass bump). North is +y in the landmark frame, i.e. heading π/2. A weight of 0 means no heading goal.
- **`home_sign` (±1):** multiplies the ORIENT program's home turn.

The defaults (all signs +1, weight 0, home_sign +1) reproduce today's tensors exactly.

**System 1 rules (`system1_engine.py`).**
- **State:** `describe()` gains `goal_seek`, `goal_avoid`, `goal_heading` and `goal_rest` (worded). The decision cache is keyed on the full state, so a goal change gets fresh decisions with no invalidation code.
- **Priority:**
  1. any threat → FLEE (unchanged; no goal overrides it)
  2. `rest` → IDLE
  3. seek home, or avoid home while home is near → ORIENT
  4. seek banana, or a heading goal → FORAGE
  5. empty goal → today's rules, unchanged
- **Other backends:** Laya, Jev and the LLM System 1 backend receive the goal fields but their criteria ignore them.

**Compiling a goal into the brain.** In `Sim`, a new goal sets:
- the odor-sign table, from remembered odor names → seek/avoid membership
- the heading and its weight, from the compass word (weight 0.4 when set, a calibration knob)
- the home sign
- the goal fields used by `describe()`

## 3. Server and page

**`agent_loop.py` refactor.** Extract a `Sim` class (World, FlyBrain, System 1 pool, current goal) with:
- `step() -> snapshot`
- `set_goal(goal)`
- `launch_predator()`
- `reset()`

`run()` (the CLI) keeps its flags, scripted predator at mid-run and stats output, built on `Sim`. The World gains an optional arena wall (±60 body lengths, heading reflected at the edge) and on-demand predator launches. With the CLI defaults the wall is off, so CLI behaviour and existing measurements are unchanged.

**`serve.py`, standard library only.**
- **Server:** a `ThreadingHTTPServer` bound to `127.0.0.1` (default port 8765, `--port` to change). A sim thread steps `Sim` every 15 ms using the same deadline loop as the CLI.
- **Endpoints:**

  | Endpoint | Behaviour |
  |---|---|
  | `GET /` | the page (`web/index.html`) |
  | `GET /stream` | Server-Sent Events, one JSON snapshot every other tick (~33 Hz): `{t, x, y, heading, bump[16], probs{4}, behaviour, jump, odor, home{x,y}, loom, tick_p50, tick_p99}` |
  | `POST /goal` | `{text}` → interpretation JSON `{goal, source, ignored, ms, note?}`; interpretation runs in a worker thread; the goal applies when ready |
  | `POST /predator`, `POST /reset` | 204 |

- **Trust boundary:**
  - Request bodies over 2 KB get 413, and goal text over 200 characters gets 400.
  - JSON parse errors get 400.
  - Static serving is limited to the one page and its local assets, so there's no path traversal.
  - LLM keys stay in the server's environment and never reach the browser.

**`web/index.html`: one file, vanilla JS and canvas, no build step.** Before writing it, load the `design-engineering` skill; reuse the video's palette (#0d0f0c / #ece8dc / #f0a03c / #e2513b) and Cascadia Code.
- **Arena:** the banana and geosmin odor halos, home, the fly as an oriented marker with a fading trail, the predator loom as an expanding ring, and the arena wall.
- **Goal panel:**
  - a text box
  - preset chips: "find the banana", "avoid the smell and head north", "go home", "rest"
  - an interpretation card: the structured goal, source badge, ignored words and interpretation time
- **Brain panel:** the live 16-wedge compass from `bump`, behaviour-probability bars, and a jump indicator.
- **Footer:** live tick p50/p99.
- **Controls:** Predator, Reset.
- **Reconnects:** the SSE stream reconnects on drop (browser default), with a visible "disconnected" state.

## 4. Testing and measurement

**Self-checks:** each module's `__main__`, one runnable check per module, no test framework.

| Module | Proves |
|---|---|
| `goals.py` | about 25 phrase → expected-goal cases (synonyms, negation, clause splitting, headings, rest, ignored words, invalid combinations); the LLM path against a local stub server (valid accepted; malformed, out-of-vocabulary and invalid rejected with partial-parse fallback) |
| `fruit_fly_circuits.py` | the default goal keeps all existing checks passing; odor sign −1 turns down the gradient; a north heading goal steers toward +y. Checked on CPU and CUDA graph. |
| `system1_engine.py` | the priority table, including "threat beats every goal" and "empty goal = today's rules" |
| `serve.py` | in-process server on port 0: `GET /` 200, three SSE snapshots, `POST /goal` interpretation, oversize body 413, overlong text 400 |

**`eval_goals.py`** produces README section 3e.
1. **Interpretation accuracy:**
   - about 30 labelled phrases, plain to messy
   - exact-match rate for parser-only vs parser + LLM (local Qwen3-4B via the `llm` env config)
   - LLM interpretation latency (p50/p95)
2. **Goal following.** Each preset runs headless for 2,000 ticks from a fixed seed, against the no-goal baseline:

   | Goal | Metric |
   |---|---|
   | seek banana | reached? ticks to reach |
   | avoid geosmin | closest approach; ever touched? |
   | head north | angle between net displacement and north |
   | go home | final distance |
   | rest | share of ticks in IDLE |

3. **Serving cost:** tick p50/p99 with the server and one SSE client connected, against the CLI run.

## Error handling summary

| Failure | Behaviour |
|---|---|
| LLM unconfigured | parser result used; the page shows the ignored words |
| LLM timeout, HTTP error, bad or invalid JSON | parser partial result, with `note` saying why |
| Goal text empty | `DEFAULT_GOAL` |
| Oversize or malformed request | 413 / 400, simulation unaffected |
| SSE client disconnects | its generator ends; the simulation continues |
| System 1 backend failure | unchanged: the existing fallback to rules |

## Known limitations (to record in code and README)

- With both odors present, steering follows the odor the mushroom body currently identifies, so the fly can wobble in the overlap zone. The fix (a separate left/right comparison per odor) waits until `eval_goals.py` shows it matters.
- Headings are map-relative (landmark frame), not egocentric.
- Interpretation is limited to the vocabulary above; anything outside it is reported, never guessed.

## Deliverables

- New files:
  - `goals.py`
  - `serve.py`
  - `web/index.html`
  - `eval_goals.py`
- Changed: `fruit_fly_circuits.py` (`set_goal`, buffers), `system1_engine.py` (goal fields and rules), `agent_loop.py` (`Sim` refactor, wall, on-demand predator).
- README:
  - a "Try it" quick start
  - section 3e
  - the inline clip of the page (recorded after implementation, uploaded like the brag video)
- `docs/VERIFY.md`, `docs/STATE.md`, `docs/DECISIONS.md` updated.
