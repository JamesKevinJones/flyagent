# Verification

Use `PY=.venv/Scripts/python.exe` (the project env, CPU torch). For the CUDA checks, Laya and `synthetic`, use a CUDA torch such as `CPY="C:/Users/kj638/Kevin codes/ComfyUI/.venv/Scripts/python.exe"` (sections below that need it say `$CPY`; older sections say `$PY`, so set `PY=$CPY` for those); with CPU torch, `fruit_fly_circuits.py` checks only the CPU path.

## Self-checks (must print OK)

```bash
"$PY" fruit_fly_circuits.py
"$PY" system1_engine.py
```

## Free-text goals

```bash
"$PY" goals.py                      # parser, LLM fallback, validation
"$PY" agent_loop.py --selfcheck     # Sim: matches the CLI, stale answers, wall, goals reached/avoided, rest
"$PY" serve.py --selfcheck          # server: page, stream, limits, last goal wins, client disconnect
"$PY" eval_goals.py --selfcheck     # evaluation sanity
"$PY" eval_goals.py                 # full numbers for README 3e (uses the llm env config if set)
"$PY" serve.py                      # then open http://127.0.0.1:8765
```

Regression check: `"$PY" agent_loop.py --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7` must still print
`{'FORAGE': 1847, 'FLEE': 25, 'ORIENT': 124, 'IDLE': 4}`, `jumps 22`, `rewards ['banana', 'geosmin']`.

## Precompiled System 1 tables

```bash
"$PY" system1_engine.py             # states, table file + loader, goal precedence, filler, resume, stall
"$PY" agent_loop.py --selfcheck     # also: complete table -> no worker; model unavailable -> status off, rules
"$PY" agent_loop.py --backends laya --ticks 2000 --tick-cpus 2,3   # CPU-only, from the committed table
"$PY" eval_system1.py laya --latency --skip-live                   # lookup p50 in microseconds; loom latch
PYTHONPATH=.deps "$CPY" eval_system1.py laya --latency             # + live Laya, same answers (CUDA env)
PYTHONPATH=.deps "$CPY" system1_engine.py --compile laya           # rebuild a table (fills only what's missing)
```

Expected: the CPU-only run prints `table laya 1938/1938 complete`, `by {'laya table': N}`, and doesn't
import Laya. `--latency` prints a lookup p50 of a few µs, and the table's loom latch is 7 ticks, the same as
rules. To check that tables don't change the default, re-run the regression oracle above.

## Hemibrain wiring

```bash
"$PY" hemibrain.py --selfcheck      # committed .npz: counts, MBON signs, compass angles; refuses a truncated archive
"$PY" fruit_fly_circuits.py         # both wirings; hemibrain: valence 0.043 / -0.048, odor overlap 0.046, spill 0.010, rotation gain 1.000
"$PY" agent_loop.py --selfcheck     # also: hemibrain fly reaches the banana, flees, keeps its wiring across reset
"$CPY" eval_connectome.py --runs 3  # README 3g numbers (docs/eval-connectome-2026-10-06.txt)
"$PY" serve.py --wiring hemibrain   # page shows "Wiring: Janelia hemibrain v1.2 (CC BY)"
```

`python hemibrain.py` rebuilds the file (downloads 45.9 MB once into `.deps/hemibrain/`). The synthetic regression
oracle above must not change.

## Closed loop under System-1 GPU load

```bash
"$CPY" agent_loop.py --backends synthetic,rules --tables-dir "$(mktemp -d)" --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7
```

A model only loads the GPU while its table fills, so this uses an empty tables dir: the `synthetic` filler
runs for most of the 2,000 ticks.

Pass: `tick period ... p99` at or under ~15.7 ms, and overruns around 1% or less. Behaviour should show
FORAGE dominating, FLEE for roughly 10–25 ticks around tick 1000, and jumps above 0. The `compass` line
should show a heading error max of about 0.025 rad; with `--landmark-gain 0` it should drift to about 1.0 rad. Close heavy apps first.

## System 1 comparison (needs Laya in `.deps/`; ~3 min on the RTX 4050)

```bash
PYTHONPATH=.deps "$PY" eval_system1.py
PYTHONPATH=.deps "$PY" agent_loop.py --device cpu --backends laya --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7
```

Expected right now: rules at 100% on every check. Laya threat→FLEE at about 99%, rewarded→FORAGE at about 26%.
Closed loop from Laya's committed table: ORIENT 1678 / FORAGE 296 / FLEE 25 / IDLE 1 ticks, tick period p99
about 15 ms (README 3f).

## Any LLM key (OpenAI-compatible)

```bash
LLM_TIMEOUT=60 LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M "$PY" eval_system1.py llm --n 120
```

Swap in `GEMINI_API_KEY` alone, or `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL`, for a hosted provider.
Expected with the local Qwen3-4B: threat→FLEE about 95%, choice agreement with rules about 86%.

## Known-failing

- Max tick period (27–39 ms) exceeds 15 ms under Windows; p99 doesn't. This isn't a regression.
