# Verification

Use `PY="C:/Users/kj638/Kevin codes/ComfyUI/.venv/Scripts/python.exe"` (CUDA torch) or any env from requirements.txt.

## Self-checks (must print OK)

```bash
"$PY" fruit_fly_circuits.py
"$PY" system1_engine.py
```

## Closed loop under System-1 GPU load

```bash
"$PY" agent_loop.py --backends synthetic,rules --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7
```

Pass: `tick period ... p99` at or under ~15.7 ms, and overruns around 1% or less. Behaviour should show
FORAGE dominating, FLEE for roughly 10–25 ticks around tick 1000, and jumps above 0. The `compass` line
should show a heading error max of about 0.025 rad; with `--landmark-gain 0` it should drift to about 1.0 rad. Close heavy apps first.

## System 1 comparison (needs Laya in `.deps/`; ~3 min on the RTX 4050)

```bash
PYTHONPATH=.deps "$PY" eval_system1.py
PYTHONPATH=.deps "$PY" agent_loop.py --device cpu --backends laya,rules --ticks 2000 --tick-cpus 2,3 --system1-cpus 4-7
```

Expected right now: rules at 100% on every check. Laya threat→FLEE at about 99%, rewarded→FORAGE at about 26%.
Closed loop with Laya: ORIENT around 1980 ticks, tick period p99 about 15 ms.

## Known-failing

- Max tick period (27–39 ms) exceeds 15 ms under Windows; p99 doesn't. This isn't a regression.
