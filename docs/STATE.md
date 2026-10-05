# Project State

**Last updated:** 2026-10-05 by claude-code

## Where things stand

All modules run and pass their self-checks on CPU and CUDA. Real Laya (`laya-typed-decisions`) is
installed project-locally in `.deps/` and was compared against the rule table over all 1,938
worded states (`eval_system1.py`) and in the closed loop (README section 3b). Result: rules win on
every policy check. Laya, with explicit criteria, flees on 99.2% of threats, but with P(FLEE) of only
0.32–0.48, and it gets two-field rules wrong. In the closed loop the fly orients 99% of the time. The
best tick timing is circuits on the CPU with Laya on the GPU: p99 15.000 ms, 6/1999 overruns.

## In progress

- Nothing half-done.

## The exact next step

`rules` is now the default System 1 and the circuits default to the CPU; Laya, Jev and synthetic are opt-in via `--backends`. If Laya comes back
into use (for example when free-text state arrives): precompute its answers for all states
at worker start (batch with `predict_batch`, 60–85 s; save to a JSON file keyed by model id and
QUESTIONS), and look them up in `worker_decide`. That removes the 300 ms GPU-wake stall.

## Open questions

- Is free-text state (goals, operator instructions) planned? That is the only case where Laya
  beats a table.
- Jev live test: needs a Cloudflare account and token, which Kevin sets himself.

## Landmark input (2026-10-05)

The ring attractor takes a visual landmark cue (`IN_LANDMARK_HEADING`, `IN_LANDMARK_GAIN`; the odor slots moved
to `IN_ODOR = 14`). The world simulates a 2°/s gyro bias by default. With the landmark, heading error is bounded at 0.025 rad;
without it, it drifts to 1.0 rad over 30 s. README section 3c.

## Known traps

- Single timing runs vary on Windows. One rules/CPU run had 173 overruns; the next two had 5 and 7.
  Judge tick timing over several runs, never one.
- With `rules` (no GPU load), circuits on the CPU beat CUDA: 5 and 7 overruns vs 12 and 27, because the idle GPU wakes up every tick.
- Anything that imports `laya` needs `PYTHONPATH=.deps`.
- Laya's default load keeps FP32 weights (1.6 GB). `laya_backend` converts them to bf16 *before* applying the VRAM cap, or the load OOMs.
- After about 5 s idle the dGPU sits at P8, and the next Laya call costs 300+ ms. The decision cache makes this idle common.
- `asyncio.sleep` on Windows overshoots by up to 13 ms. Don't put the tick back on asyncio.
- `torch.cuda.mem_get_info` under WDDM doesn't show other processes' contexts (reads 0 MB).
- No `torch` in WSL here; there is no Triton/TileLang on Windows, so Laya runs eager only.
