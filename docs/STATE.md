# Project State

**Last updated:** 2026-10-05 by claude-code

## Where things stand

Public at https://github.com/JamesKevinJones/flyagent (`main`). All modules pass their self-checks on
CPU and CUDA. Defaults follow the README measurements: System 1 = `rules`, circuits on the CPU,
landmark-corrected compass with a simulated 2°/s gyro bias. Opt-in System 1 backends: `laya`
(installed in `.deps/`), `http` (Jev / laya-serve), `llm` (any OpenAI-compatible key, Gemini via
`GEMINI_API_KEY` alone), `synthetic`. Laya lost every policy check to the rule table (README 3b);
a local Qwen3-4B through `llm` was far more accurate than Laya but 1.7–5 s per decision (README 3d).
The README opens with a 21 s brag video, served inline from a GitHub user-attachments URL, with the
source file in `docs/media/brag.mp4`.

## In progress

- Nothing half-done.

## The exact next step

Set the CI secret so the security-review workflow can run on pull requests (Kevin runs this; never
paste the key into chat): `gh secret set CLAUDE_API_KEY --repo JamesKevinJones/flyagent`.
After that, run the hosted-key check: `python eval_system1.py llm --n 120` with `GEMINI_API_KEY` set.

## Open questions

- Is free-text state (goals, operator instructions) planned? That is the only case where Laya or an
  LLM beats the table. If so, precompute their answers per state (finite space) or keep the GPU warm.
- Jev live test: needs a Cloudflare account and token, set in the environment.
- Gemini / OpenAI / Anthropic through `llm`: only the self-check stub and local Ollama have run.

## Known traps

- Single timing runs vary on Windows (one rules/CPU run had 173 overruns, the next two 5 and 7).
  Judge tick timing over several runs.
- With `rules` (no GPU load), circuits on the CPU beat CUDA because the idle GPU wakes every tick.
- Anything that imports `laya` needs `PYTHONPATH=.deps`.
- Laya's default load keeps FP32 weights (1.6 GB); `laya_backend` casts to bf16 *before* the VRAM cap.
- After about 5 s idle the dGPU sits at P8, and the next model call costs 300+ ms.
- Ollama's first request loads the model and can exceed `LLM_TIMEOUT` (5 s); that call falls back to rules.
- `asyncio.sleep` on Windows overshoots by up to 13 ms. Don't put the tick back on asyncio.
- `torch.cuda.mem_get_info` under WDDM doesn't show other processes' contexts (reads 0 MB).
- No `torch` in WSL here; no Triton/TileLang on Windows, so Laya runs eager only.
- The video project lives in `brag-output/` (gitignored, local only). Re-rendering needs it; the
  README video is the uploaded `user-attachments` copy, so a new render means a new upload and URL.
- Inline README videos come only from GitHub's web upload. `gh` and `git` can't create that URL.
