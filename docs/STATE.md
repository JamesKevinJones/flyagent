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

**Free-text goals (sub-project 1 of 3)** are merged into `main` (2026-10-05), after a fresh-reviewer pass whose 7 Important findings were fixed:
- `goals.py` holds the parser and the optional LLM.
- `Sim.set_goal` compiles goals into the brain buffers and goal-aware rules.
- `serve.py` + `web/index.html` serve the local page at http://127.0.0.1:8765.
- `eval_goals.py` produces the README 3e numbers.

Spec and plan are in `docs/superpowers/`. The next sub-projects are (2) fast System 1 on novel states
and (3) a full connectome.

## In progress

- Nothing half-done.

## The exact next step

Push `main` (ahead of origin; `git log origin/main..main` lists the commits): run `/security-review` on the unpushed diff first, as the
global rules require. Then record a clip of the goal page and upload it inline like the brag video
(needs Kevin's go-ahead for the Chrome upload). Still open: set the `CLAUDE_API_KEY` CI secret
(`gh secret set CLAUDE_API_KEY --repo JamesKevinJones/flyagent`). After that, brainstorm sub-project 2
(fast System 1 on novel states).

Deferred minors from the final review: a malformed Content-Length isn't rejected cleanly; an
overlong goal can cancel one still being interpreted; no crash guard in the sim thread; LLM reply
validation is stricter than needed; provider error text can reach the page; `World` only works via `Sim`.

## Open questions

- LLM goal merging: keep the parser's items and let the LLM only add? That would fix the paraphrase
  regression in README 3e (70% → 50% with Qwen3-4B).
- Jev live test: needs a Cloudflare account and token, set in the environment.
- Gemini / OpenAI / Anthropic through `llm`: only the self-check stub and local Ollama have run.

## Known traps

- Single timing runs vary on Windows (one rules/CPU run had 173 overruns, the next two 5 and 7).
  Judge tick timing over several runs.
- With `rules` (no GPU load), circuits on the CPU beat CUDA because the idle GPU wakes every tick.
- `.venv` has CPU-only torch: `--device cuda`, Laya and `synthetic` need the ComfyUI env (CUDA torch). Plain `python` is the system 3.14 with no torch.
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
