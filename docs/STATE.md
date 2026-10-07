# Project State

**Last updated:** 2026-10-07 by claude-code

## Where things stand

Public at https://github.com/JamesKevinJones/flyagent (`main`). All modules pass their self-checks on
CPU and CUDA. Defaults follow the README measurements: System 1 = `rules`, circuits on the CPU,
landmark-corrected compass with a simulated 2°/s gyro bias. Opt-in System 1 backends: `laya`
(installed in `.deps/`), `http` (Jev / laya-serve), `llm` (any OpenAI-compatible key, Gemini via
`GEMINI_API_KEY` alone), `synthetic`. The README opens with a 21 s brag video and an inline clip of the
goal page.

**Free-text goals (sub-project 1 of 3)** are on `main`: `goals.py`, `Sim.set_goal`, `serve.py` +
`web/index.html`, `eval_goals.py` (README 3e).

**Precompiled System 1 (sub-project 2 of 3)** is merged into `main` (2026-10-06):
- A model backend's answers for all 1,938 states are compiled into `tables/<backend>-<slug>-<hash8>.jsonl`.
  It's filled in the background, threat states first; the tick only looks answers up, and rules cover misses.
- Committed tables: Laya (136 s to compile) and Qwen3-4B (57 min).
- Measured results (README 3f):
  - Decisions take 1–3 µs, against 44–159 ms for live Laya and 1.4 s for live Qwen, with the same choices.
  - Laya's table latches FLEE in 7 ticks, like rules.
- A fresh-reviewer pass found 4 Important issues, all fixed with tests. The 10 deferred minors are listed below.
- Spec and plan: `docs/superpowers/specs/2026-10-06-precompiled-system1-design.md`, `docs/superpowers/plans/2026-10-06-precompiled-system1.md`.

**Real wiring (sub-project 3 of 3)** is merged and pushed (2026-10-07, `5c65705`), final-reviewed and fixed. The
security review of the pushed diff found nothing:
- `--wiring hemibrain` runs the mushroom body on Janelia hemibrain v1.2 wiring (CC BY) from the committed 96 KB
  `data/hemibrain_mb_cx.npz`; `hemibrain.py` rebuilds it.
- The real MB has 63 PN types and 1,927 KCs, with per-KC normalised input. Valence comes from the 44 right-side
  MBONs, count-weighted, with signs derived from their dopamine input. Learning and odor coding are now close to
  synthetic.
- Both connectome compasses failed and are reported in README 3g: the per-neuron model maxes out at about 60°/s, and
  the derived-kernel ring has a dead zone below 0.012 rad/tick. The compass stays the synthetic ring (Kevin's call,
  2026-10-07). The default wiring stays synthetic.
- Spec (with its 2026-10-07 amendments) and plan: `docs/superpowers/specs/2026-10-06-hemibrain-wiring-design.md`,
  `docs/superpowers/plans/2026-10-06-hemibrain-wiring.md`.

## In progress

- Nothing half-done.

## The exact next step

All three sub-projects are done and pushed. Two steps remain:
1. Kevin sets the `CLAUDE_API_KEY` CI secret (`gh secret set CLAUDE_API_KEY --repo JamesKevinJones/flyagent`), so the
   security workflow can run.
2. Optionally, pick a follow-up. None is planned.
   - **Faster per-neuron compass:** a shorter time constant, no rate clamp, or the ring neurons' input. The searched
     model maxes out at about 0.016 rad/tick.
   - **The other hemisphere.**
   - **FlyWire,** if its CC BY-NC licence suits.
   - **README 3g clip:** a recording of `serve.py --wiring hemibrain`.
   - **The deferred minors below.**

Deferred minors from the sub-project 3 review:
- `fetch` has no sha256 pin, and download and extract aren't atomic.
- `hemibrain.py`'s `DATA_PATH` is relative to the working directory.
- `vram_mb()` misses the KC→MBON tensors.
- `HemibrainMB` ignores `FlyBrain.kc_novelty_decay`.
- The angle-map candidate search isn't committed.
- `eval_connectome` reports tick period, not tick compute.
- `eval_connectome` has no `--wiring` flag; it always runs both.

Deferred minors from the sub-project 2 review:
- `close()` blocks on an in-flight chunk (up to about 160 s for an LLM) and discards it.
- `missing()` and `append_table` cost about 6–10 ms on one tick per chunk; caching `fill_order()` would cut most of it.
- The loader crashes on a non-UTF-8 table file.
- A crash mid-write glues the next line on, so one state is re-asked forever.
- The page says "unavailable, rules" while a partial table still decides.
- A newly compiled answer for the current state isn't applied until the worded state changes.
- `serve.py` and `--compile` on the same file can interleave lines and double the work.
- `--compile`: `rules` or a missing name gives a traceback, a failed compile exits 0, and "threat states done" is wrong on resume.
- Stale text: the `--backends` help says "fallback chain", and a `describe()` comment mentions the removed decision cache.
- `latch_ticks` counts a latch on the loom's last tick as "never".

Deferred minors from the sub-project 1 review:
- A malformed Content-Length isn't rejected cleanly.
- An overlong goal can cancel one still being interpreted.
- The sim thread has no crash guard.
- LLM reply validation is stricter than needed.
- Provider error text can reach the page.
- `World` only works via `Sim`.

## Open questions

- LLM goal merging: keep the parser's items and let the LLM only add? That would fix the paraphrase
  regression in README 3e (70% → 50% with Qwen3-4B).
- Jev live test: needs a Cloudflare account and token, set in the environment.
- Gemini / OpenAI / Anthropic through `llm`: only the self-check stub and local Ollama have run.

## Known traps

- Single timing runs vary on Windows (one rules/CPU run had 173 overruns, the next two 5 and 7); live Laya
  p50 was 159, 96 and 44 ms in three back-to-back runs. Judge timing over several runs.
- With `rules` (no GPU load), circuits on the CPU beat CUDA because the idle GPU wakes every tick.
- `.venv` has CPU-only torch: `--device cuda`, Laya and `synthetic` need the ComfyUI env (CUDA torch). Plain `python` is the system 3.14 with no torch.
- Anything that imports `laya` needs `PYTHONPATH=.deps`.
- Laya's default load keeps FP32 weights (1.6 GB); `laya_backend` casts to bf16 *before* the VRAM cap.
- After about 5 s idle the dGPU sits at P8, and the next model call costs 300+ ms.
- Ollama's first request loads the model and can exceed `LLM_TIMEOUT` (5 s, live goal interpretation). The
  table filler uses `LLM_FILL_TIMEOUT` (60 s) and `JEV_FILL_TIMEOUT` (10 s) instead.
- A table's filename hash includes `LLM_BASE_URL`, so `localhost` and `127.0.0.1` are different tables.
- A complete table means no GPU load: to measure the tick under model load, pass `--tables-dir` with an empty directory.
- `asyncio.sleep` on Windows overshoots by up to 13 ms. Don't put the tick back on asyncio.
- `torch.cuda.mem_get_info` under WDDM doesn't show other processes' contexts (reads 0 MB).
- No `torch` in WSL here; no Triton/TileLang on Windows, so Laya runs eager only.
- The video project lives in `brag-output/` (gitignored, local only). Re-rendering needs it; the
  README video is the uploaded `user-attachments` copy, so a new render means a new upload and URL.
- Inline README videos come only from GitHub's web upload. `gh` and `git` can't create that URL.
