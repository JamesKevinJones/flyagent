# flyagent

A hybrid agent for one laptop (Acer Nitro V 15: i5-13420H, RTX 4050 6 GB, 16 GB RAM). A
*Drosophila* circuit model (central complex ring attractor + path integration, mushroom-body
sparse expansion, VNC CPG + giant-fiber escape) runs at a fixed 15 ms tick. A typed System-1
decision model (Laya local, TypeSafe Jev hosted, rules fallback) chooses behaviour
asynchronously in its own process. README.md holds the measured tradeoff matrix and tuning guide.

## Stack

- Python 3.12 (measured), torch 2.14+cu130, numpy, psutil; transformers 5.x for the synthetic backend
- Optional: `laya` 0.3.27 (PyPI, Apache-2.0) for the local backend
- Optional: any OpenAI-compatible LLM for the `llm` backend, configured by env (`GEMINI_API_KEY`, or `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`). Never commit keys
- Own env: `.venv` (Python 3.12, CPU torch; `uv venv --python 3.12 .venv`, then CPU torch from download.pytorch.org/whl/cpu, then `requirements.txt`). GPU work (`--device cuda`, Laya, synthetic) still needs a CUDA torch, e.g. `Kevin codes\ComfyUI\.venv`

## Layout

- `fruit_fly_circuits.py`: `FlyBrain`. Fixed-shape tensor program; `IN_*` / `OUT_*` index the pinned I/O vectors
- `system1_engine.py`: `describe()` (circuit state to words), backends, worker-process entry points
- `agent_loop.py`: `Sim` (one tick of work, goal compilation), toy `World`, the CLI's deadline loop, pinning, stats
- `goals.py`: `Goal`, the parser, LLM interpretation (`interpret`, `make_llm`)
- `serve.py` + `web/index.html`: the local goal page (stdlib server, SSE, one static file)
- `eval_goals.py`, `eval_system1.py`: measurements behind README 3b and 3e

## Rules

1. `FlyBrain._step()` must stay CUDA-graph-capturable: preallocated buffers, in-place writes, no host syncs, no data-dependent Python branching.
2. Nothing numeric crosses into System 1: extend `describe()` with words or bins, never raw phases or KC indices.
3. The tick never blocks on System 1. Reflexes live in the circuits, and System 1 only modulates them.
4. Any latency claim in README.md must come from a measurement on this machine, or be labelled *published* or *estimate*.
5. Each module's `__main__` self-check must pass (docs/VERIFY.md).

## Read these too

- `docs/STATE.md`: where we stopped, what's next
- `docs/DECISIONS.md`: why things are the way they are
- `docs/VERIFY.md`: how to prove a change works
