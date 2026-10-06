"""Compare a System-1 backend against the rule table over every state `describe()` can emit.

    PYTHONPATH=.deps python eval_system1.py            # local Laya vs rules, all states
    python eval_system1.py llm --n 120                 # any OpenAI-compatible LLM (env-configured), random sample
    PYTHONPATH=.deps python eval_system1.py laya --latency    # live model vs precompiled table vs rules (README 3f)

Ground truth exists only where the intended policy is unambiguous (the `EXPECT` cases below);
everywhere else the report is plain agreement with the rule table, which is a baseline, not truth.
"""
import argparse
import math
import random
import time
from collections import Counter

import numpy as np

from fruit_fly_circuits import BEHAVIOURS
from system1_engine import (BACKENDS, N_STATES, QUESTIONS, THREATS, all_states, base_key, load_table, lookup,
                            rules_backend, table_path)

# (name, which states, what is acceptable)
EXPECT = [
    ("threat -> FLEE", lambda s: s["threat"] != "none", lambda d: d["choice"] == "FLEE"),
    ("imminent -> P(jump) > 0.5", lambda s: s["threat"] == "imminent", lambda d: d["p_jump"] > 0.5),
    ("no threat -> P(jump) < 0.5", lambda s: s["threat"] == "none", lambda d: d["p_jump"] < 0.5),
    ("rewarded odor, safe -> FORAGE", lambda s: s["threat"] == "none" and s["odor_memory"] == "rewarded",
     lambda d: d["choice"] == "FORAGE"),
    ("punished odor, safe -> not FORAGE", lambda s: s["threat"] == "none" and s["odor_memory"] == "punished",
     lambda d: d["choice"] != "FORAGE"),
]


def laya_decisions(states, questions=QUESTIONS, batch=32):
    import laya
    agent = laya.load("convaiinnovations/laya-typed-decisions", device="cuda")
    t0 = time.perf_counter()
    out = []
    for i in range(0, len(states), batch):
        for r in agent.predict_batch(states[i:i + batch], questions, batch_size=batch):
            a = r["answers"]
            out.append({"choice": a["behaviour"]["choice"], "p": a["behaviour"]["probabilities"],
                        "urgency": a["urgency"]["score"] / 3, "p_jump": a["jump"]["noul"]})
    print(f"laya: {len(states)} states in {time.perf_counter() - t0:.1f}s (batched {batch})")
    return out


def backend_decisions(states, decide=None):
    timed = decide is not None
    decide = decide or rules_backend()
    out = []
    t0 = time.perf_counter()
    for s in states:
        d = decide(s)
        out.append({"choice": BEHAVIOURS[d.probs.index(max(d.probs))], "p": dict(zip(BEHAVIOURS, d.probs)),
                    "urgency": d.urgency, "p_jump": d.p_jump})
    if timed:
        print(f"{len(states)} states in {time.perf_counter() - t0:.1f}s")
    return out


def report(name, states, ds, ref):
    print(f"\n== {name}")
    for label, which, ok in EXPECT:
        idx = [i for i, s in enumerate(states) if which(s)]
        if not idx:
            continue
        print(f"  {label:36s} {sum(ok(ds[i]) for i in idx) / len(idx):6.1%}   (n={len(idx)})")
    for t in THREATS:
        idx = [i for i, s in enumerate(states) if s["threat"] == t]
        mean = lambda k: sum(ds[i][k] for i in idx) / len(idx)
        pf = sum(ds[i]["p"]["FLEE"] for i in idx) / len(idx)
        print(f"  threat={t:12s} mean P(FLEE) {pf:.2f}  urgency {mean('urgency'):.2f}  P(jump) {mean('p_jump'):.2f}")
    print(f"  choices: {dict(Counter(d['choice'] for d in ds))}")
    if ref is not None:
        agree = sum(d["choice"] == r["choice"] for d, r in zip(ds, ref)) / len(ds)
        print(f"  choice agreement with rules: {agree:.1%}")


def latch_ticks(sim, runs=20, cap=30):
    """Ticks from loom onset until the applied behaviour is FLEE; cap (the loom's length) = never."""
    out = []
    for k in range(runs):
        sim.reset()
        for _ in range(200):
            sim.step()
        sim.launch_predator(bearing=k * 2 * math.pi / runs)
        n = next((i for i in range(1, cap + 1) if sim.step()["behaviour"] == "FLEE"), cap)
        out.append(n)
    return out


def latency(backend, tables_dir, skip_live):
    from agent_loop import Sim
    path = table_path(backend, tables_dir)
    table = load_table(path, backend) if path else {}
    if len(table) < N_STATES:
        raise SystemExit(f"{path}: {len(table)}/{N_STATES} states; run: python system1_engine.py --compile {backend}")
    sample = random.Random(0).sample(list(all_states()), 100)
    rules = rules_backend()
    if not skip_live:
        live, ms, same, dp = BACKENDS[backend](), [], 0, 0.0
        for s in sample:
            d, t = live(s), table[base_key(s)]
            ms.append(d.ms)
            same += d.probs.index(max(d.probs)) == t.probs.index(max(t.probs))
            dp = max(dp, max(abs(a - b) for a, b in zip(d.probs, t.probs)))
        print(f"live           {backend} p50 {np.percentile(ms, 50):.1f}  p99 {np.percentile(ms, 99):.1f} ms "
              f"(warm, back-to-back, n=100)")
        print(f"same answers   choice {same}/100  max |dP| {dp:.4f}")
    us = []
    for s in sample:
        for _ in range(100):
            t0 = time.perf_counter_ns()
            lookup(s, table, rules)
            us.append((time.perf_counter_ns() - t0) / 1e3)
    print(f"lookup         p50 {np.percentile(us, 50):.2f}  p99 {np.percentile(us, 99):.2f} us (n={len(us)})")
    for name, sim in ((f"{backend} table", Sim(backends=(backend,), tables_dir=tables_dir)), ("rules", Sim())):
        ticks = latch_ticks(sim)
        sim.close()
        print(f"loom latch     {name:12s} p50 {np.percentile(ticks, 50):.0f}  max {max(ticks)} ticks "
              f"({np.percentile(ticks, 50) * 15:.0f} / {max(ticks) * 15} ms), never within the loom {ticks.count(30)}/20")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", default="laya", choices=sorted(set(BACKENDS) - {"rules", "synthetic"}))
    ap.add_argument("--n", type=int, default=0, help="random sample of states (0 = all 1,938)")
    ap.add_argument("--latency", action="store_true", help="live vs precompiled table vs rules (README 3f)")
    ap.add_argument("--skip-live", action="store_true", help="with --latency: no model calls (CPU-only check)")
    ap.add_argument("--tables-dir", default="tables")
    args = ap.parse_args()
    if args.latency:
        raise SystemExit(latency(args.backend, args.tables_dir, args.skip_live))
    states = list(all_states())
    if args.n:
        states = random.Random(0).sample(states, args.n)
    ref = backend_decisions(states)
    report("rules", states, ref, None)
    ds = laya_decisions(states) if args.backend == "laya" else backend_decisions(states, BACKENDS[args.backend]())
    report(args.backend, states, ds, ref)
