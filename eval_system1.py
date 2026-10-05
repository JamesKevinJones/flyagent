"""Compare a System-1 backend against the rule table over every state `describe()` can emit.

    PYTHONPATH=.deps python eval_system1.py            # local Laya vs rules, all states
    python eval_system1.py llm --n 120                 # any OpenAI-compatible LLM (env-configured), random sample

Ground truth exists only where the intended policy is unambiguous (the `EXPECT` cases below);
everywhere else the report is plain agreement with the rule table, which is a baseline, not truth.
"""
import argparse
import itertools
import random
import time
from collections import Counter

from fruit_fly_circuits import BEHAVIOURS
from system1_engine import BACKENDS, QUESTIONS, rules_backend

THREATS = ("none", "approaching", "imminent")
ODORS = [("none", "n/a", "neutral")] + list(itertools.product(("banana", "geosmin", "unknown"), ("new", "familiar"),
                                                              ("rewarded", "punished", "neutral")))
HOMES = ["here"] + [f"{d}, {r}" for d in ("ahead", "ahead-left", "left", "behind-left", "behind", "behind-right",
                                         "right", "ahead-right") for r in ("near", "far")]


def all_states():
    for threat, (odor, fam, mem), home, moving in itertools.product(THREATS, ODORS, HOMES, ("yes", "no")):
        yield {"threat": threat, "odor": odor, "odor_familiarity": fam, "odor_memory": mem, "home": home,
               "moving": moving}


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", default="laya", choices=sorted(set(BACKENDS) - {"rules", "synthetic"}))
    ap.add_argument("--n", type=int, default=0, help="random sample of states (0 = all 1,938)")
    args = ap.parse_args()
    states = list(all_states())
    if args.n:
        states = random.Random(0).sample(states, args.n)
    ref = backend_decisions(states)
    report("rules", states, ref, None)
    ds = laya_decisions(states) if args.backend == "laya" else backend_decisions(states, BACKENDS[args.backend]())
    report(args.backend, states, ds, ref)
