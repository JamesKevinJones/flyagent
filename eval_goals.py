"""Measure free-text goals: interpretation accuracy (parser vs parser + LLM), goal following, serving cost.

    python eval_goals.py                 # all three tables; uses the `llm` env config if set
    python eval_goals.py --no-llm        # parser only
"""
import argparse
import http.client
import math
import sys
import threading
import time

import numpy as np

from goals import interpret, make_llm, parse

# (text, (seek, avoid, heading, rest)). Plain phrases are written in the parser's vocabulary; paraphrased and
# messy ones are what people actually type. Labels are what a person would mean, not what the parser can do.
PHRASES = {
    "plain": [
        ("find the banana", ({"banana"}, set(), None, False)),
        ("go home", ({"home"}, set(), None, False)),
        ("avoid the smell", (set(), {"geosmin"}, None, False)),
        ("head north", (set(), set(), "north", False)),
        ("rest", (set(), set(), None, True)),
        ("avoid the smell and head north", (set(), {"geosmin"}, "north", False)),
        ("find food but avoid the mould", ({"banana"}, {"geosmin"}, None, False)),
        ("go west", (set(), set(), "west", False)),
        ("stay away from home", (set(), {"home"}, None, False)),
        ("find the banana and go home", ({"banana", "home"}, set(), None, False)),
    ],
    "paraphrased": [
        ("grab some fruit", ({"banana"}, set(), None, False)),
        ("get back to the nest", ({"home"}, set(), None, False)),
        ("keep clear of that earthy odour", (set(), {"geosmin"}, None, False)),
        ("make your way up to the top", (set(), set(), "north", False)),
        ("take a break", (set(), set(), None, True)),
        ("steer clear of the stink and go north", (set(), {"geosmin"}, "north", False)),
        ("return to base", ({"home"}, set(), None, False)),
        ("eat something sweet", ({"banana"}, set(), None, False)),
        ("don't come home", (set(), {"home"}, None, False)),
        ("wander east", (set(), set(), "east", False)),
    ],
    "messy": [
        ("I'm starving but that mouldy stink is gross", ({"banana"}, {"geosmin"}, None, False)),
        ("ugh I'm so hungry, find me some sugar", ({"banana"}, set(), None, False)),
        ("it's late, time to head back to where you started", ({"home"}, set(), None, False)),
        ("that smell is disgusting, get as far north as you can", (set(), {"geosmin"}, "north", False)),
        ("chill out for a bit", (set(), set(), None, True)),
        ("honestly just go home, avoid the earthy stuff", ({"home"}, {"geosmin"}, None, False)),
        ("the predator scared you, get back to your nest", ({"home"}, set(), None, False)),
        ("go get the fruit but stay well away from the mould", ({"banana"}, {"geosmin"}, None, False)),
        ("please don't move", (set(), set(), None, True)),
        ("explore toward the north west and ignore smells", (set(), set(), "north-west", False)),
    ],
}


def _key(goal):
    return set(goal.seek), set(goal.avoid), goal.heading, goal.rest


def interpretation_accuracy(llm):
    """Exact-match rate per phrase group for the parser alone and for parser + LLM (when `llm` is given)."""
    out = {"parser": {}, "parser_misses": [], "llm": {}, "llm_misses": [], "llm_ms": []}
    for group, phrases in PHRASES.items():
        hits = [_key(parse(text)[0]) == want for text, want in phrases]
        out["parser"][group] = sum(hits) / len(hits)
        out["parser_misses"] += [text for (text, _), ok in zip(phrases, hits) if not ok]
        if llm is not None:
            hits = []
            for text, want in phrases:
                r = interpret(text, llm)
                hits.append(_key(r.goal) == want)
                if r.goal.source == "llm":
                    out["llm_ms"].append(r.ms)
                if not hits[-1]:
                    out["llm_misses"].append(f"{text}  ->  {r.goal.source}: {sorted(r.goal.seek)} / "
                                             f"{sorted(r.goal.avoid)} / {r.goal.heading} / rest={r.goal.rest}"
                                             + (f"  [{r.note}]" if r.note else ""))
            out["llm"][group] = sum(hits) / len(hits)
    return out


def follow(goal_text, ticks=2000, warmup=0):
    """Run a goal headless from the start position (after `warmup` ticks of default foraging) and measure it."""
    from agent_loop import Sim
    sim = Sim(wall=True)
    for _ in range(warmup):
        sim.step()
    goal = interpret(goal_text).goal if goal_text else sim.goal
    sim.set_goal(goal)
    start = np.array([sim.world.pos[0], sim.world.pos[1]])
    geosmin = sim.world.sources["geosmin"][0]
    first_banana, min_geo, idle = None, math.inf, 0
    for t in range(ticks):
        if t == ticks // 2:                              # the CLI's scripted predator, so baselines match the CLI run
            sim.launch_predator()
        snap = sim.step()
        if first_banana is None and "banana" in sim.world.rewarded:
            first_banana = t
        min_geo = min(min_geo, float(np.linalg.norm(sim.world.pos - geosmin)))
        idle += snap["behaviour"] == "IDLE"
    moved = sim.world.pos - start
    north_err = abs(math.degrees(math.remainder(math.atan2(moved[1], moved[0]) - math.pi / 2, 2 * math.pi)))
    sim.close()
    return {"reached_banana": first_banana is not None and first_banana >= 0, "ticks_to_banana": first_banana,
            "min_dist_geosmin": round(min_geo, 1), "touched_geosmin": "geosmin" in sim.world.rewarded,
            "north_error_deg": round(north_err, 1), "net_move": round(float(np.linalg.norm(moved)), 1),
            "final_home_dist": round(float(np.linalg.norm(sim.world.pos)), 1), "idle_share": round(idle / ticks, 3)}


def serving_cost(ticks=2000):
    """Tick period with the simulation thread alone vs with the server and one live SSE client."""
    from agent_loop import Sim
    from serve import SimRunner, make_server
    result = {}
    for label in ("headless", "server + 1 SSE client"):
        sim = Sim(wall=True)
        runner = SimRunner(sim)
        runner.periods = __import__("collections").deque(maxlen=ticks)
        server = None
        if label != "headless":
            server = make_server(sim, port=0, runner=runner)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            client = http.client.HTTPConnection("127.0.0.1", server.server_address[1])
            client.request("GET", "/stream")
            resp = client.getresponse()
            threading.Thread(target=lambda: [resp.fp.readline() for _ in iter(int, 1)], daemon=True).start()
        runner.start()
        while sim.tick < ticks:
            time.sleep(0.2)
        runner.stop()
        if server:
            server.shutdown()
        p = np.array(runner.periods)
        result[label] = {"p50": round(float(np.percentile(p, 50)), 3), "p99": round(float(np.percentile(p, 99)), 3),
                         "overruns_gt_16ms": int((p > 16.0).sum()), "ticks": len(p)}
        sim.close()
    return result


FOLLOW = [("(no goal: default foraging)", "", 0), ("find the banana", "find the banana", 0),
          ("avoid the smell", "avoid the smell", 0), ("head north", "head north", 0),
          ("go home (after 300 ticks of foraging)", "go home", 300), ("rest", "rest", 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    llm = None if args.no_llm else make_llm()

    acc = interpretation_accuracy(llm)
    print("== 1. Interpretation accuracy (30 labelled phrases, exact match on seek/avoid/heading/rest)")
    print(f"{'group':12s} {'parser':>8s} {'parser+LLM':>11s}")
    for g in PHRASES:
        llm_cell = f"{acc['llm'][g]:.0%}" if llm else "n/a"
        print(f"{g:12s} {acc['parser'][g]:>8.0%} {llm_cell:>11s}")
    if acc["llm_ms"]:
        print(f"LLM interpretation time: p50 {np.percentile(acc['llm_ms'], 50):.0f} ms, "
              f"p95 {np.percentile(acc['llm_ms'], 95):.0f} ms ({len(acc['llm_ms'])} calls)")
    print("parser misses:", *acc["parser_misses"], sep="\n  ")
    if llm:
        print("parser+LLM misses:", *acc["llm_misses"], sep="\n  ")

    print("\n== 2. Goal following (2,000 ticks, wall on, deterministic)")
    for label, text, warmup in FOLLOW:
        print(f"{label:40s} {follow(text, warmup=warmup)}")

    print("\n== 3. Serving cost (tick period, ms)")
    for label, r in serving_cost().items():
        print(f"{label:24s} {r}")


def selfcheck():
    acc = interpretation_accuracy(None)
    assert acc["parser"]["plain"] == 1.0, acc["parser_misses"]
    banana = follow("find the banana")
    assert banana["reached_banana"] is True, banana
    rest = follow("rest")
    assert rest["idle_share"] > 0.9, rest
    print("eval_goals self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
