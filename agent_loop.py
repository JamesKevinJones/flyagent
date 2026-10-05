"""Closed-loop agent: world -> fly circuits (every 15 ms tick) -> System 1 (async, own process) -> VNC.

The tick never waits on System 1. Decisions are latched into the brain's input vector when they
arrive; between arrivals the VNC keeps executing the last blend, and the giant-fiber escape runs
inside the tick regardless.

Deliberately not asyncio: asyncio's timer on Windows overshoots a 13.5 ms sleep by up to 13 ms (p99,
measured), which alone breaks the budget. The tick is a deadline loop (high-resolution sleep, then
spin the last 1.5 ms); System 1 is still asynchronous, as a future from its own process.

    python agent_loop.py --tick-cpus 2,3 --system1-cpus 4-7                      # rules (default)
    python agent_loop.py --backends synthetic,rules --ticks 2000                 # Laya-shaped GPU load
    PYTHONPATH=.deps python agent_loop.py --backends laya,rules                  # real Laya, opt-in
    python agent_loop.py --device cuda --n-kc 50000                              # big MB: GPU earns its keep
"""
import argparse
import gc
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

import fruit_fly_circuits as fc
from goals import DEFAULT_GOAL, HEADINGS
from system1_engine import describe, worker_decide, worker_init


# ------------------------------------------------------------------ CPU placement
def parse_cpus(spec):
    cpus = set()
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return sorted(cpus)


def p_cores():
    """Logical CPUs on Performance cores. Native Linux on a hybrid Intel CPU exposes them in sysfs;
    WSL2 and Windows do not (WSL2 vCPUs float across P and E cores), so pass --tick-cpus there."""
    try:
        with open("/sys/devices/cpu_core/cpus") as f:
            return parse_cpus(f.read().strip())
    except OSError:
        return None


def pin(cpus):
    if not cpus:
        return
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    else:
        import psutil
        psutil.Process().cpu_affinity(cpus)


def raise_priority():
    """HIGH_PRIORITY_CLASS on Windows (no admin needed); SCHED_FIFO on Linux (needs CAP_SYS_NICE)."""
    try:
        if sys.platform == "win32":
            import psutil
            psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
        return True
    except (OSError, PermissionError):
        return False


def _system1_process(backends, cpus):
    pin(cpus)
    worker_init(backends)


# ------------------------------------------------------------------ toy world (sensors + physics)
WALL = 60.0                                           # arena half-width when the wall is on (body lengths)
class World:
    def __init__(self, n_pn, gyro_bias=0.0, landmark_gain=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.gyro_bias = gyro_bias          # rad/tick the angular-velocity sensor reads too high
        self.landmark_gain = landmark_gain  # 0 = no landmark (e.g. dark or featureless scene)
        self.rng = rng
        self.sources = {"banana": (np.array([30.0, 10.0]), rng.random(n_pn), +1.0),   # food: reward on contact
                        "geosmin": (np.array([-20.0, 25.0]), rng.random(n_pn), -1.0)}  # aversive: punish
        self.pos, self.heading = np.zeros(2), 0.0
        self.loom, self.loom_bearing = 0.0, 0.0
        self.rewarded = set()

    def odor_at(self, p):
        total = 0.0
        for _, (src, pattern, _) in self.sources.items():
            total = total + pattern * math.exp(-np.sum((p - src) ** 2) / 400.0)
        return total

    def launch_predator(self, tick, bearing=None):
        """A predator looms for 30 ticks from `tick`; bearing is egocentric (random if None)."""
        self.loom_start = tick
        self.loom_bearing = float(self.rng.uniform(-math.pi, math.pi)) if bearing is None else bearing

    def sense(self, tick, inp, fwd, turn):
        # physics: the VNC's last command moves the body; that self-motion is what the CX integrates
        turn = float(np.clip(turn, -0.35, 0.35))
        self.heading += turn
        self.pos += fwd * np.array([math.cos(self.heading), math.sin(self.heading)])
        if self.wall:                                   # clamp, don't reflect: a reflection is a turn the
            np.clip(self.pos, -WALL, WALL, out=self.pos)  # compass never senses as self-motion
        since = tick - self.loom_start
        self.loom = min(1.0, since / 20) if 0 <= since < 30 else 0.0
        dopamine = 0.0
        for name, (src, _, value) in self.sources.items():
            if np.linalg.norm(self.pos - src) < 3 and name not in self.rewarded:
                dopamine = value
                self.rewarded.add(name)
        left = self.pos + 0.3 * np.array([math.cos(self.heading + 0.8), math.sin(self.heading + 0.8)])
        right = self.pos + 0.3 * np.array([math.cos(self.heading - 0.8), math.sin(self.heading - 0.8)])
        odor = self.odor_at(self.pos)
        inp[fc.IN_ANGVEL] = turn + self.gyro_bias
        inp[fc.IN_LANDMARK_HEADING] = math.remainder(self.heading, 2 * math.pi)   # e.g. the sun or a skyline
        inp[fc.IN_LANDMARK_GAIN] = self.landmark_gain
        inp[fc.IN_SPEED] = fwd
        inp[fc.IN_LOOM] = self.loom
        inp[fc.IN_LOOM_BEARING] = self.loom_bearing
        inp[fc.IN_ODOR_LR] = float(np.sum(self.odor_at(left) - self.odor_at(right))) * 5
        inp[fc.IN_DOPAMINE] = dopamine
        inp[fc.IN_ODOR:] = torch.from_numpy(odor.astype(np.float32))


# ------------------------------------------------------------------ one simulation, any front end
class _Done:
    """A completed future: in-process rules answer instantly but are applied on the next step, like the
    paced CLI's process pool, so headless runs reproduce its behaviour exactly."""
    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


GOAL_KEYS = ("goal_seek", "goal_avoid", "goal_heading", "goal_rest")


class Sim:
    """World + FlyBrain + System 1 + the current goal. step() is one tick of work and returns a snapshot;
    pacing, CPU pinning and stats belong to the caller (the CLI below, or serve.py)."""

    def __init__(self, device="cpu", n_kc=2000, backends=("rules",), system1_cpus=(), gyro_bias=0.0005,
                 landmark_gain=0.02, wall=False):
        self.device, self.n_kc = device, n_kc
        self.gyro_bias, self.landmark_gain, self.wall = gyro_bias, landmark_gain, wall
        self.backends = tuple(backends)
        if self.backends == ("rules",):
            worker_init(self.backends)
            self.pool = None
        else:
            self.pool = ProcessPoolExecutor(1, initializer=_system1_process, initargs=(self.backends, system1_cpus))
        self.goal = DEFAULT_GOAL
        self.tick_stats = (0.0, 0.0)                    # (p50, p99) tick period, written by the pacing loop
        self.reset()
        self._submit(describe(self.brain.out, 0.0, self.names)).result()   # load models before the clock starts

    def reset(self):
        self.brain = fc.FlyBrain(n_kc=self.n_kc, device=self.device)
        self.world = World(self.brain.n_pn, self.gyro_bias, self.landmark_gain)
        self.world.wall, self.world.loom_start = self.wall, -10**9
        self.names = {}
        for i, (name, (_, pattern, _)) in enumerate(self.world.sources.items()):   # innate odor library
            self.brain.inp.zero_()
            self.brain.inp[fc.IN_ODOR:] = torch.from_numpy(pattern.astype(np.float32))
            self.brain.tick()
            self.brain.remember_odor(i)
            self.names[i] = name
        self.brain.inp.zero_()
        self.brain.inp[fc.IN_P_BEHAVIOUR + 3] = 1.0      # start IDLE until System 1 speaks
        self.graph = self.brain.capture()
        self.tick, self.fwd, self.turn = 0, 0.0, 0.0
        self.pending, self.pending_state, self.last_key, self.t_sent = None, None, None, 0.0
        self.s1_ms, self.s1_age, self.backends_used = [], [], {}
        self.set_goal(self.goal)

    def set_goal(self, goal):
        """Compile a goal into brain buffers; the goal fields in the worded state re-key the decisions."""
        signs = [1.0] * self.brain.odor_tags.shape[0]
        if goal.seek or goal.avoid or goal.heading:     # any directive: odors it does not name are ignored
            for i, name in self.names.items():
                signs[i] = 1.0 if name in goal.seek else -1.0 if name in goal.avoid else 0.0
        heading = HEADINGS.get(goal.heading, 0.0)
        self.brain.set_goal(signs, heading, 0.4 if goal.heading else 0.0, -1.0 if "home" in goal.avoid else 1.0)
        self.goal = goal

    def launch_predator(self, bearing=None):
        self.world.launch_predator(self.tick, bearing)

    def _submit(self, state):
        return self.pool.submit(worker_decide, state) if self.pool else _Done(worker_decide(state))

    def step(self):
        brain, world = self.brain, self.world
        world.sense(self.tick, brain.inp, self.fwd, self.turn)
        out = brain.tick()
        self.fwd, self.turn = float(out[fc.OUT_FWD]), float(out[fc.OUT_TURN])
        state = describe(out, world.loom, self.names, self.goal)
        key = tuple(state.values())
        if self.pending is not None and self.pending.done():
            d, asked = self.pending.result(), self.pending_state
            self.pending = None
            if all(asked[k] == state[k] for k in GOAL_KEYS):   # answers for an old goal are discarded
                brain.inp[fc.IN_P_BEHAVIOUR:fc.IN_P_BEHAVIOUR + 4] = torch.tensor(d.probs)
                brain.inp[fc.IN_URGENCY] = d.urgency
                brain.inp[fc.IN_P_JUMP] = d.p_jump
                self.s1_ms.append(d.ms)
                self.s1_age.append((time.perf_counter() - self.t_sent) * 1e3)
                self.backends_used[d.backend] = self.backends_used.get(d.backend, 0) + 1
            else:
                self.last_key = None                    # re-ask under the current goal
        if self.pending is None and key != self.last_key:   # only ask System 1 when the worded state changes
            self.pending, self.pending_state = self._submit(state), state
            self.t_sent = time.perf_counter()
            self.last_key = key
        probs = brain.inp[fc.IN_P_BEHAVIOUR:fc.IN_P_BEHAVIOUR + 4].tolist()
        self.tick += 1
        return {
            "t": round(self.tick * 0.015, 3), "x": float(world.pos[0]), "y": float(world.pos[1]),
            "heading": math.remainder(world.heading, 2 * math.pi), "brain_heading": float(out[fc.OUT_HEADING]),
            "bump": [round(v, 4) for v in brain.bump.tolist()],
            "probs": dict(zip(fc.BEHAVIOURS, probs)), "behaviour": fc.BEHAVIOURS[probs.index(max(probs))],
            "jump": bool(out[fc.OUT_JUMP]), "odor": state["odor"], "home": {"x": 0.0, "y": 0.0},
            "loom": world.loom, "sources": {n: src.tolist() for n, (src, _, _) in world.sources.items()},
            "wall": WALL if world.wall else None, "tick_p50": self.tick_stats[0], "tick_p99": self.tick_stats[1],
        }

    def close(self):
        if self.pool:
            self.pool.shutdown(cancel_futures=True)


# ------------------------------------------------------------------ main loop
def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def run(args):
    if sys.platform == "win32":                       # default Windows timer is 15.6 ms: useless for a 15 ms tick
        import ctypes
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
    tick_cpus = parse_cpus(args.tick_cpus) if args.tick_cpus else (p_cores() or [])[:2]
    pin(tick_cpus)
    realtime = raise_priority()
    torch.set_num_threads(1)                          # tiny ops; thread pools only add wake-up jitter

    sim = Sim(args.device, args.n_kc, tuple(args.backends.split(",")), parse_cpus(args.system1_cpus),
              args.gyro_bias, args.landmark_gain)
    gc.collect()
    gc.freeze()                                       # long-lived objects out of the collector's way
    period = args.tick_ms / 1000
    compute_ms, period_ms, heading_err = [], [], []
    behaviour_ticks = dict.fromkeys(fc.BEHAVIOURS, 0)
    jumps, overruns = 0, 0
    t_prev = next_t = time.perf_counter()

    for tick in range(args.ticks):
        t0 = time.perf_counter()
        if tick == args.ticks // 2:                    # a predator looms for 30 ticks in the middle of the run
            sim.launch_predator()
        snap = sim.step()
        jumps += snap["jump"]
        heading_err.append(abs(math.remainder(snap["brain_heading"] - snap["heading"], 2 * math.pi)))
        behaviour_ticks[snap["behaviour"]] += 1

        t1 = time.perf_counter()
        compute_ms.append((t1 - t0) * 1e3)
        next_t += period
        if next_t - t1 > 0.0015:
            time.sleep(next_t - t1 - 0.0015)            # high-res timer on Windows (3.11+), then spin
        while time.perf_counter() < next_t:
            pass
        now = time.perf_counter()
        if tick:
            period_ms.append((now - t_prev) * 1e3)
            overruns += period_ms[-1] > args.tick_ms + 1.0
        if now - next_t > period:                       # fell a whole tick behind: don't burst to catch up
            next_t = now
        t_prev = now

    sim.close()
    vram = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.0
    s1_ms, s1_age = sim.s1_ms, sim.s1_age
    print(f"device={args.device} graph={sim.graph} n_kc={args.n_kc} tick_cpus={tick_cpus or 'unpinned'} "
          f"high_priority={realtime} "
          f"backends={args.backends}")
    print(f"tick compute   p50 {pct(compute_ms, 50):.3f}  p99 {pct(compute_ms, 99):.3f}  max {max(compute_ms):.3f} ms")
    print(f"tick period    p50 {pct(period_ms, 50):.3f}  p99 {pct(period_ms, 99):.3f}  max {max(period_ms):.3f} ms"
          f"  overruns(>{args.tick_ms + 1:.0f} ms) {overruns}/{len(period_ms)}")
    print(f"system1        calls {len(s1_ms)} {sim.backends_used}  model p50 {pct(s1_ms, 50):.2f}  "
          f"p99 {pct(s1_ms, 99):.2f} ms  sent->latched p50 {pct(s1_age, 50):.2f}  p99 {pct(s1_age, 99):.2f}  max {max(s1_age, default=0):.2f} ms")
    print(f"behaviour      ticks {behaviour_ticks}  "
          f"jumps {jumps}  rewards {sorted(sim.world.rewarded)}  brain VRAM peak {vram:.1f} MB")
    print(f"compass        gyro bias {args.gyro_bias} rad/tick, landmark gain {args.landmark_gain}  "
          f"heading error p50 {pct(heading_err, 50):.3f}  max {max(heading_err):.3f} rad")
    return pct(period_ms, 99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu beats cuda at 2k KCs unless the GPU is kept busy anyway (README 3b, docs/STATE.md)")
    ap.add_argument("--n-kc", type=int, default=2000)
    ap.add_argument("--gyro-bias", type=float, default=0.0005, help="rad/tick (~2 deg/s), 0 for a perfect gyro")
    ap.add_argument("--landmark-gain", type=float, default=0.02, help="0 hides the landmark")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--tick-ms", type=float, default=15.0)
    ap.add_argument("--backends", default="rules",
                    help="fallback chain, e.g. laya,http,rules; see system1_engine.py and README 3b for why rules is default")
    ap.add_argument("--tick-cpus", default="", help="e.g. 2,3 (one P-core, both hyperthreads)")
    ap.add_argument("--system1-cpus", default="", help="e.g. 4-7 (other P-cores)")
    run(ap.parse_args())


def selfcheck():
    from goals import parse
    oracle = {"FORAGE": 1847, "FLEE": 25, "ORIENT": 124, "IDLE": 4}

    sim = Sim()                                            # default_matches_cli
    counts = dict.fromkeys(fc.BEHAVIOURS, 0)
    for t in range(2000):
        if t == 1000:
            sim.launch_predator()
        counts[sim.step()["behaviour"]] += 1
    assert counts == oracle and sorted(sim.world.rewarded) == ["banana", "geosmin"], (counts, sim.world.rewarded)

    sim = Sim()                                            # stale_decision_discarded
    sim.set_goal(parse("find the banana")[0])
    first = sim.step()                                     # asks System 1 under the banana goal
    sim.set_goal(parse("rest")[0])
    assert sim.step()["behaviour"] != "FORAGE"             # the banana answer must not latch
    assert sim.step()["behaviour"] == "IDLE"
    assert first["behaviour"] == "IDLE"                    # nothing latched before the first answer

    sim = Sim(wall=True)                                   # wall_holds
    sim.set_goal(parse("head north")[0])
    for _ in range(3000):
        snap = sim.step()
        assert math.isfinite(snap["x"]) and math.isfinite(snap["y"]) and max(abs(snap["x"]), abs(snap["y"])) <= 60, snap
    assert snap["y"] > 50, snap["y"]                       # it actually went north

    sim = Sim()                                            # seek_banana_reaches
    sim.set_goal(parse("find the banana")[0])
    for _ in range(2000):
        sim.step()
    assert "banana" in sim.world.rewarded, sim.world.rewarded

    sim = Sim()                                            # avoid_geosmin_never_touches (the default run touches it)
    sim.set_goal(parse("avoid the smell")[0])
    for _ in range(2000):
        sim.step()
    assert "geosmin" not in sim.world.rewarded, sim.world.rewarded
    for s in (sim,):
        s.close()
    print("agent_loop self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
