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

    def sense(self, tick, n_ticks, inp, fwd, turn):
        # physics: the VNC's last command moves the body; that self-motion is what the CX integrates
        turn = float(np.clip(turn, -0.35, 0.35))
        self.heading += turn
        self.pos += fwd * np.array([math.cos(self.heading), math.sin(self.heading)])
        # a predator looms for 30 ticks in the middle of the run
        start = n_ticks // 2
        self.loom = min(1.0, (tick - start) / 20) if start <= tick < start + 30 else 0.0
        if tick == start:
            self.loom_bearing = float(self.rng.uniform(-math.pi, math.pi))
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

    brain = fc.FlyBrain(n_kc=args.n_kc, device=args.device)
    world = World(brain.n_pn, args.gyro_bias, args.landmark_gain)
    names = {}
    for i, (name, (_, pattern, _)) in enumerate(world.sources.items()):   # innate odor library
        brain.inp.zero_()
        brain.inp[fc.IN_ODOR:] = torch.from_numpy(pattern.astype(np.float32))
        brain.tick()
        brain.remember_odor(i)
        names[i] = name
    brain.inp.zero_()
    brain.inp[fc.IN_P_BEHAVIOUR + 3] = 1.0           # start IDLE until System 1 speaks
    graph = brain.capture()

    pool = ProcessPoolExecutor(1, initializer=_system1_process,
                               initargs=(tuple(args.backends.split(",")), parse_cpus(args.system1_cpus)))
    pool.submit(worker_decide, describe(brain.out, 0.0, names)).result()   # load models before the clock starts
    gc.collect()
    gc.freeze()                                       # long-lived objects out of the collector's way
    period = args.tick_ms / 1000
    compute_ms, period_ms, s1_ms, s1_age, heading_err = [], [], [], [], []
    backends_used, behaviour_ticks = {}, np.zeros(4)
    pending, last_key, jumps, overruns = None, None, 0, 0
    fwd = turn = 0.0
    t_prev = next_t = time.perf_counter()

    for tick in range(args.ticks):
        t0 = time.perf_counter()
        world.sense(tick, args.ticks, brain.inp, fwd, turn)
        out = brain.tick()
        fwd, turn = float(out[fc.OUT_FWD]), float(out[fc.OUT_TURN])
        jumps += int(out[fc.OUT_JUMP])
        heading_err.append(abs(math.remainder(float(out[fc.OUT_HEADING]) - world.heading, 2 * math.pi)))

        state = describe(out, world.loom, names)
        key = tuple(state.values())
        if pending is not None and pending.done():
            d = pending.result()
            brain.inp[fc.IN_P_BEHAVIOUR:fc.IN_P_BEHAVIOUR + 4] = torch.tensor(d.probs)
            brain.inp[fc.IN_URGENCY] = d.urgency
            brain.inp[fc.IN_P_JUMP] = d.p_jump
            s1_ms.append(d.ms)
            s1_age.append((time.perf_counter() - t_sent) * 1e3)
            backends_used[d.backend] = backends_used.get(d.backend, 0) + 1
            pending = None
        if pending is None and key != last_key:       # only ask System 1 when the worded state changes
            pending = pool.submit(worker_decide, state)
            t_sent = time.perf_counter()
            last_key = key
        behaviour_ticks[int(np.argmax(brain.inp[fc.IN_P_BEHAVIOUR:fc.IN_P_BEHAVIOUR + 4].numpy()))] += 1

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

    pool.shutdown(cancel_futures=True)
    vram = torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.0
    print(f"device={args.device} graph={graph} n_kc={args.n_kc} tick_cpus={tick_cpus or 'unpinned'} "
          f"high_priority={realtime} "
          f"backends={args.backends}")
    print(f"tick compute   p50 {pct(compute_ms, 50):.3f}  p99 {pct(compute_ms, 99):.3f}  max {max(compute_ms):.3f} ms")
    print(f"tick period    p50 {pct(period_ms, 50):.3f}  p99 {pct(period_ms, 99):.3f}  max {max(period_ms):.3f} ms"
          f"  overruns(>{args.tick_ms + 1:.0f} ms) {overruns}/{len(period_ms)}")
    print(f"system1        calls {len(s1_ms)} {backends_used}  model p50 {pct(s1_ms, 50):.2f}  "
          f"p99 {pct(s1_ms, 99):.2f} ms  sent->latched p50 {pct(s1_age, 50):.2f}  p99 {pct(s1_age, 99):.2f}  max {max(s1_age, default=0):.2f} ms")
    print(f"behaviour      ticks {dict(zip(fc.BEHAVIOURS, behaviour_ticks.astype(int).tolist()))}  "
          f"jumps {jumps}  rewards {sorted(world.rewarded)}  brain VRAM peak {vram:.1f} MB")
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


if __name__ == "__main__":
    main()
