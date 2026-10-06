"""Closed-loop agent: world -> fly circuits (every 15 ms tick) -> System 1 (table lookup) -> VNC.

The tick never waits on a model. System 1 decides by looking the worded state up in the model's
precompiled table (rules on a miss); the model itself runs only in a filler process that completes the
table in the background. Decisions are latched into the brain's input vector on the next tick, and the
giant-fiber escape runs inside the tick regardless.

Deliberately not asyncio: asyncio's timer on Windows overshoots a 13.5 ms sleep by up to 13 ms (p99,
measured), which alone breaks the budget. The tick is a deadline loop (high-resolution sleep, then
spin the last 1.5 ms); the table filler is a future from its own process.

    python agent_loop.py --tick-cpus 2,3 --system1-cpus 4-7                      # rules (default)
    python agent_loop.py --backends synthetic --tables-dir <empty dir>          # Laya-shaped GPU load while filling
    PYTHONPATH=.deps python agent_loop.py --backends laya                        # Laya's table (fills if incomplete)
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
from system1_engine import (CHUNK, N_STATES, STALL_CHUNKS, append_table, base_key, compile_chunk, describe,
                            filler_init, load_table, lookup, missing, rules_backend, table_path)


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


def _system1_process(model, cpus):
    pin(cpus)
    filler_init(model)


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
        step = fwd * np.array([math.cos(self.heading), math.sin(self.heading)])
        self.pos += step
        speed = fwd
        if self.wall and np.abs(self.pos).max() > WALL:   # stop at the wall along the heading: no reflection (a turn
            before = self.pos - step                       # the compass can't sense) and no sideways slide (a motion
            room = [(WALL - abs(b)) / abs(d) for b, d in zip(before, step) if abs(b + d) > WALL]   # the odometer can't)
            frac = max(0.0, min(1.0, min(room)))
            self.pos = before + frac * step
            speed = frac * fwd                             # the odometer counts exactly what the body moved
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
        inp[fc.IN_SPEED] = speed
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
                 landmark_gain=0.02, wall=False, tables_dir="tables", wiring="synthetic"):
        self.device, self.n_kc, self.wiring = device, n_kc, wiring
        self.gyro_bias, self.landmark_gain, self.wall = gyro_bias, landmark_gain, wall
        self.backends = tuple(backends)
        # System 1 decides by lookup: the first model backend's precompiled table, rules on a miss.
        # The model itself only runs in the filler process, one chunk at a time, until the table is full.
        self.rules = rules_backend()
        self.model = next((b for b in self.backends if b != "rules"), None)
        self.path = table_path(self.model, tables_dir) if self.model else None
        self.table = load_table(self.path, self.model) if self.path else {}
        self.pool, self.fill, self.fill_streak, self.failed, self.write_error = None, None, 0, set(), False
        if self.path is None:
            self.table_status = "off"
        elif len(self.table) >= N_STATES:
            self.table_status = "complete"              # no worker, no model load, no VRAM
        else:
            self.table_status = "filling"
            self.pool = ProcessPoolExecutor(1, initializer=_system1_process, initargs=(self.model, system1_cpus))
            self._next_chunk()
        self.goal = DEFAULT_GOAL
        self.tick_stats = (0.0, 0.0)                    # (p50, p99) tick period, written by the pacing loop
        self.reset()

    def reset(self):
        self.brain = fc.FlyBrain(n_kc=self.n_kc, device=self.device, wiring=self.wiring)
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
        self.lookup_us, self.s1_age, self.backends_used, self.decided_by = [], [], {}, "none"
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
        t0 = time.perf_counter()
        d = lookup(state, self.table, self.rules)
        self.lookup_us.append((time.perf_counter() - t0) * 1e6)
        return _Done(d)

    def _next_chunk(self):
        """Submit the next CHUNK states not yet answered, skipping ones that already failed this session
        (or they'd fill every chunk and stall the table). False when nothing is left to ask."""
        todo = [s for s in missing(self.table) if base_key(s) not in self.failed][:CHUNK]
        if todo:
            self.fill = self.pool.submit(compile_chunk, todo)
        return bool(todo)

    def _poll_fill(self):
        """Merge a finished chunk and ask for the next; never waits on the filler."""
        if self.fill is None or not self.fill.done():
            return
        try:
            res = self.fill.result()
        except Exception as e:                          # the worker process died
            print(f"[system1] filler failed: {type(e).__name__}: {e}", flush=True)
            res = None
        self.fill = None
        good = [(s, d) for s, d in res or () if d is not None]
        self.failed.update(base_key(s) for s, d in res or () if d is None)
        if good:
            try:
                append_table(self.path, good)           # ponytail: file write on the tick thread, once per chunk
            except OSError as e:                        # locked or read-only file: it's only a cache, keep going
                if not self.write_error:
                    print(f"[system1] can't save {self.path}: {e}", flush=True)
                self.write_error = True
            self.table.update((base_key(s), d._replace(backend=f"{self.model} table")) for s, d in good)
        self.fill_streak = 0 if good else self.fill_streak + 1
        if res is None:
            self.table_status = "off"
        elif len(self.table) >= N_STATES:
            self.table_status = "complete"
        elif self.fill_streak < STALL_CHUNKS and self._next_chunk():
            return
        else:
            self.table_status = "stalled"
        self.pool.shutdown(wait=False)                  # done with the model: free its VRAM

    def step(self):
        brain, world = self.brain, self.world
        world.sense(self.tick, brain.inp, self.fwd, self.turn)
        self._poll_fill()
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
                self.decided_by = d.backend
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
            "table": {"backend": self.model or "rules", "filled": len(self.table), "total": N_STATES,
                      "status": self.table_status}, "decided_by": self.decided_by, "wiring": self.wiring,
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
              args.gyro_bias, args.landmark_gain, tables_dir=args.tables_dir, wiring=args.wiring)
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
    lookup_us, s1_age = sim.lookup_us, sim.s1_age
    print(f"device={args.device} wiring={args.wiring} graph={sim.graph} n_kc={sim.brain.n_kc} "
          f"tick_cpus={tick_cpus or 'unpinned'} "
          f"high_priority={realtime} "
          f"backends={args.backends}")
    print(f"tick compute   p50 {pct(compute_ms, 50):.3f}  p99 {pct(compute_ms, 99):.3f}  max {max(compute_ms):.3f} ms")
    print(f"tick period    p50 {pct(period_ms, 50):.3f}  p99 {pct(period_ms, 99):.3f}  max {max(period_ms):.3f} ms"
          f"  overruns(>{args.tick_ms + 1:.0f} ms) {overruns}/{len(period_ms)}")
    print(f"system1        decisions {len(lookup_us)} by {sim.backends_used}  lookup p50 {pct(lookup_us, 50):.1f}  "
          f"p99 {pct(lookup_us, 99):.1f} us  sent->latched p50 {pct(s1_age, 50):.2f}  p99 {pct(s1_age, 99):.2f} ms  "
          f"table {sim.model or 'rules'} {len(sim.table)}/{N_STATES} {sim.table_status}")
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
    ap.add_argument("--tables-dir", default="tables", help="precompiled System 1 tables (see system1_engine.py)")
    ap.add_argument("--wiring", default="synthetic", choices=("synthetic", "hemibrain"),
                    help="hemibrain: mushroom body and compass from the Janelia connectome (README 3g)")
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
    sim.set_goal(parse("go home")[0])                      # home_after_wall: the odometer must not count
    for _ in range(2000):                                  # steps the wall stopped (final review)
        sim.step()
    assert float(np.linalg.norm(sim.world.pos)) < 10, sim.world.pos

    sim = Sim()                                            # seek_banana_reaches
    sim.set_goal(parse("find the banana")[0])
    for _ in range(2000):
        sim.step()
    assert "banana" in sim.world.rewarded, sim.world.rewarded

    for text, touches in (("", True), ("avoid the smell", False)):   # with the CLI's predator, no goal touches geosmin
        sim = Sim()
        if text:
            sim.set_goal(parse(text)[0])
        for t in range(2000):
            if t == 1000:
                sim.launch_predator()
            sim.step()
        assert ("geosmin" in sim.world.rewarded) == touches, (text, sim.world.rewarded)

    sim = Sim()                                            # rest_stays_put
    sim.set_goal(parse("rest")[0])
    for _ in range(300):
        sim.step()
    assert float(np.linalg.norm(sim.world.pos)) < 1.0, sim.world.pos
    for s in (sim,):
        s.close()

    import tempfile
    from system1_engine import Decision, N_STATES, all_states, append_table, table_path
    with tempfile.TemporaryDirectory() as tmp:
        full = [(st, Decision((0.25, 0.25, 0.25, 0.25), 0.5, 0.5, "laya", 40.0)) for st in all_states()]
        append_table(table_path("laya", tmp), full)
        sim = Sim(backends=("laya",), tables_dir=tmp)                   # complete_table_no_pool
        assert sim.pool is None and sim.table_status == "complete" and "laya" not in sys.modules
        sim.step()
        snap = sim.step()
        assert snap["decided_by"] == "laya table" and snap["table"] == \
            {"backend": "laya", "filled": N_STATES, "total": N_STATES, "status": "complete"}, snap["table"]
        sim.set_goal(parse("rest")[0])                                  # goal precedence: rules know goals
        sim.step()
        snap = sim.step()
        assert snap["decided_by"] == "rules" and snap["behaviour"] == "IDLE", snap["decided_by"]
        sim.close()
        os.environ.pop("JEV_URL", None)                                 # unavailable_model_is_off, in any env:
        sim = Sim(backends=("http",), tables_dir=tmp + "/empty")       # http can't load without JEV_URL
        assert sim.table_status == "filling"
        t0 = time.perf_counter()
        while sim.table_status == "filling" and time.perf_counter() - t0 < 60:
            sim.step()
        snap = sim.step()
        assert sim.table_status == "off" and snap["decided_by"] == "rules", (sim.table_status, snap["decided_by"])
        assert snap["table"]["status"] == "off" and snap["table"]["filled"] == 0
        sim.close()
        # fill_skips_failed_states + write_error_keeps_ticking (final review): an in-process filler stands in
        # for the worker, since a stub can't cross a Windows spawn
        import system1_engine as s1
        from concurrent.futures import ThreadPoolExecutor
        rules = s1.rules_backend()
        bad = {s1.base_key(st) for st in all_states() if st["home"] in ("behind, far", "left, near")}

        def flaky(state):
            if s1.base_key(state) in bad:
                raise TimeoutError("stub")
            return rules(state)
        sim = Sim(backends=("http",), tables_dir=tmp + "/flaky")
        while sim.table_status == "filling":
            sim.step()
        s1._filler = flaky
        sim.pool, sim.table_status = ThreadPoolExecutor(1), "filling"
        os.makedirs(sim.path)                                           # the table file can't be written
        sim._next_chunk()
        t0 = time.perf_counter()
        while sim.table_status == "filling" and time.perf_counter() - t0 < 60:
            sim.step()
        assert len(sim.table) == N_STATES - len(bad) and sim.table_status == "stalled", (len(sim.table), sim.table_status)
        s1._filler = None
        sim.close()

    from system1_engine import BASE_KEYS, VOCAB
    sim = Sim(wiring="hemibrain")                                       # hemibrain_reaches_and_flees
    assert sim.brain.n_kc == 1927
    fled, seen = False, set()
    for t in range(2000):
        if t == 1000:
            sim.launch_predator()
        snap = sim.step()
        fled |= 1000 <= t < 1030 and snap["behaviour"] == "FLEE"
        st = describe(sim.brain.out, sim.world.loom, sim.names, sim.goal)
        seen.add(tuple(st[k] for k in BASE_KEYS))
    assert "banana" in sim.world.rewarded and fled, (sim.world.rewarded, fled)
    assert all(v in VOCAB[k] for st in seen for k, v in zip(BASE_KEYS, st)), seen   # describe_vocab_holds
    sim.reset()                                                          # reset_keeps_wiring
    assert sim.brain.n_kc == 1927 and sim.step()["wiring"] == "hemibrain"
    sim.close()
    print("agent_loop self-check OK")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
