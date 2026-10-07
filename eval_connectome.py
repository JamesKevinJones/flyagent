"""Synthetic vs hemibrain wiring, measured; and the search that tuned the per-neuron compass.

    python eval_connectome.py --runs 3       # A/B: odor coding, learning, compass, closed loop (README 3g)
    python eval_connectome.py --tune-cx      # coordinate-descent search over the CX gains (prints CX_DEFAULTS)
"""
import argparse
import contextlib
import io
import math
import re

import numpy as np
import torch

import fruit_fly_circuits as fc

from hemibrain_circuits import CX_DEFAULTS, CX_GAIN_KEYS, HemibrainCX, cx_acceptance, derived_ring_kernel, load_wiring

WIRINGS = ("synthetic", "hemibrain")

GRID = {**{k: (0.0, 0.5, 1.0, 2.0) for k in CX_GAIN_KEYS},
        "av_gain": (-8.0, -4.0, -2.0, 2.0, 4.0, 8.0), "substeps": (2, 4, 8), "tau_ticks": (0.5, 1.0, 2.0),
        "bias": (0.0, 0.1, 0.3)}
LANDMARK_GAIN_SCALE = (1.0, 5.0, 20.0)                     # only affects test 4; tuned in the closed loop


def score(res):
    """0 when tests 1-3 pass; otherwise how far off, summed."""
    gains = sum(max(0.0, abs(res[f"gain_{w}"] - 1) - 0.1) for w in (0.02, 0.1, 0.35))
    width = max(0.0, 60 - res["fwhm_deg"], res["fwhm_deg"] - 120) / 60
    return abs(res["n_peaks"] - 1) + width + max(0.0, res["drift_deg_per_s"] - 5) / 5 + gains


def tune_cx(rounds=3):
    data = load_wiring()
    cfg = {**{k: 1.0 for k in CX_GAIN_KEYS}, "av_gain": 4.0, "substeps": 4, "tau_ticks": 1.0, "bias": 0.0}

    def run(c):
        gains = {k: c[k] for k in CX_GAIN_KEYS}
        return cx_acceptance(lambda: HemibrainCX(data, "cpu", gains, c["av_gain"], 5.0, int(c["substeps"]),
                                                 c["tau_ticks"], c["bias"]))
    best = run(cfg)
    for rnd in range(rounds):
        for key, values in GRID.items():
            for v in values:
                if v == cfg[key]:
                    continue
                trial = {**cfg, key: v}
                res = run(trial)
                if score(res) < score(best):
                    cfg, best = trial, res
            print(f"round {rnd} {key:12s} score {score(best):.3f} {fmt(best)}", flush=True)
    print("best", cfg)
    print("result", fmt(best), "passed" if best["passed"] else "FAILED")
    return cfg, best


def fmt(res):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items())


def _code(b, odor):
    b.inp.zero_()
    b.inp[fc.IN_ODOR:] = odor
    b.tick()
    return b.kc.cpu().bool().clone()


def jaccard(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def odor_overlap(wiring, n=50):
    """Mean KC-code overlap: unrelated odors, and odors differing in one glomerulus (lower = better separated)."""
    b = fc.FlyBrain(device="cpu", wiring=wiring)
    g = torch.Generator().manual_seed(0)
    rand, near = [], []
    for _ in range(n):
        x, y = torch.rand(b.n_pn, generator=g), torch.rand(b.n_pn, generator=g)
        rand.append(jaccard(_code(b, x), _code(b, y)))
        z = x.clone()
        z[int(torch.randint(b.n_pn, (1,), generator=g))] = float(torch.rand(1, generator=g))
        near.append(jaccard(_code(b, x), _code(b, z)))
    return np.mean(rand), np.mean(near)


def learning(wiring):
    """Valence after one reward, after one punishment, and on a one-glomerulus-different odor after the reward."""
    out = {}
    g = torch.Generator().manual_seed(1)
    for name, dop in (("reward", 1.0), ("punish", -1.0)):
        b = fc.FlyBrain(device="cpu", wiring=wiring)
        x = torch.rand(b.n_pn, generator=g)
        _code(b, x)
        b.inp[fc.IN_DOPAMINE] = dop
        b.tick()
        b.inp[fc.IN_DOPAMINE] = 0.0
        b.tick()
        out[name] = float(b.out[fc.OUT_VALENCE])
        if name == "reward":
            z = x.clone()
            z[0] = 1 - float(z[0])
            _code(b, z)
            out["similar"] = float(b.out[fc.OUT_VALENCE])
    return out


def compass(wiring):
    """Drift in darkness (deg/s); and heading error over 2,000 turning ticks with a biased gyro, without and with the
    landmark (the demo's protocol): (p50 over the run, final), the spec's test 4 being p50 <= 0.05 rad with it."""
    b = fc.FlyBrain(device="cpu", wiring=wiring)
    b.inp.zero_()
    b.tick()
    h0 = float(b.out[fc.OUT_HEADING])
    for _ in range(2000):
        b.tick()
    dark = abs(math.degrees(fc._wrap(float(b.out[fc.OUT_HEADING]) - h0))) / 30
    err = {}
    for gain in (0.0, 0.02):
        b = fc.FlyBrain(device="cpu", wiring=wiring)
        b.inp.zero_()
        b.tick()
        true_h = float(b.out[fc.OUT_HEADING])
        errs = []
        for i in range(2000):
            w = 0.1 * math.sin(i / 29)
            true_h += w
            b.inp[fc.IN_ANGVEL] = w + 0.0005
            b.inp[fc.IN_LANDMARK_HEADING] = fc._wrap(true_h)
            b.inp[fc.IN_LANDMARK_GAIN] = gain
            b.tick()
            errs.append(abs(fc._wrap(float(b.out[fc.OUT_HEADING]) - true_h)))
        err[gain] = (float(np.percentile(errs, 50)), errs[-1])
    return dark, err[0.0], err[0.02]


def derived_ring():
    """The fallback compass that was measured and rejected: the synthetic ring with its kernel derived from the real
    wiring. Rotation gain over 200 ticks per speed, and where it settles on a stationary landmark (gain 0.02)."""
    def brain():
        b = fc.FlyBrain(device="cpu")
        b.w_ring = derived_ring_kernel(load_wiring())
        for _ in range(50):                                    # settle to the kernel's own bump shape
            r = torch.relu(b.w_ring @ b.bump)
            b.bump.copy_(r / r.sum())
        b.inp.zero_()
        b.tick()
        return b
    gains = {}
    for w in (0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.35):
        b = brain()
        prev, total = float(b.out[fc.OUT_HEADING]), 0.0
        for _ in range(200):
            b.inp[fc.IN_ANGVEL] = w
            b.tick()
            total += fc._wrap(float(b.out[fc.OUT_HEADING]) - prev)
            prev = float(b.out[fc.OUT_HEADING])
        gains[w] = total / (200 * w)
    stall = {}
    for deg in (16.875, 33.75, 90.0):
        b = brain()
        b.inp[fc.IN_LANDMARK_HEADING] = math.radians(deg)
        b.inp[fc.IN_LANDMARK_GAIN] = 0.02
        for _ in range(1500):
            b.tick()
        stall[deg] = math.degrees(float(b.out[fc.OUT_HEADING]))
    return gains, stall


def closed_loop(wiring):
    from agent_loop import Sim
    sim = Sim(wiring=wiring)
    counts = dict.fromkeys(fc.BEHAVIOURS, 0)
    jumps = 0
    for t in range(2000):
        if t == 1000:
            sim.launch_predator()
        snap = sim.step()
        counts[snap["behaviour"]] += 1
        jumps += snap["jump"]
    sim.close()
    return counts, sorted(sim.world.rewarded), jumps


def paced(wiring, device, runs):
    """The CLI's paced 2,000-tick run (tick-cpus 2,3), `runs` times: tick period p50 / p99 and overruns."""
    import agent_loop
    rows = []
    for _ in range(runs):
        args = argparse.Namespace(device=device, n_kc=2000, gyro_bias=0.0005, landmark_gain=0.02, ticks=2000,
                                  tick_ms=15.0, backends="rules", tick_cpus="2,3", system1_cpus="4-7",
                                  tables_dir="tables", wiring=wiring)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent_loop.run(args)
        m = re.search(r"tick period\s+p50 ([\d.]+)\s+p99 ([\d.]+).*overruns\(>\d+ ms\) (\d+)/", buf.getvalue())
        rows.append(f"p50 {m[1]} p99 {m[2]} overruns {m[3]}")
    return rows


def measure(runs):
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")
    res = cx_acceptance(lambda: HemibrainCX(load_wiring(), "cpu", **CX_DEFAULTS))
    print("per-neuron compass (closest searched config):", fmt(res))
    gains, stall = derived_ring()
    print("derived-kernel ring, rotation gain:", " ".join(f"{w}:{g:.3f}" for w, g in gains.items()))
    print("derived-kernel ring, landmark at -> settles at (deg):", " ".join(f"{a}->{s:.1f}" for a, s in stall.items()))
    for w in WIRINGS:
        print(f"\n== {w}")
        rnd, near = odor_overlap(w)
        print(f"odor overlap   unrelated {rnd:.3f}  one glomerulus changed {near:.3f}  (Jaccard of KC codes, n=50)")
        lr = learning(w)
        print(f"learning       valence reward {lr['reward']:.3f}  punish {lr['punish']:.3f}  "
              f"similar odor after reward {lr['similar']:.3f}")
        dark, nolm, lm = compass(w)
        print(f"compass        dark drift {dark:.2f} deg/s  biased gyro error p50/final: no landmark "
              f"{nolm[0]:.3f}/{nolm[1]:.3f} rad, with landmark {lm[0]:.3f}/{lm[1]:.3f} rad")
        counts, rewards, jumps = closed_loop(w)
        print(f"closed loop    {counts}  rewards {rewards}  jumps {jumps}")
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for dev in devices:
            print(f"tick ({dev})     " + " | ".join(paced(w, dev, runs)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-cx", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    tune_cx() if args.tune_cx else measure(args.runs)
