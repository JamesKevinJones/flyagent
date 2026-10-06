"""Synthetic vs hemibrain wiring, measured; and the search that tuned the per-neuron compass.

    python eval_connectome.py --tune-cx      # coordinate-descent search over the CX gains (prints CX_DEFAULTS)
"""
import argparse
import math

from hemibrain_circuits import CX_GAIN_KEYS, HemibrainCX, cx_acceptance, load_wiring

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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-cx", action="store_true")
    args = ap.parse_args()
    if args.tune_cx:
        tune_cx()
