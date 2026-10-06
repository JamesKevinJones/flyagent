"""Mushroom body and compass on wiring from the Janelia hemibrain v1.2 connectome (data/hemibrain_mb_cx.npz).

FlyBrain(wiring="hemibrain") builds these instead of its synthetic circuits. Like the rest of FlyBrain they are
fixed-shape tensor programs: preallocated buffers, in-place writes, no host syncs, so one tick still captures
as one CUDA graph. Data: Janelia FlyEM hemibrain v1.2 (Scheffer et al. 2020, eLife), CC BY 4.0.
"""
from pathlib import Path

import numpy as np
import torch

from fruit_fly_circuits import OUT_KC_ACTIVE, OUT_NOVELTY, OUT_VALENCE

DATA_PATH = Path(__file__).with_name("data") / "hemibrain_mb_cx.npz"
# Smallest of {0.05, 0.1, 0.2, 0.5, 1.0} for which one reward gives valence >= 0.1 and one punishment <= -0.1
MB_LR = 0.5


def load_wiring(path=DATA_PATH):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


class HemibrainMB:
    """Real PN->KC expansion, APL as 5% k-winners-take-all, and learning at the real KC->MBON synapses.
    Dopamine depresses the active KCs' synapses onto the MBONs it innervates (Aso et al. 2014): reward through PAM,
    punishment through PPL1, split per MBON by its measured PAM/PPL1 input. Valence is read against the naive
    network: reward depresses avoidance MBONs (sign < 0) -> positive; punishment depresses approach MBONs -> negative."""

    def __init__(self, data, device, kc_sparsity=0.05, novelty_decay=0.999, lr=MB_LR):
        dev = torch.device(device)
        self.n_pn, self.n_kc = len(data["pn_types"]), int(data["n_kc"])
        self.k_active = max(1, int(self.n_kc * kc_sparsity))
        self.novelty_decay, self.lr = novelty_decay, lr
        idx = torch.stack([torch.from_numpy(data["pn_kc_row"]).long(), torch.from_numpy(data["pn_kc_col"]).long()])
        self.w_pn_kc = torch.sparse_coo_tensor(idx, torch.from_numpy(data["pn_kc_w"]), (self.n_kc, self.n_pn),
                                               check_invariants=False).coalesce().to_sparse_csr().to(dev)
        row = torch.from_numpy(data["kc_mbon_row"]).long()
        self.col = torch.from_numpy(data["kc_mbon_col"]).long().to(dev)
        self.w0 = torch.from_numpy(data["kc_mbon_w0"]).to(dev)
        self.sign_w0 = torch.from_numpy(data["mbon_sign"])[row].to(dev) * self.w0
        self.pam = torch.from_numpy(data["mbon_pam_frac"])[row].to(dev)
        self.ppl1 = torch.from_numpy(data["mbon_ppl1_frac"])[row].to(dev)
        self.d = torch.zeros_like(self.w0)                       # learned depression per KC->MBON synapse
        self.kc = torch.zeros(self.n_kc, device=dev)
        self.kc_familiarity = torch.zeros(self.n_kc, device=dev)

    def step(self, pn, dopamine, o):
        drive = (self.w_pn_kc @ pn[:, None]).squeeze(1)
        thr = drive.topk(self.k_active).values[-1]
        self.kc.copy_(((drive >= thr) & (drive > 0)).float())
        n_active = self.kc.sum().clamp_min(1.0)
        o[OUT_KC_ACTIVE] = n_active
        o[OUT_NOVELTY] = 1.0 - (self.kc_familiarity * self.kc).sum() / n_active
        self.kc_familiarity.copy_(torch.maximum(self.kc_familiarity * self.novelty_decay, self.kc))
        kc_g = self.kc.index_select(0, self.col)
        self.d.add_(self.lr * kc_g * (torch.relu(dopamine) * self.pam + torch.relu(-dopamine) * self.ppl1)).clamp_(0, 1)
        o[OUT_VALENCE] = -(self.sign_w0 * self.d * kc_g).sum() / ((self.w0 * kc_g).sum() + 1e-6)

    def state(self):
        return [self.kc, self.kc_familiarity, self.d]


# ------------------------------------------------------------------ compass: one rate unit per real CX neuron
CX_GROUP = {"EPG": "EPG", "EPGt": "EPG", "PEN1": "PEN", "PEN2": "PEN", "Delta7": "Delta7", "PEG": "PEG"}
CX_GAIN_KEYS = ("EPG>Delta7", "Delta7>all", "EPG>PEN", "PEN>EPG", "EPG>PEG", "PEG>EPG", "EPG>EPG")


class HemibrainCX:
    """EPG, PEN, Delta7 and PEG neurons as rate units, W = synapse counts x sign (Delta7 inhibitory) x a gain per
    type pair, each row normalised by its total input count. Rates saturate at 1. Angular velocity drives the PENs of
    each bridge side in opposite directions; the real PEN->EPG wiring (one wedge either way) moves the bump.
    A landmark drives the EPGs, cosine-tuned, like the synthetic ring's ring-neuron cue."""

    def __init__(self, data, device, gains, av_gain, landmark_gain_scale, substeps, tau_ticks, bias=0.0):
        dev = torch.device(device)
        group = np.array([CX_GROUP[t] for t in data["cx_types"]])
        counts = data["cx_w"].astype(np.float64)                  # [post, pre]
        gain = np.zeros_like(counts)
        for i, post in enumerate(group):
            for j, pre in enumerate(group):
                gain[i, j] = gains.get("Delta7>all" if pre == "Delta7" else f"{pre}>{post}", 0.0)
        sign = np.where(group == "Delta7", -1.0, 1.0)[None, :]
        total = counts.sum(1, keepdims=True)
        self.W = torch.from_numpy(counts * gain * sign / np.where(total > 0, total, 1)).float().to(dev)
        theta = torch.from_numpy(data["cx_theta"]).float()
        self.theta = theta.to(dev)
        self.epg = torch.from_numpy(group == "EPG").to(dev)
        side = np.where(data["cx_side"] == "R", 1.0, -1.0) * (group == "PEN")
        self.av_drive = torch.from_numpy(side).float().to(dev) * av_gain
        self.lm_drive = self.epg.float() * landmark_gain_scale
        self.bias = torch.full_like(self.theta, bias)
        self.substeps, self.rate = substeps, 1.0 / (tau_ticks * substeps)
        self.epg_cos, self.epg_sin = torch.cos(self.theta) * self.epg, torch.sin(self.theta) * self.epg
        self.bin16 = ((torch.remainder(theta, 2 * np.pi) / (2 * np.pi / 16)).round().long() % 16).to(dev)
        self.b16 = torch.zeros(16, device=dev)
        self.r = torch.relu(torch.cos(self.theta)) * self.epg     # start: a bump at heading 0

    def step(self, angvel, landmark_heading, landmark_gain, o=None):
        I = self.bias + self.av_drive * angvel + self.lm_drive * landmark_gain * torch.cos(self.theta - landmark_heading)
        for _ in range(self.substeps):
            self.r.add_(self.rate * (torch.clamp(self.W @ self.r + I, 0.0, 1.0) - self.r))
        heading = torch.atan2((self.r * self.epg_sin).sum(), (self.r * self.epg_cos).sum())
        if o is not None:
            from fruit_fly_circuits import OUT_HEADING
            o[OUT_HEADING] = heading
        return heading

    def bump16(self):
        """EPG activity binned into the page's 16 compass sectors, summing to 1."""
        self.b16.zero_().index_add_(0, self.bin16, self.r * self.epg)
        return self.b16 / self.b16.sum().clamp_min(1e-6)

    def state(self):
        return [self.r]


def _tiles(cx):
    """EPG activity per 45-degree tile (left and right EPGs share tiles under the data-chosen map)."""
    t = (torch.remainder(cx.theta, 2 * np.pi) / (np.pi / 4)).round().long() % 8
    return torch.zeros(8).index_add_(0, t[cx.epg].cpu(), cx.r[cx.epg].cpu()).numpy()


def _shape(p):
    """(number of peaks >= half max, full width at half max in degrees) of a circular 8-tile profile."""
    if p.max() <= 0:
        return 0, 0.0
    peaks = sum(1 for i in range(8) if p[i] >= 0.5 * p.max() and p[i] > p[i - 1] and p[i] >= p[(i + 1) % 8])
    fine = np.interp(np.arange(360), np.arange(9) * 45, np.append(p, p[0]))
    return peaks, float((fine >= 0.5 * p.max()).sum())


def cx_acceptance(make_cx):
    """The spec's compass tests 1-3 on a fresh circuit each: forms, holds, rotation gain."""
    import math
    zero = torch.zeros(())
    res = {}
    cx = make_cx()
    cx.r.copy_(torch.rand(cx.r.shape, generator=torch.Generator().manual_seed(0)) * 0.1)
    for _ in range(200):
        cx.step(zero, zero, zero)
    res["n_peaks"], res["fwhm_deg"] = _shape(_tiles(cx))
    cx = make_cx()
    h0 = float(cx.step(zero, zero, zero))
    total = 0.0
    prev = h0
    for _ in range(2000):
        h = float(cx.step(zero, zero, zero))
        total += math.remainder(h - prev, 2 * math.pi)
        prev = h
    res["drift_deg_per_s"] = abs(math.degrees(total)) / (2000 * 0.015)
    for w in (0.02, 0.1, 0.35):
        cx = make_cx()
        for _ in range(50):
            cx.step(zero, zero, zero)
        prev, total = float(cx.step(zero, zero, zero)), 0.0
        for _ in range(200):
            h = float(cx.step(torch.tensor(w), zero, zero))
            total += math.remainder(h - prev, 2 * math.pi)
            prev = h
        res[f"gain_{w}"] = total / (200 * w)
    res["passed"] = (res["n_peaks"] == 1 and 60 <= res["fwhm_deg"] <= 120 and res["drift_deg_per_s"] < 5
                     and all(0.9 <= res[f"gain_{w}"] <= 1.1 for w in (0.02, 0.1, 0.35)))
    return res


# The closest per-neuron configuration found (eval_connectome.py --tune-cx, then the edge-extended ranges in
# docs/eval-connectome-2026-10-06.txt, 2026-10-06). It forms one bump (FWHM 117 deg) and holds it (~3 deg/s drift),
# but fails the rotation-gain test: the bump moves at most ~0.016 rad/tick (~60 deg/s) whatever the drive, so the agent
# uses derived_ring_kernel instead. Kept so the measurement can be re-run.
CX_DEFAULTS = {"gains": {"EPG>Delta7": 1.0, "Delta7>all": 0.5, "EPG>PEN": 1.0, "PEN>EPG": 2.0, "EPG>PEG": 0.0,
                         "PEG>EPG": 1.0, "EPG>EPG": 1.0},
               "av_gain": -64.0, "landmark_gain_scale": 5.0, "substeps": 32, "tau_ticks": 1.0}


def derived_ring_kernel(data, n_wedges=16):
    """The fallback compass kernel: effective EPG->EPG coupling from the real wiring (direct, plus the PEN and PEG
    loops, minus the Delta7 loop; each neuron's input as fractions of its total), averaged into a circulant profile
    over the 8 EPG tiles, symmetrised, resampled to n_wedges and made zero-mean like the synthetic cosine kernel.
    The PEN shift itself stays the synthetic ring's exact P-EN interpolation."""
    group = np.array([CX_GROUP[t] for t in data["cx_types"]])
    counts = data["cx_w"].astype(np.float64)
    a = counts / np.where(counts.sum(1, keepdims=True) > 0, counts.sum(1, keepdims=True), 1)
    e = group == "EPG"
    block = {g: group == g for g in ("PEN", "PEG", "Delta7")}

    def path(g):
        return a[np.ix_(e, block[g])] @ a[np.ix_(block[g], e)]
    eff = a[np.ix_(e, e)] + path("PEN") + path("PEG") - path("Delta7")
    tile = (np.round(np.mod(data["cx_theta"][e], 2 * np.pi) / (np.pi / 4)).astype(int)) % 8
    prof = np.zeros(8)
    for d in range(8):                                     # coupling from tile t-d onto tile t, averaged over t
        prof[d] = np.mean([eff[np.ix_(tile == t, tile == (t - d) % 8)].sum(1).mean() for t in range(8)])
    prof = (prof + np.roll(prof[::-1], 1)) / 2             # symmetric: the turning asymmetry is the P-EN shift's job
    x = np.arange(n_wedges) * 8 / n_wedges
    k = np.interp(x, np.arange(9), np.append(prof, prof[0]))
    k = k - k.mean()
    k = k / np.abs(k).max()
    idx = (np.arange(n_wedges)[:, None] - np.arange(n_wedges)[None, :]) % n_wedges
    return torch.from_numpy(k[idx]).float()
