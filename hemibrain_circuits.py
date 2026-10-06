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
