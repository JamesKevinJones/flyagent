"""Drosophila circuits as one fixed-shape tensor program.

  AL  -> divisive normalisation of glomerular odor input
  MB  -> PN->KC sparse expansion (CSR), APL k-winners-take-all, novelty, dopamine-gated MBON valence
  CX  -> EB/PB 16-wedge ring attractor (P-EN shift, ring-neuron landmark cue, Delta7-style normalisation),
         FB path integration
  VNC -> descending command blend, tripod CPG, giant-fiber escape reflex

Every buffer is preallocated and `_step()` only writes in place, so on CUDA the whole tick,
including the host<->device copies, is captured as ONE CUDA graph: one launch + one sync per tick.
The agent loop talks to it through two small pinned host vectors (`inp`, `out`), laid out below.
"""
import math
import sys
import warnings

import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")

BEHAVIOURS = ("FORAGE", "FLEE", "ORIENT", "IDLE")

# --- input vector (host writes, every tick) ---
IN_ANGVEL, IN_SPEED, IN_LOOM, IN_LOOM_BEARING, IN_ODOR_LR, IN_DOPAMINE = range(6)
IN_P_BEHAVIOUR = 6            # 4 slots: System-1 Choice probabilities, BEHAVIOURS order
IN_URGENCY = 10               # System-1 Score, normalised to 0..1
IN_P_JUMP = 11                # System-1 Noul
IN_LANDMARK_HEADING = 12      # heading implied by a visual landmark (rad, same frame as OUT_HEADING)
IN_LANDMARK_GAIN = 13         # 0 = no landmark visible; per-tick pull of the bump toward it, ~0.01..0.1
IN_ODOR = 14                  # n_pn slots of glomerular odor input
# --- output vector (device writes, every tick) ---
(OUT_HEADING, OUT_HOME_X, OUT_HOME_Y, OUT_NOVELTY, OUT_VALENCE, OUT_ODOR_ID, OUT_ODOR_MATCH,
 OUT_KC_ACTIVE, OUT_JUMP, OUT_FWD, OUT_TURN, OUT_CPG) = range(12)
N_OUT = 12


class FlyBrain:
    def __init__(self, n_pn=50, n_kc=2000, kc_fan_in=6, kc_sparsity=0.05, n_wedges=16, n_odor_tags=32,
                 device="cuda", seed=0, wiring="synthetic"):
        self.device = torch.device(device)
        dev, f32 = self.device, torch.float32
        g = torch.Generator().manual_seed(seed)
        self.wiring, self.mb = wiring, None
        if wiring == "hemibrain":                # real MB from the connectome; n_pn and n_kc come from the data
            from hemibrain_circuits import HemibrainMB, load_wiring
            self.mb = HemibrainMB(load_wiring(), dev, kc_sparsity)
            n_pn, n_kc = self.mb.n_pn, self.mb.n_kc
        elif wiring != "synthetic":
            raise ValueError(f"wiring must be 'synthetic' or 'hemibrain', not {wiring!r}")
        self.n_pn, self.n_kc, self.n_wedges = n_pn, n_kc, n_wedges
        self.k_active = max(1, int(n_kc * kc_sparsity))

        # ---- tunables (the calibration knobs; real flies and real sensors drift) ----
        self.gf_threshold = 0.7        # looming level that fires the giant fiber with no System-1 priming
        self.kc_novelty_decay = 0.999  # familiarity half-life ~ 700 ticks ~ 10 s at 15 ms
        self.dopamine_lr = 0.05
        self.base_speed = torch.tensor([0.6, 1.5, 0.2, 0.0], device=dev)  # per behaviour, BEHAVIOURS order
        self.turn_gain = 0.5
        self.cpg_hz_per_speed = 12.0
        self.dt = 0.015

        # ---- MB: PN->KC random fan-in, binary weights, stored as CSR (n_kc x n_pn) ----
        if self.mb is None:
            rows = torch.arange(n_kc).repeat_interleave(kc_fan_in)
            cols = torch.randint(0, n_pn, (n_kc * kc_fan_in,), generator=g)
            self.w_pn_kc = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(rows.numel()),
                                                   (n_kc, n_pn), check_invariants=False
                                                   ).coalesce().to_sparse_csr().to(dev)
            self.kc = torch.zeros(n_kc, device=dev)
            self.kc_familiarity = torch.zeros(n_kc, device=dev)
        else:                                    # the same tensors, so odor tags and recall read the real KCs
            self.w_pn_kc, self.kc, self.kc_familiarity = self.mb.w_pn_kc, self.mb.kc, self.mb.kc_familiarity
            self.k_active = self.mb.k_active
        self.w_kc_mbon = torch.zeros(n_kc, device=dev)               # signed valence, learned by dopamine
        self.odor_tags = torch.zeros(n_odor_tags, n_kc, device=dev)  # remembered KC codes, row = odor id
        # goal buffers (set_goal): per-odor approach sign, and [goal heading, heading weight, home sign]
        self.odor_sign = torch.ones(n_odor_tags, device=dev)
        self.goal_vec = torch.tensor([0.0, 0.0, 1.0], device=dev)

        # ---- CX: wedge angles, attractor kernel, bump state, FB memory ----
        self.theta = torch.arange(n_wedges, device=dev, dtype=f32) * (2 * math.pi / n_wedges)
        d = self.theta[:, None] - self.theta[None, :]
        self.w_ring = torch.cos(d)                   # local excitation; zero-mean => built-in global inhibition
        self.e_cos, self.e_sin = torch.cos(self.theta), torch.sin(self.theta)
        self.bump = torch.relu(torch.cos(self.theta))
        self.bump /= self.bump.sum()
        self.fb_mem = torch.zeros(n_wedges, device=dev)
        # population-vector gain of a normalised bump, so FB memory decodes to body lengths
        self.pv_gain = float(torch.hypot((self.bump * self.e_cos).sum(), (self.bump * self.e_sin).sum()))
        self.cpg_phase = torch.zeros((), device=dev)

        # ---- host <-> device boundary ----
        n_in = IN_ODOR + n_pn
        pin = self.device.type == "cuda"
        self.inp = torch.zeros(n_in, pin_memory=pin)                 # host writes here
        self.out = torch.zeros(N_OUT, pin_memory=pin)                # host reads here
        self._inp_dev = torch.zeros(n_in, device=dev)
        self._out_dev = torch.zeros(N_OUT, device=dev)
        self._graph = None
        self._done = None

    # ------------------------------------------------------------------ one tick
    @torch.no_grad()
    def _step(self):
        self._inp_dev.copy_(self.inp, non_blocking=True)
        x = self._inp_dev
        o = self._out_dev

        # AL: divisive normalisation -> KC code is concentration-invariant
        odor = x[IN_ODOR:]
        pn = odor / (odor.mean() + 1e-3)

        # MB: sparse expansion + APL k-WTA (only KCs with real drive may fire)
        if self.mb is not None:                  # fixed at construction, so still one graph
            self.mb.step(pn, x[IN_DOPAMINE], o)
            n_active = self.kc.sum().clamp_min(1.0)
        else:
            drive = (self.w_pn_kc @ pn[:, None]).squeeze(1)
            thr = drive.topk(self.k_active).values[-1]
            self.kc.copy_(((drive >= thr) & (drive > 0)).float())
            n_active = self.kc.sum().clamp_min(1.0)
            o[OUT_KC_ACTIVE] = n_active
            o[OUT_NOVELTY] = 1.0 - (self.kc_familiarity * self.kc).sum() / n_active
            self.kc_familiarity.copy_(torch.maximum(self.kc_familiarity * self.kc_novelty_decay, self.kc))
            # dopamine (DAN) gates plasticity at the KC->MBON synapse; zero on most ticks
            self.w_kc_mbon.add_(self.kc * x[IN_DOPAMINE] * self.dopamine_lr).clamp_(-1.0, 1.0)
            o[OUT_VALENCE] = (self.w_kc_mbon * self.kc).sum() / n_active
        match = (self.odor_tags @ self.kc) / n_active             # overlap with each remembered odor
        best = match.max(0)
        o[OUT_ODOR_ID] = best.indices.float()
        o[OUT_ODOR_MATCH] = best.values

        # CX/PB: P-EN neurons shift the bump by the angular velocity. Interpolation weight `a` is
        # solved so the first-harmonic phase moves by exactly `angvel` (|angvel| <= one wedge/tick).
        step = 2 * math.pi / self.n_wedges
        psi = x[IN_ANGVEL].clamp(-step, step)
        t = torch.tan(psi.abs())
        a = t / (math.sin(step) + t * (1 - math.cos(step)))
        nb = torch.where(psi >= 0, torch.roll(self.bump, 1), torch.roll(self.bump, -1))
        shifted = (1 - a) * self.bump + a * nb
        # Ring neurons: a visible landmark adds a cosine-tuned bump at the heading it implies. Both bumps
        # sum to 1, so after the recurrence the phase moves ~gain/(1+gain) of the way toward the landmark
        # per tick. That bounds gyro drift at about bias*(1+gain)/gain instead of letting it grow forever.
        cue = torch.relu(torch.cos(self.theta - x[IN_LANDMARK_HEADING]))
        shifted = shifted + x[IN_LANDMARK_GAIN] * cue / cue.sum()
        # EB attractor: cosine recurrence keeps the phase and restores the bump shape; Delta7-style
        # global normalisation fixes the amplitude, so the bump neither dies nor explodes.
        r = torch.relu(self.w_ring @ shifted)
        self.bump.copy_(r / r.sum().clamp_min(1e-6))
        heading = torch.atan2((self.bump * self.e_sin).sum(), (self.bump * self.e_cos).sum())
        o[OUT_HEADING] = heading

        # FB: path integration (Stone et al. 2017, CPU4-style): columns accumulate speed x heading tuning
        self.fb_mem.add_(self.bump * x[IN_SPEED])
        dx = (self.fb_mem * self.e_cos).sum() / self.pv_gain
        dy = (self.fb_mem * self.e_sin).sum() / self.pv_gain
        o[OUT_HOME_X] = -dx
        o[OUT_HOME_Y] = -dy

        # VNC: blend per-behaviour motor programs by System-1 probabilities
        p = x[IN_P_BEHAVIOUR:IN_P_BEHAVIOUR + 4]
        home_turn = torch.sin(torch.atan2(-dy, -dx) - heading)
        # goal context: the identified odor's sign (+1 approach, -1 avoid, 0 ignore) and a PFL3-style heading drive
        odor_sign = torch.where(best.values > 0.6, self.odor_sign.index_select(0, best.indices.reshape(1))[0], 1.0)
        heading_drive = self.goal_vec[1] * torch.sin(self.goal_vec[0] - heading)
        turns = torch.stack([
            odor_sign * x[IN_ODOR_LR] + heading_drive,  # FORAGE: bilateral odor comparison, plus goal heading
            -torch.sin(x[IN_LOOM_BEARING]),      # FLEE: turn away from the looming side
            self.goal_vec[2] * home_turn,        # ORIENT: face (or, with home sign -1, turn from) home
            torch.zeros((), device=self.device),  # IDLE
        ])
        urgency = 0.5 + x[IN_URGENCY]
        o[OUT_FWD] = (p * self.base_speed).sum() * urgency
        o[OUT_TURN] = (p * turns).sum() * self.turn_gain
        self.cpg_phase.copy_(torch.remainder(self.cpg_phase + 2 * math.pi * self.cpg_hz_per_speed * o[OUT_FWD]
                                             * self.dt, 2 * math.pi))
        o[OUT_CPG] = self.cpg_phase              # tripod A = phase, tripod B = phase + pi
        # Giant fiber: the reflex never waits on System 1. System 1's Noul only primes it, lowering
        # the threshold by up to half, the way behavioural state modulates escape in real flies.
        o[OUT_JUMP] = (x[IN_LOOM] > self.gf_threshold * (1 - 0.5 * x[IN_P_JUMP])).float()

        self.out.copy_(o, non_blocking=True)

    # ------------------------------------------------------------------ public API
    def capture(self):
        """Record one tick as a CUDA graph. Mutates state (warm-up ticks), so call before the loop."""
        if self.device.type != "cuda":
            return False
        saved = [t.clone() for t in self._state()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._step()
        for t, v in zip(self._state(), saved):
            t.copy_(v)
        self._done = torch.cuda.Event()
        torch.cuda.synchronize()
        return True

    def tick(self):
        """Run one tick on `self.inp`; results land in `self.out` when this returns."""
        if self._graph is None:
            self._step()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            return self.out
        self._graph.replay()
        self._done.record()
        self._done.synchronize()
        return self.out

    def set_goal(self, odor_signs, goal_heading, heading_weight, home_sign):
        """Compile a goal into the brain (host call, outside the graph). Writes in place, so a captured
        graph reads the new values on its next replay without re-capture."""
        self.odor_sign.copy_(torch.tensor(odor_signs, dtype=torch.float32))
        self.goal_vec.copy_(torch.tensor([goal_heading, heading_weight, home_sign], dtype=torch.float32))

    def remember_odor(self, odor_id):
        """Store the current KC code as odor `odor_id` (host call, outside the graph)."""
        self.odor_tags[odor_id].copy_(self.kc)

    def _state(self):
        mb = self.mb.state() if self.mb is not None else [self.kc, self.kc_familiarity, self.w_kc_mbon]
        return mb + [self.bump, self.fb_mem, self.cpg_phase]

    def vram_mb(self):
        tensors = self._state() + [self.odor_tags, self.w_ring, self._inp_dev, self._out_dev,
                                   self.w_pn_kc.values(), self.w_pn_kc.col_indices(), self.w_pn_kc.crow_indices()]
        return sum(t.numel() * t.element_size() for t in tensors) / 2**20


# ---------------------------------------------------------------------------- self-check
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def demo(device):
    b = FlyBrain(device=device)
    graph = b.capture()
    odor_a = torch.rand(b.n_pn, generator=torch.Generator().manual_seed(1))
    odor_b = torch.rand(b.n_pn, generator=torch.Generator().manual_seed(2))

    # 1. ring attractor integrates angular velocity exactly (no landmark)
    b.inp.zero_()
    true_h = 0.0
    for i in range(400):
        w = 0.3 * math.sin(i / 17)                        # up to ~20 rad/s at 15 ms ticks
        b.inp[IN_ANGVEL] = w
        true_h += w
        b.tick()
    err = abs(_wrap(float(b.out[OUT_HEADING]) - true_h))
    assert err < 0.02, f"heading drift {err:.4f} rad"

    # 2. landmark: a biased gyro drifts without one; with one the error stays bounded, and a
    #    conflicting landmark pulls the bump onto it
    bias, drift = 0.0005, {}                              # ~2 deg/s, an uncalibrated MEMS gyro
    for gain in (0.0, 0.02):
        b.inp.zero_()
        true_h = float(b.out[OUT_HEADING])
        for i in range(2000):
            w = 0.1 * math.sin(i / 29)
            true_h += w
            b.inp[IN_ANGVEL] = w + bias                   # the circuit only ever sees the biased gyro
            b.inp[IN_LANDMARK_HEADING] = _wrap(true_h)
            b.inp[IN_LANDMARK_GAIN] = gain
            b.tick()
        drift[gain] = abs(_wrap(float(b.out[OUT_HEADING]) - true_h))
    assert drift[0.0] > 0.9 and drift[0.02] < 0.05, drift  # 2000 * 0.0005 = 1.0 rad uncorrected
    b.inp.zero_()
    h0 = float(b.out[OUT_HEADING])
    b.inp[IN_LANDMARK_HEADING] = _wrap(h0 + 1.5)
    b.inp[IN_LANDMARK_GAIN] = 0.05
    for _ in range(300):
        b.tick()
    settle = abs(_wrap(float(b.out[OUT_HEADING]) - (h0 + 1.5)))
    assert settle < 0.01, f"bump did not settle on the landmark: {settle:.4f} rad off"

    # 3. path integration: walk a square, home vector returns to ~0
    b.inp.zero_()
    for leg in range(4):
        for _ in range(25):
            b.inp[IN_SPEED] = 1.0
            b.tick()
        b.inp[IN_SPEED] = 0.0
        for _ in range(4):                                # four quarter-wedge-limited turns of pi/8
            b.inp[IN_ANGVEL] = math.pi / 8
            b.tick()
        b.inp[IN_ANGVEL] = 0.0
    home = math.hypot(float(b.out[OUT_HOME_X]), float(b.out[OUT_HOME_Y]))
    assert home < 1.0, f"home vector after a closed square {home:.3f} (side 25)"

    # 4. MB: ~5% sparse code, novelty drops on repeat, odor recall, dopamine sets valence
    b.inp.zero_()
    b.inp[IN_ODOR:] = odor_a
    b.tick()
    assert abs(float(b.out[OUT_KC_ACTIVE]) - b.k_active) <= 2, float(b.out[OUT_KC_ACTIVE])
    assert float(b.out[OUT_NOVELTY]) > 0.9
    b.remember_odor(3)
    b.inp[IN_DOPAMINE] = 1.0                              # reward paired with odor A
    b.tick()
    b.inp[IN_DOPAMINE] = 0.0
    b.tick()
    assert float(b.out[OUT_NOVELTY]) < 0.05
    assert int(b.out[OUT_ODOR_ID]) == 3 and float(b.out[OUT_ODOR_MATCH]) > 0.95
    assert float(b.out[OUT_VALENCE]) > 0
    b.inp[IN_ODOR:] = odor_b
    b.tick()
    assert float(b.out[OUT_NOVELTY]) > 0.5 and float(b.out[OUT_ODOR_MATCH]) < 0.5
    b.inp[IN_ODOR:] = odor_a * 5                          # concentration-invariant identity
    b.tick()
    assert float(b.out[OUT_ODOR_MATCH]) > 0.95

    # 5. giant fiber: fires on loom alone; System-1 priming lowers the threshold
    b.inp[IN_LOOM] = 0.5
    b.tick()
    assert float(b.out[OUT_JUMP]) == 0.0
    b.inp[IN_P_JUMP] = 1.0
    b.tick()
    assert float(b.out[OUT_JUMP]) == 1.0

    # 6. goal buffers: set outside the graph, read by it every tick (no re-capture)
    def goal_brain(behaviour):
        g = FlyBrain(device=device)
        g.capture()
        g.inp.zero_()
        g.inp[IN_P_BEHAVIOUR + BEHAVIOURS.index(behaviour)] = 1.0
        return g

    g = goal_brain("FORAGE")                              # odor_sign_flips_turn
    g.inp[IN_ODOR:] = odor_a
    g.tick()
    g.remember_odor(3)
    g.inp[IN_ODOR_LR] = 0.5
    g.tick()
    assert float(g.out[OUT_TURN]) > 0, float(g.out[OUT_TURN])
    signs = [1.0] * g.odor_tags.shape[0]
    signs[3] = -1.0
    g.set_goal(signs, 0.0, 0.0, 1.0)
    g.tick()
    assert float(g.out[OUT_TURN]) < 0, float(g.out[OUT_TURN])

    g = goal_brain("FORAGE")                              # heading_drive_north: heading 0 = east, no odor
    g.set_goal([1.0] * 32, math.pi / 2, 0.4, 1.0)
    g.tick()
    assert float(g.out[OUT_TURN]) > 0, float(g.out[OUT_TURN])
    g.set_goal([1.0] * 32, math.pi * 3 / 2, 0.4, 1.0)
    g.tick()
    assert float(g.out[OUT_TURN]) < 0, float(g.out[OUT_TURN])

    g = goal_brain("ORIENT")                              # home_sign_flips_orient: walk east, turn north, walk
    for speed, angvel, n in ((1.0, 0.0, 10), (0.0, math.pi / 8, 4), (1.0, 0.0, 10), (0.0, 0.0, 1)):
        g.inp[IN_SPEED], g.inp[IN_ANGVEL] = speed, angvel
        for _ in range(n):
            g.tick()
    toward = float(g.out[OUT_TURN])
    g.set_goal([1.0] * 32, 0.0, 0.0, -1.0)
    g.tick()
    away = float(g.out[OUT_TURN])
    assert abs(toward) > 0.05 and toward * away < 0, (toward, away)
    print(f"[{device}] self-check OK  graph={graph}  heading_err={err:.4f} rad  gyro-bias drift {drift[0.0]:.2f} -> {drift[0.02]:.3f} rad with landmark"
          f"  home_after_square={home:.3f}"
          f"  circuit VRAM={b.vram_mb():.2f} MB")


def demo_hemibrain(device):
    b = FlyBrain(device=device, wiring="hemibrain")
    graph = b.capture()
    assert b.n_kc == 1927 and "hemibrain" not in sys.modules            # offline_load: never the downloader
    odor_a = torch.rand(b.n_pn, generator=torch.Generator().manual_seed(1))
    b.inp.zero_()
    b.inp[IN_ODOR:] = odor_a
    b.tick()
    assert abs(float(b.out[OUT_KC_ACTIVE]) - b.mb.k_active) <= 2 and float(b.out[OUT_NOVELTY]) > 0.9
    assert abs(float(b.out[OUT_VALENCE])) < 1e-6                           # an untrained odor reads 0
    b.remember_odor(3)
    b.inp[IN_DOPAMINE] = 1.0                                               # one reward
    b.tick()
    b.inp[IN_DOPAMINE] = 0.0
    b.tick()
    assert float(b.out[OUT_NOVELTY]) < 0.05 and int(b.out[OUT_ODOR_ID]) == 3
    reward = float(b.out[OUT_VALENCE])
    assert reward >= 0.1, reward
    p = FlyBrain(device=device, wiring="hemibrain")                        # one punishment
    p.inp[IN_ODOR:] = odor_a
    p.tick()
    p.inp[IN_DOPAMINE] = -1.0
    p.tick()
    p.inp[IN_DOPAMINE] = 0.0
    p.tick()
    punish = float(p.out[OUT_VALENCE])
    assert punish <= -0.1, punish
    for _ in range(100):                                                   # valence_saturates
        b.inp[IN_DOPAMINE] = 1.0
        b.tick()
    v = float(b.out[OUT_VALENCE])
    assert math.isfinite(v) and -1.0 <= v <= 1.0, v
    print(f"[{device}] hemibrain self-check OK  graph={graph}  n_pn={b.n_pn} n_kc={b.n_kc}  "
          f"valence reward {reward:.3f} punish {punish:.3f} saturated {v:.3f}")


if __name__ == "__main__":
    demo("cpu")
    demo_hemibrain("cpu")
    if torch.cuda.is_available():
        demo("cuda")
        demo_hemibrain("cuda")
