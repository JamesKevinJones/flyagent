"""Compile mushroom-body and central-complex wiring from the Janelia hemibrain v1.2 connectome.

    python hemibrain.py                 # download (once, 45.9 MB, into .deps/hemibrain/), derive, write the .npz
    python hemibrain.py --selfcheck     # check the committed .npz (no download)

The runtime never imports this module: FlyBrain(wiring="hemibrain") reads only data/hemibrain_mb_cx.npz.
Data: Janelia FlyEM hemibrain v1.2 (Scheffer et al. 2020, eLife), CC BY 4.0; see data/DATA_LICENSE.
"""
import csv
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ARCHIVE_URL = "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
ARCHIVE_SIZE = 45_872_577
DATA_PATH = Path("data/hemibrain_mb_cx.npz")
ATTRIBUTION = "Janelia FlyEM hemibrain v1.2 (Scheffer et al. 2020, eLife), CC BY 4.0"


PN_TYPE = re.compile(r"^[A-Za-z0-9+_]+_(adPN|lPN|vPN|lvPN|ilPN|l2PN|ivPN)\d*$")
CX_TYPES = {"EPG": "EPG", "EPGt": "EPGt", "PEN_a(PEN1)": "PEN1", "PEN_b(PEN2)": "PEN2", "Delta7": "Delta7",
            "PEG": "PEG"}
MIN_PN_KC = 3                                   # PN->KC pairs with fewer synapses are dropped as noise


def fetch(cache_dir=".deps/hemibrain"):
    """Download the archive once (refusing a wrong size: there is no published checksum), extract, return the dir."""
    cache = Path(cache_dir)
    archive = cache / ARCHIVE_URL.rsplit("/", 1)[1]
    if not archive.exists():
        cache.mkdir(parents=True, exist_ok=True)
        print(f"downloading {ARCHIVE_URL} (45.9 MB)", flush=True)
        with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
    if archive.stat().st_size != ARCHIVE_SIZE:
        raise ValueError(f"{archive} is {archive.stat().st_size} bytes, expected {ARCHIVE_SIZE}: delete it and rerun")
    out = cache / "exported-traced-adjacencies-v1.2"
    if not (out / "traced-total-connections.csv").exists():
        with tarfile.open(archive) as t:
            t.extractall(cache, members=[m for m in t.getmembers() if not Path(m.name).name.startswith("._")],
                         filter="data")
    return out


def _glomeruli(instance):
    """'EPG(PB08)_L3' -> ['L3']; 'Delta7(PB15)_L1L9R8_R' -> ['L1', 'L9', 'R8']."""
    return re.findall(r"[LR]\d", instance.split(")_", 1)[1].split("_")[0])


def _theta(glomerulus):
    """Bridge glomerulus -> compass angle: 45 degrees per glomerulus (1 and 9 share an angle), the left bridge running
    the other way from 45 degrees. Chosen from the data, not assumed: of the candidate maps (either direction, any
    22.5-degree offset), this is the one under which the real PEN->EPG wiring shifts the bump by one wedge in opposite
    directions per side (+22.4 / -22.3 degrees), as published (Turner-Evans et al. 2020). A global rotation or mirror
    is absorbed by the angular-velocity gain's sign and by the landmark."""
    k = (int(glomerulus[1]) - 1) % 8
    return math.radians(45 - k * 45 if glomerulus[0] == "L" else k * 45)


def derive(src_dir):
    src = Path(src_dir)
    with open(src / "traced-neurons.csv", encoding="utf-8") as f:
        neurons = {int(r["bodyId"]): (r["type"] or "", r["instance"] or "") for r in csv.DictReader(f)}
    kcs = sorted(b for b, (t, _) in neurons.items() if t.startswith("KC"))
    mbons = sorted((b for b, (t, _) in neurons.items() if t.startswith("MBON")), key=lambda b: (neurons[b][0], b))
    cx = sorted((b for b, (t, _) in neurons.items() if t in CX_TYPES), key=lambda b: (CX_TYPES[neurons[b][0]], b))
    kc_i, mbon_i, cx_i = ({b: i for i, b in enumerate(x)} for x in (kcs, mbons, cx))
    pn_kc, kc_mbon, pam, ppl1 = {}, {}, np.zeros(len(mbons)), np.zeros(len(mbons))
    cx_w = np.zeros((len(cx), len(cx)), np.float32)
    with open(src / "traced-total-connections.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a, b, w = int(r["bodyId_pre"]), int(r["bodyId_post"]), int(r["weight"])
            ta = neurons.get(a, ("", ""))[0]
            if b in kc_i and PN_TYPE.match(ta) and w >= MIN_PN_KC:
                pn_kc[(kc_i[b], ta)] = pn_kc.get((kc_i[b], ta), 0) + w
            elif b in mbon_i:
                if a in kc_i:
                    kc_mbon[(mbon_i[b], kc_i[a])] = w
                elif ta.startswith("PAM"):
                    pam[mbon_i[b]] += w
                elif ta.startswith("PPL1"):
                    ppl1[mbon_i[b]] += w
            if a in cx_i and b in cx_i:
                cx_w[cx_i[b], cx_i[a]] += w
    pn_types = sorted({t for _, t in pn_kc})
    pt = {t: i for i, t in enumerate(pn_types)}
    (pk_row, pk_col), pk_w = zip(*((k, t) for k, t in ((k, pt[t]) for k, t in pn_kc))), list(pn_kc.values())
    km = sorted(kc_mbon.items())
    km_row = np.array([m for (m, _), _ in km], np.int32)
    km_w0 = np.array([w for _, w in km], np.float32)          # raw synapse counts: an MBON weighs what it receives
    dan = pam + ppl1
    safe = np.where(dan > 0, dan, 1)
    glom = [_glomeruli(neurons[b][1]) for b in cx]
    theta = [math.atan2(sum(math.sin(_theta(g)) for g in gs), sum(math.cos(_theta(g)) for g in gs)) for gs in glom]
    side = [neurons[b][1].rsplit("_", 1)[1][0] if CX_TYPES[neurons[b][0]] == "Delta7" else gs[0][0]
            for b, gs in zip(cx, glom)]
    meta = {"dataset": "hemibrain:v1.2", "url": ARCHIVE_URL, "size": ARCHIVE_SIZE, "min_pn_kc": MIN_PN_KC,
            "attribution": ATTRIBUTION}
    return {
        "pn_types": np.array(pn_types), "pn_kc_row": np.array(pk_row, np.int32), "pn_kc_col": np.array(pk_col, np.int32),
        "pn_kc_w": np.array(pk_w, np.float32), "n_kc": np.array(len(kcs)),
        "kc_types": np.array([neurons[b][0] for b in kcs]),
        "kc_mbon_row": km_row, "kc_mbon_col": np.array([k for (_, k), _ in km], np.int32), "kc_mbon_w0": km_w0,
        "mbon_types": np.array([neurons[b][0] for b in mbons]),
        # hemibrain traces the right mushroom body in full; left-side MBONs are cut at the volume's edge
        "mbon_side": np.array([neurons[b][1].rsplit("_", 1)[-1][:1] for b in mbons]),
        "mbon_sign": ((ppl1 - pam) / safe).astype(np.float32), "mbon_pam_frac": (pam / safe).astype(np.float32),
        "mbon_ppl1_frac": (ppl1 / safe).astype(np.float32),
        "cx_types": np.array([CX_TYPES[neurons[b][0]] for b in cx]), "cx_side": np.array(side),
        "cx_theta": np.array(theta, np.float32), "cx_w": cx_w, "meta": np.array([json.dumps(meta)]),
    }


def main():
    DATA_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(DATA_PATH, **derive(fetch()))
    print(f"wrote {DATA_PATH} ({DATA_PATH.stat().st_size / 1024:.0f} KB)")


def selfcheck(path=DATA_PATH):
    d = np.load(path, allow_pickle=False)
    assert int(d["n_kc"]) == len(d["kc_types"]) == 1927, int(d["n_kc"])
    assert len(d["mbon_types"]) == 68, len(d["mbon_types"])
    count = {t: int((d["cx_types"] == t).sum()) for t in ("EPG", "EPGt", "PEN1", "PEN2", "Delta7", "PEG")}
    assert count == {"EPG": 46, "EPGt": 4, "PEN1": 20, "PEN2": 22, "Delta7": 42, "PEG": 18}, count
    per_kc = np.bincount(d["pn_kc_row"], minlength=1927)
    assert np.median(per_kc[per_kc > 0]) == 5, np.median(per_kc[per_kc > 0])
    assert (d["kc_mbon_w0"] >= 1).all() and (d["kc_mbon_w0"] == np.round(d["kc_mbon_w0"])).all()   # raw counts
    right = d["mbon_side"] == "R"                              # the fully traced mushroom body
    sign = dict(zip(d["mbon_types"][right].tolist(), d["mbon_sign"][right].tolist()))
    assert all(sign[m] < 0 for m in ("MBON01", "MBON02", "MBON03")), sign
    assert all(sign[m] > 0 for m in ("MBON11", "MBON14")), sign
    assert np.isfinite(d["cx_theta"]).all()
    epg = np.sort(d["cx_theta"][d["cx_types"] == "EPG"] % (2 * math.pi))
    gaps = np.diff(np.concatenate([epg, epg[:1] + 2 * math.pi]))
    assert gaps.max() <= math.radians(45) + 1e-4, math.degrees(gaps.max())   # 8 tiles, L and R EPGs per tile
    assert json.loads(str(d["meta"][0]))["attribution"] == ATTRIBUTION
    with tempfile.TemporaryDirectory() as tmp:                 # wrong_size_refused: a truncated archive
        (Path(tmp) / ARCHIVE_URL.rsplit("/", 1)[1]).write_bytes(b"0123456789")
        try:
            fetch(tmp)
            raise AssertionError("a 10-byte archive was accepted")
        except ValueError:
            pass
    print("hemibrain self-check OK", count, f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv else main()
