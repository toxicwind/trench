"""Optional user-supplied WAV one-shots. Files are matched to roles by filename."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

from .dsp import SR

ROLES = {
    "kick": ("kick",),
    "snare": ("snare", "clap"),
    "hat": ("hat", "top"),
    "cowbell": ("cowbell", "cow"),
    "bell": ("bell",),
    "growl": ("growl", "grwl"),
    "yah": ("yah", "vox", "vocal"),
    "impact": ("pow", "thud", "thunk", "impact", "boom"),
}
NOTE = re.compile(r"(?<![A-Za-z])([A-G])(#?)(-?\d)(?!\d)")
SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def root_midi(name: str):
    m = NOTE.search(name)
    if not m:
        return None
    return 12 * (int(m.group(3)) + 1) + SEMI[m.group(1)] + (1 if m.group(2) else 0)


def load_dir(path) -> dict:
    out = {r: [] for r in ROLES}
    p = Path(path) if path else None
    if not p or not p.is_dir():
        return out
    for f in sorted(p.rglob("*.wav")):
        low = f.stem.lower()
        for role, keys in ROLES.items():
            if any(k in low for k in keys):
                x, sr = sf.read(f, always_2d=True)
                x = x.mean(1)
                if sr != SR:
                    g = np.gcd(sr, SR)
                    x = ss.resample_poly(x, SR // g, sr // g)
                if np.max(np.abs(x)) > 0:
                    out[role].append((f.stem, x / np.max(np.abs(x)), root_midi(f.stem)))
                break
    return out


def repitch(x, semis):
    r = 2 ** (semis / 12)
    n = int(len(x) / r)
    return np.interp(np.arange(n) * r, np.arange(len(x)), x)
