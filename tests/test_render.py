import itertools
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import scipy.signal as ss
import soundfile as sf

from trench import arrange as ar

WAV = Path(__file__).parent.parent / "out" / "trench_flow.wav"
SR = 44100


@pytest.fixture(scope="module")
def y():
    x, sr = sf.read(WAV)
    assert sr == SR
    return x.T


def seg(y, a, b):
    return y[:, (a - 1) * ar.BAR: b * ar.BAR]


def lufs(x):
    return pyln.Meter(SR).integrated_loudness(x.T)


def bands(x):
    f, p = ss.welch(x.mean(0), SR, nperseg=8192)
    edges = [20, 40, 80, 160, 320, 640, 1300, 2600, 5200, 10000]
    v = np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in itertools.pairwise(edges)])
    return v / v.sum()


def test_exact_two_minutes(y):
    assert abs(y.shape[1] / SR - 120.0) < 0.01


def test_finite_and_no_clipping(y):
    assert np.isfinite(y).all()
    assert np.abs(y).max() <= 0.9
    assert (np.abs(y) > 0.999).sum() == 0


def test_two_drops_loud_vs_rest(y):
    d1, d2 = lufs(seg(y, 13, 28)), lufs(seg(y, 41, 60))
    assert d1 > lufs(seg(y, 1, 8)) + 3
    assert d1 > lufs(seg(y, 29, 36)) + 3
    assert d2 > lufs(seg(y, 29, 36)) + 3


def test_drops_are_timbrally_different(y):
    b1, b2 = bands(seg(y, 13, 28)), bands(seg(y, 41, 60))
    cos = b1 @ b2 / (np.linalg.norm(b1) * np.linalg.norm(b2))
    assert np.abs(np.log10(b1 + 1e-9) - np.log10(b2 + 1e-9)).mean() > 0.08 or cos < 0.995


def test_sub_energy_in_drops(y):
    def sub_share(x):
        f, p = ss.welch(x.mean(0), SR, nperseg=8192)
        return p[(f >= 30) & (f <= 80)].sum() / p.sum()
    assert sub_share(seg(y, 13, 28)) > sub_share(seg(y, 1, 4)) * 2
    assert sub_share(seg(y, 41, 60)) > 0.2


def test_sub_is_mono(y):
    from trench.dsp import lp
    low = lp(y, 120, 4)
    mid, side = (low[0] + low[1]) / 2, (low[0] - low[1]) / 2
    assert 10 * np.log10((side**2).mean() / (mid**2).mean() + 1e-12) < -25


def test_loudness_target(y):
    assert -13 < lufs(y) < -9


def test_drops_not_squashed(y):
    for a, b in ((13, 28), (41, 60)):
        x = seg(y, a, b)
        crest = 20 * np.log10(np.abs(x).max() / np.sqrt((x**2).mean()))
        assert crest > 6.5, f"crest {crest:.1f} dB — master is flat-topping"


def test_peak_ceiling(y):
    assert 20 * np.log10(np.abs(y).max()) <= -0.9
