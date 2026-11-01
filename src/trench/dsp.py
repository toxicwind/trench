"""Small DSP toolbox (numpy/scipy only, CPU)."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

SR = 44100


def tt(n: int) -> np.ndarray:
    return np.arange(n) / SR


def expenv(n: int, tau: float) -> np.ndarray:
    return np.exp(-np.arange(n) / SR / tau)


def phase(freq: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.cumsum(freq) / SR


def _sos(kind, fc, order):
    return ss.butter(order, fc, kind, fs=SR, output="sos")


def lp(x, fc, order=2):
    return ss.sosfilt(_sos("low", fc, order), x, axis=-1)


def hp(x, fc, order=2):
    return ss.sosfilt(_sos("high", fc, order), x, axis=-1)


def bp(x, lo, hi, order=2):
    return ss.sosfilt(_sos("band", [lo, hi], order), x, axis=-1)


def drive(x, d):
    return np.tanh(d * x) / np.tanh(d)


def fold(x, k):
    return np.sin(k * x)


def freq_shift(x, hz):
    """Single-sideband frequency shifter (Hilbert). Output is inharmonic."""
    a = ss.hilbert(x)
    return np.real(a * np.exp(2j * np.pi * hz * np.arange(len(x)) / SR))


def comb(x, ms, fb):
    d = max(1, int(ms * 1e-3 * SR))
    den = np.zeros(d + 1)
    den[0] = 1.0
    den[d] = -fb
    return ss.lfilter([1.0], den, x)


def reverb(x, sec, mix, seed=1):
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (sec * SR / 5))
    ir = lp(ir, 5000)
    wet = ss.fftconvolve(x, ir)[: len(x)]
    wet *= np.sqrt(np.mean(x**2) + 1e-12) / (np.sqrt(np.mean(wet**2)) + 1e-12)
    return (1 - mix) * x + mix * wet


def fade(n, a=0.003, r=0.02):
    e = np.ones(n)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    e[:na] = np.linspace(0, 1, na)
    e[n - nr:] = np.minimum(e[n - nr:], np.linspace(1, 0, nr))
    return e


def place(buf: np.ndarray, sig: np.ndarray, start: int, gain=1.0):
    """Add sig into buf (1-D) at start sample, clipped to bounds."""
    if start >= len(buf) or start + len(sig) <= 0:
        return
    s0 = max(0, -start)
    e0 = min(len(sig), len(buf) - start)
    buf[start + s0:start + e0] += gain * sig[s0:e0].astype(buf.dtype)


def normalize(x, peak=0.9):
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)
