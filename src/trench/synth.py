"""Voices: drums, sub, two mid-bass designs, bells, cowbell, risers, QDTS bed."""
from __future__ import annotations

import numpy as np

from .dsp import SR, bp, comb, drive, expenv, fade, fold, hp, lp, phase, tt

F1 = 43.654  # F1 in Hz; the whole track sits on F


def hz(semi: float, base=F1) -> float:
    return base * 2 ** (semi / 12)


def kick(rng):
    n = int(0.5 * SR)
    t = tt(n)
    f = 44 + 140 * np.exp(-t / 0.028)
    y = np.sin(phase(f)) * expenv(n, 0.2)
    click = hp(rng.standard_normal(n), 2500) * expenv(n, 0.003) * 0.35
    return drive(y + click, 1.8) * fade(n, 0.0005, 0.02)


def snare(rng):
    n = int(0.35 * SR)
    t = tt(n)
    noise = bp(rng.standard_normal(n), 1500, 7500) * expenv(n, 0.07)
    body = np.sin(2 * np.pi * 185 * t) * expenv(n, 0.05) * 0.5
    clap = np.zeros(n)
    for ms in (0, 9, 18, 28):
        s = int(ms * 1e-3 * SR)
        clap[s:] += bp(rng.standard_normal(n - s), 1000, 3500) * expenv(n - s, 0.012 if ms < 28 else 0.1)
    return drive(0.7 * noise + body + 0.8 * clap, 1.6) * fade(n, 0.0003, 0.03)


def hat(rng, open_=False):
    n = int((0.22 if open_ else 0.06) * SR)
    y = hp(rng.standard_normal(n), 7000) * expenv(n, 0.07 if open_ else 0.014)
    return y * fade(n, 0.0002, 0.01)


def impact(rng):
    n = int(2.2 * SR)
    t = tt(n)
    y = np.sin(2 * np.pi * (36 + 60 * np.exp(-t / 0.08)) * t) * expenv(n, 0.8)
    y += lp(rng.standard_normal(n), 1800) * expenv(n, 0.25) * 0.25
    return drive(y, 1.4) * fade(n, 0.001, 0.2)


def sub(f, n, glide_to=None, glide_t=0.06, wob=14.0, seed=0):
    t = tt(n)
    fr = np.full(n, float(f))
    if glide_to:
        g = min(n, int(glide_t * SR))
        ramp = np.ones(n)
        ramp[:g] = np.linspace(0, 1, g)
        fr = f + (glide_to - f) * ramp
    cents = wob * np.sin(2 * np.pi * 0.9 * t + seed) * np.minimum(1, t / 0.2)  # slow micro-bend, stays off scale
    ph = phase(fr * 2 ** (cents / 1200))
    y = np.sin(ph) + 0.10 * np.sin(2 * ph + 0.6)
    return drive(y, 1.25) * fade(n, 0.004, 0.025)


def wonk(f, n, rng, fshift=0.0):
    """Drop-1 mid bass: inharmonic FM -> sine fold -> swept 3-band morph."""
    t = tt(n)
    f0 = f * 2
    ph = phase(np.full(n, f0))
    idx = 0.6 + 5.0 * (t / max(t[-1], 1e-3)) ** 1.5
    sweep_up = rng.random() < 0.5
    y = np.sin(ph + idx * np.sin(1.5 * ph))
    y = fold(y * 1.8, 1.6)
    s = np.linspace(0, 1, n)
    s = s if sweep_up else s[::-1]
    low, mid, high = lp(y, 320), bp(y, 300, 1000), bp(y, 900, 3200)
    y = low * (1 - s) + mid * np.sin(np.pi * s) + high * s
    return drive(y, 3.0) * fade(n, 0.003, 0.02)


def saw_stack(f, n, rng):
    """Drop-2 mid bass: detuned saws -> hard clip -> metallic comb."""
    y = np.zeros(n)
    for c in (-13, 0, 11):
        ph = phase(np.full(n, f * 2 * 2 ** (c / 1200)))
        y += 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    y = np.clip(y * 1.2, -1, 1)
    y = bp(y, 120, 2200)
    y = comb(y, 3.7, 0.45)
    return drive(y, 2.2) * fade(n, 0.003, 0.02) * 0.6


def cowbell(f, n_s=0.22):
    n = int(n_s * SR)
    y = np.sign(np.sin(phase(np.full(n, f)))) + np.sign(np.sin(phase(np.full(n, f * 1.4835))))
    return bp(y, 500, 3500) * expenv(n, 0.07) * fade(n, 0.0005, 0.02) * 0.5


def bell(f, n_s=2.6):
    n = int(n_s * SR)
    t = tt(n)
    y = np.sin(2 * np.pi * f * t) * expenv(n, 1.0)
    y += 0.5 * np.sin(2 * np.pi * f * 3.51 * t) * expenv(n, 0.45)
    y += 0.25 * np.sin(2 * np.pi * f * 5.43 * t) * expenv(n, 0.2)
    return y * fade(n, 0.002, 0.3) * 0.5


def riser(n, rng):
    t = tt(n)
    x = t / t[-1]
    noise = hp(rng.standard_normal(n), 1500) * x**2
    sweep = np.sin(phase(150 + 2400 * x**2)) * x**2 * 0.4
    return (noise * 0.5 + sweep) * fade(n, 0.01, 0.01)


def qdts_bed(F, n, C=2000.0, N=9):
    """Quadratic-difference-tone carrier complex (Haworth 2011 / arXiv 2609.10913).
    Carriers at C + k*F make the ear synthesise phantom F, 2F, 3F... Needs loudspeakers."""
    t = tt(n)
    y = sum(np.sin(2 * np.pi * (C + k * F) * t + 0.9 * k) for k in range(N + 1)) / (N + 1)
    return y * fade(n, 0.5, 0.5)
