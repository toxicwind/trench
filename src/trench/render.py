"""Mix + master + export + plots.  python -m trench.render --out out --samples samples"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from . import arrange as ar
from .dsp import SR, freq_shift, hp
from .samples import load_dir


def duck_env(trig, depth, rel=0.11):
    env = np.zeros(ar.N, np.float32)
    n = int(0.6 * SR)
    shape = np.exp(-np.arange(n) / SR / rel).astype(np.float32)
    for p, v in trig:
        e = min(n, ar.N - p)
        np.maximum(env[p:p + e], v * shape[:e], out=env[p:p + e])
    return 1.0 - depth * env


def mix(c: ar.Ctx):
    d_sub, d_bass, d_mus = duck_env(c.ktrig, 0.8), duck_env(c.ktrig, 0.65), duck_env(c.ktrig, 0.35)
    sub = c.sub * d_sub
    a = c.bassA.astype(np.float64)
    a = 0.8 * a + 0.28 * freq_shift(a, 63.0)          # inharmonic SSB layer on drop-1 bass
    bass = (a.astype(np.float32) * 0.85 + c.bassB * 0.85) * d_bass
    side = hp(np.roll(bass, int(0.007 * SR)).astype(np.float64), 180).astype(np.float32) * 0.6
    mono_low = c.kick * 0.62 + sub * 0.55
    dr, mu = 0.6, 0.9
    L = mono_low + bass + side + dr * c.drums[0] + mu * c.music[0] * d_mus
    R = mono_low + bass - side + dr * c.drums[1] + mu * c.music[1] * d_mus
    return np.stack([L, R]).astype(np.float64)


def limiter(y, ceil=0.891, look=0.005, rel=0.09, blk=44):
    """Lookahead peak limiter: block-wise gain computer, min-filter attack, exp release."""
    from scipy.ndimage import minimum_filter1d
    pk = np.abs(y).max(0)
    nb = len(pk) // blk
    g = np.minimum(1.0, ceil / np.maximum(pk[: nb * blk].reshape(nb, blk).max(1), 1e-9))
    g = minimum_filter1d(g, size=2 * max(1, int(look * SR / blk)) + 1)
    a = np.exp(-blk / (rel * SR))
    gs = np.empty_like(g)
    cur = 1.0
    for i, v in enumerate(g):
        cur = v if v < cur else min(v, cur + (1 - cur) * (1 - a))
        gs[i] = cur
    gs = np.minimum(gs, g)
    full = np.interp(np.arange(len(pk)), (np.arange(nb) + 0.5) * blk, gs)
    return y * full


def master(y, target=-11.0):
    y = hp(y, 22)
    m = pyln.Meter(SR)
    for _ in range(3):
        y = y * 10 ** ((target - m.integrated_loudness(y.T)) / 20)
        y = limiter(y)
    t = np.arange(y.shape[1]) / SR
    y = y * (np.clip((t[-1] - t) / 4.0, 0, 1) ** 2 * np.clip(t / 0.05, 0, 1))
    return np.clip(y, -0.891, 0.891)


def plots(y, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    mono = y.mean(0)
    ax[0].specgram(mono, NFFT=4096, Fs=SR, noverlap=3072, cmap="magma", vmin=-130, vmax=-30)
    ax[0].set_ylim(0, 8000); ax[0].set_title("trench_flow — spectrogram")
    hop = SR // 2
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + hop] ** 2)) + 1e-9) for i in range(0, len(mono) - hop, hop)]
    ax[1].plot(np.arange(len(rms)) / 2, rms, lw=1)
    for name, a, b in ar.SECTIONS:
        s, e = (a - 1) * ar.BAR / SR, b * ar.BAR / SR
        ax[1].axvspan(s, e, alpha=0.12, color=("C3" if name.startswith("drop") else "C0"))
        ax[1].text((s + e) / 2, -8, name, ha="center", fontsize=8)
    ax[1].set_ylim(-45, 0); ax[1].set_xlim(0, 120); ax[1].set_ylabel("RMS dBFS")
    fig.tight_layout(); fig.savefig(out / "trench_flow_spectrogram.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.getenv("OUT_DIR", "out"))
    ap.add_argument("--samples", default=os.getenv("SAMPLES_DIR", "samples"))
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "7")))
    ap.add_argument("--lufs", type=float, default=float(os.getenv("TARGET_LUFS", "-11")))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = load_dir(a.samples)
    used = {k: [n for n, *_ in v] for k, v in S.items() if v}
    c = ar.build(a.seed, S)
    y = master(mix(c), a.lufs)
    wav = out / "trench_flow.wav"
    sf.write(wav, y.T, SR, subtype="PCM_24")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "320k", str(out / "trench_flow.mp3")], check=False)
    plots(y, out)
    m = pyln.Meter(SR)
    rep = {"duration_s": y.shape[1] / SR, "bpm": ar.BPM, "bars": ar.NB, "lufs": m.integrated_loudness(y.T),
           "peak_dbfs": float(20 * np.log10(np.max(np.abs(y)))), "samples_used": used or "none (synth stand-ins)",
           "sections": {n: [a_, b_] for n, a_, b_ in ar.SECTIONS}}
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
