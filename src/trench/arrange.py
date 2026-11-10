"""70 bars @ 140 BPM half-time = exactly 120.0 s. Two different drops."""
from __future__ import annotations

import numpy as np

from . import synth as sy
from .dsp import SR, hp, place, reverb
from .samples import repitch

BPM = 140
BEAT = SR * 60 // BPM          # 18900 samples
STEP = BEAT // 4               # sixteenth
BAR = BEAT * 4
NB = 70
N = BAR * NB
SECTIONS = [("intro", 1, 8), ("build1", 9, 12), ("drop1", 13, 28), ("break", 29, 36),
            ("build2", 37, 40), ("drop2", 41, 60), ("outro", 61, 70)]


def pos(bar, step, swing=0.0):
    return (bar - 1) * BAR + int(step * STEP + (swing * STEP if step % 2 else 0))


class Ctx:
    def __init__(self, seed, samples):
        self.rng = np.random.default_rng(seed)
        self.S = samples
        z = lambda *s: np.zeros(s, np.float32)
        self.kick, self.sub, self.bassA, self.bassB = z(N), z(N), z(N), z(N)
        self.drums, self.music = z(2, N), z(2, N)
        self.ktrig: list[tuple[int, float]] = []
        self._k = [sy.kick(self.rng)]
        self._sn, self._hc, self._ho = sy.snare(self.rng), sy.hat(self.rng), sy.hat(self.rng, True)

    def pick(self, role):
        lst = self.S.get(role) or []
        return lst[self.rng.integers(len(lst))] if lst else None

    # --- drum hits -------------------------------------------------
    def k(self, p, v=1.0):
        s = self.pick("kick")
        y = self._k[0] * (0.6 if s else 1.0)
        if s:
            y = y.copy()
            y[: len(s[1])] += 0.9 * s[1][: len(y)]
        place(self.kick, y, p, v)
        self.ktrig.append((p, v))

    def sn(self, p, v=1.0):
        s = self.pick("snare")
        y = s[1] if s else self._sn
        place(self.drums[0], y, p, 0.8 * v)
        place(self.drums[1], y, p, 0.8 * v)

    def h(self, p, v=0.3, open_=False, pan=0.5):
        s = self.pick("hat") if not open_ else None
        y = s[1][: int(0.12 * SR)] if s else (self._ho if open_ else self._hc)
        place(self.drums[0], y, p, v * (1 - pan) * 2)
        place(self.drums[1], y, p, v * pan * 2)

    def cow(self, p, semi, v=0.35):
        s = self.pick("cowbell")
        if s:
            tgt = 65 + semi
            y = repitch(s[1], tgt - s[2] if s[2] else semi)[: int(0.6 * SR)]
        else:
            y = sy.cowbell(sy.hz(36 + semi))
        place(self.music[0], y, p, v * 0.8)
        place(self.music[1], y, p, v * 1.0)

    def bells(self, bar, chord, arp=True, v=0.4, step0=0, gap=2):
        for i, sm in enumerate(chord):
            p = pos(bar, step0 + (i * gap if arp else 0))
            y = sy.bell(sy.hz(sm + 24))
            place(self.music[0], y, p, v * (1.0 - 0.1 * i))
            place(self.music[1], y, p + 300 * i, v)

    # --- bass ------------------------------------------------------
    def bass(self, bar, events, style, vsub=0.9, vmid=0.5):
        for ev in events:
            st, ln, semi = ev[:3]
            glide = ev[3] if len(ev) > 3 else None
            n = max(STEP, ln * STEP - 400)
            p = pos(bar, st)
            f = sy.hz(semi)
            place(self.sub, sy.sub(f, n, sy.hz(glide) if glide is not None else None, seed=bar), p, vsub)
            if style == "wonk":
                place(self.bassA, sy.wonk(f, n, self.rng), p, vmid)
            elif style == "saw":
                place(self.bassB, sy.saw_stack(f, n, self.rng), p, vmid)


A = [(0, 5, 0), (6, 2, 0), (8, 3, 3), (12, 4, 0)]
B = [(0, 3, 0), (4, 2, 7), (6, 2, 5), (8, 3, 3), (11, 5, 0)]
D2A = [(0, 3, 0), (3, 3, 0), (6, 2, 3, 0), (8, 3, 0), (11, 3, 5, 3), (14, 2, 3)]
D2B = [(0, 3, 0), (3, 3, 12), (6, 2, 10), (8, 3, 7), (11, 3, 5), (14, 2, 3)]
CH1, CH2 = [12, 15, 19, 22], [8, 12, 15, 19]   # F-Ab-C-Eb / Db-F-Ab-C colour tones
HOOK = [0, 3, 7, 3, 10, 7]
HOOK_STEPS = [0, 3, 6, 8, 11, 14]               # 3+3+2 | 3+3+2 grouping


def build(seed: int, samples: dict) -> Ctx:
    c = Ctx(seed, samples)
    sw = 0.12
    # ---- intro (1-8)
    for b in range(1, 9):
        if b % 2:
            c.bells(b, CH1 if b % 4 == 1 else CH2, v=0.35, gap=3)
        if b >= 3:
            for s in range(0, 16, 2):
                c.h(pos(b, s, sw), 0.10 + 0.05 * (s % 4 == 2), pan=0.3 + 0.4 * (s % 4 == 2))
        if b >= 5:
            c.bass(b, [(0, 16, 0)], None, vsub=0.45 + 0.05 * (b - 5))
        if b >= 7:
            c.k(pos(b, 0), 0.8)
            c.k(pos(b, 10), 0.7)
    # ---- builds (9-12, 37-40)
    for start in (9, 37):
        for i in range(4):
            b = start + i
            div = (4, 8, 16, 32)[i]
            steps = np.arange(div) * (16 / div)
            for j, s in enumerate(steps):
                if i == 3 and s >= 14:
                    break
                v = 0.25 + 0.75 * ((i * 16 + j * 16 / div) / 64)
                c.sn(pos(b, 0) + int(s * STEP), v)
            c.k(pos(b, 0), 0.9)
            if i == 3:
                c.k(pos(b, 8), 0.9)
            c.bass(b, [(0, 16, 0, 12)] if i < 3 else [(0, 14, 12)], None, vsub=0.5)
        r = sy.riser(4 * BAR, c.rng)
        place(c.music[0], r, pos(start, 0), 0.55)
        place(c.music[1], r[::-1].copy()[::-1], pos(start, 0), 0.55)
    # ---- drop 1 (13-28): deep trench, sparse, flowing
    imp = c.pick("impact")
    place(c.drums[0], imp[1] if imp else sy.impact(c.rng), pos(13, 0), 0.7)
    place(c.drums[1], imp[1] if imp else sy.impact(c.rng), pos(13, 0), 0.7)
    for b in range(13, 29):
        v = (b - 13) % 4
        ks = [0, 6, 10, 13] if v == 3 else [0, 10]
        for s in ks:
            c.k(pos(b, s))
        c.sn(pos(b, 8))
        for s in (2, 6, 10, 14):
            c.h(pos(b, s, sw), 0.22 if s != 14 else 0.18, pan=0.25 + 0.5 * (s % 8 == 6))
        if v in (1, 3):
            c.h(pos(b, 14, sw), 0.3, open_=True, pan=0.7)
        c.bass(b, B if v == 3 else A, "wonk", 0.95, 0.45)
    # ---- break (29-36): phonk bells + phantom-sub bed
    for b in range(29, 37):
        if b % 2:
            c.bells(b, CH1 if (b // 2) % 2 else CH2, v=0.42, gap=2)
        if b >= 33:
            for s, sm in zip(HOOK_STEPS[:3], HOOK[:3]):
                c.cow(pos(b, s), sm, 0.18 + 0.03 * (b - 33))
        c.bass(b, [(0, 16, 0)], None, vsub=0.35)
    q = sy.qdts_bed(sy.hz(12), 8 * BAR)
    place(c.music[0], q, pos(29, 0), 0.07)
    place(c.music[1], q, pos(29, 0), 0.07)
    y = c.pick("yah")
    for b in (28, 36, 40):
        if y:
            place(c.music[0], y[1], pos(b, 12), 0.6)
            place(c.music[1], y[1], pos(b, 12), 0.6)
    # ---- drop 2 (41-60): phonk-cowbell + saw bass, syncopated, busier
    imp = c.pick("impact")
    place(c.drums[0], imp[1] if imp else sy.impact(c.rng), pos(41, 0), 0.8)
    place(c.drums[1], imp[1] if imp else sy.impact(c.rng), pos(41, 0), 0.8)
    for b in range(41, 61):
        v = (b - 41) % 4
        second = b >= 51
        for s in ([0, 3, 10] if v != 3 else [0, 3, 6, 10, 14]):
            c.k(pos(b, s))
        c.sn(pos(b, 8))
        if v == 3:
            c.sn(pos(b, 15), 0.5)
        for s in range(16):
            vel = 0.20 if s % 2 else 0.14
            if second and v == 3 and s >= 12:
                vel = 0.28
            c.h(pos(b, s, sw), vel, pan=0.2 + 0.6 * (s % 2))
        c.bass(b, (D2B if second and v % 2 else D2A), "saw", 0.9, 0.55)
        for s, sm in zip(HOOK_STEPS, HOOK):
            c.cow(pos(b, s), sm + (12 if second and v == 2 else 0), 0.3)
    # ---- outro (61-70)
    for b in range(61, 71):
        if b <= 64:
            c.k(pos(b, 0))
            c.k(pos(b, 10), 0.8)
            c.sn(pos(b, 8), 0.7)
            c.bass(b, D2A[:4], None, 0.85)
        else:
            c.bass(b, [(0, 16, 0)], None, 0.6 - 0.07 * (b - 65))
            if b <= 66:
                c.k(pos(b, 0), 0.7)
        if b % 2:
            c.bells(b, CH1 if b % 4 == 1 else CH2, v=0.35, gap=3)
    # reverb on the musical bus (two decorrelated channels)
    c.music[0] = reverb(c.music[0].astype(np.float64), 2.4, 0.33, 11).astype(np.float32)
    c.music[1] = reverb(c.music[1].astype(np.float64), 2.4, 0.33, 12).astype(np.float32)
    return c


def keep_highs(x):  # helper for side channel
    return hp(x, 180)
