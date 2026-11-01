# trench — 2:00 procedural arrangement (140 BPM half-time, 70 bars, two different drops)

Run: `pip install -r requirements.txt && make all`  (CPU only, ~20 s; needs ffmpeg only for the MP3).
Language note: Python (numpy/scipy) instead of Bun/Rust because the DSP + loudness stack (scipy, pyloudnorm) is the ecosystem for this.

Structure (bars): intro 1-8 | build 9-12 | DROP 1 13-28 | break 29-36 | build 37-40 | DROP 2 41-60 | outro 61-70
- Drop 1: sparse, sub-led. Sine sub on F1 with slow off-grid micro-bends + inharmonic FM -> sine-fold -> swept 3-band mid bass,
  plus a Hilbert single-sideband frequency-shifter layer. Half-time kick/clap.
- Drop 2: denser. 808-style sub slides + detuned-saw/hard-clip/comb bass, 16th hats, 3+3+2 cowbell hook.
- Break: bells + a Quadratic Difference Tone Spectrum carrier comb (arXiv 2609.10913). Phantom bass is perceptual: needs loudspeakers.

Sourcing: Infekt (MusicRadar / Beatportal interviews): sound design first, sub-bass focus, "less focused on rhythm".
The 3+3+2 grouping, F tonic, shifter amount etc. are design choices, NOT claims about any artist's method.

Verify: `make test` checks duration, finite/no clipping, drop-vs-break loudness, timbral difference of the drops,
sub energy, mono sub, LUFS window, anti-squash crest factor, -0.9 dBFS ceiling.
Samples: see samples/README.md. No secrets are used; .env.example documents the optional env vars.
