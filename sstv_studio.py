#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 EXPERIMENTAL SSTV STUDIO
 Time-compressed Slow Scan TV — cut on-air time, pay with CPU on both ends.
=============================================================================

The idea
--------
Encode an image as normal SSTV audio, then TIME-COMPRESS the audio before it
goes on air (2x, 3x, 4x ...). The receiving side records the short burst,
time-STRETCHES it back to the original rate, cleans it up with a little DSP,
and feeds it to a normal SSTV decoder. Half (or less) the airtime; the price
is a moment of number crunching on each end.

Two compression methods are provided because they behave very differently
over a radio channel:

  * "stretch"  — phase-vocoder time compression, PITCH IS PRESERVED. The
                 signal stays inside 1200–2300 Hz so it still fits a normal
                 SSB voice channel. Mild smearing artifacts appear; the
                 decoder's sync-regression + cleanup filtering handles them.
  * "resample" — classic tape-style speed-up. Nearly lossless round trip,
                 but every frequency is multiplied by the factor (2x puts
                 the signal at 2400–4600 Hz), so it needs a wideband channel
                 (FM, digital links, direct audio) — a 2.7 kHz SSB filter
                 will destroy it. Included for experiments and comparison.

Supported modes: Martin M1/M2, Scottie S1/S2, Robot 36/72 (full VIS headers).

Usage
-----
  python3 sstv_studio.py                        # launch the GUI studio
  python3 sstv_studio.py --selftest             # headless encode->decode QA
  python3 sstv_studio.py --encode pic.jpg --mode "Martin M1" \
                         --factor 2 --method stretch --out onair.wav
  python3 sstv_studio.py --decode onair.wav --out rx.png     # auto-detects
                                                             # factor+method

Dependencies: numpy, scipy, Pillow.  Optional: sounddevice (play / live RX).

This is an EXPERIMENT for the ham community: both ends need this tool (or an
external time-stretcher). It is not a standardized SSTV mode. Identify your
transmissions per your local regulations.
=============================================================================
"""

import argparse
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime

import numpy as np
from scipy import signal as sps
from scipy import ndimage as ndi
from scipy.fft import next_fast_len
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont

try:
    import sounddevice as sd
    HAVE_AUDIO = True
except Exception:
    sd = None
    HAVE_AUDIO = False

# ---------------------------------------------------------------------------
# Constants & mode tables
# ---------------------------------------------------------------------------
FS = 44100                      # internal / WAV sample rate
F_BLACK, F_WHITE = 1500.0, 2300.0
F_SPAN = F_WHITE - F_BLACK
F_SYNC = 1200.0

METHOD_RESAMPLE = "resample"
METHOD_FM = "fm"
FACTORS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
RATIONAL = {1.0: (1, 1), 1.25: (5, 4), 1.5: (3, 2), 5.0: (5, 1), 6.0: (6, 1), 7.0: (7, 1), 8.0: (8, 1),
            2.0: (2, 1), 3.0: (3, 1), 4.0: (4, 1)}

# Timings from the classic N7CXI "SSTV Mode Specifications" (seconds).
MODES = {
    "Martin M1": dict(family="martin", vis=44, w=320, h=256,
                      scan=0.146432, sync=0.004862, porch=0.000572,
                      sep=0.000572),
    "Martin M2": dict(family="martin", vis=40, w=320, h=256,
                      scan=0.073216, sync=0.004862, porch=0.000572,
                      sep=0.000572),
    "Scottie S1": dict(family="scottie", vis=60, w=320, h=256,
                       scan=0.138240, sync=0.009, porch=0.0015, sep=0.0015),
    "Scottie S2": dict(family="scottie", vis=56, w=320, h=256,
                       scan=0.088064, sync=0.009, porch=0.0015, sep=0.0015),
    "Robot 36": dict(family="robot36", vis=8, w=320, h=240,
                     y_scan=0.088, c_scan=0.044, sync=0.009, porch=0.003,
                     sep=0.0045, sep_porch=0.0015),
    "Robot 72": dict(family="robot72", vis=12, w=320, h=240,
                     y_scan=0.138, c_scan=0.069, sync=0.009, porch=0.003,
                     sep=0.0045, sep_porch=0.0015),
}
VIS2MODE = {m["vis"]: name for name, m in MODES.items()}

VIS_LEADER = 0.300
VIS_BREAK = 0.010
VIS_BIT = 0.030
PAD = 0.250                     # silence before/after the burst


def line_period(mode):
    f = mode["family"]
    if f == "martin":
        return mode["sync"] + mode["porch"] + 3 * (mode["scan"] + mode["sep"])
    if f == "scottie":
        return 2 * mode["sep"] + mode["sync"] + mode["porch"] + 3 * mode["scan"]
    if f == "robot36":
        return (mode["sync"] + mode["porch"] + mode["y_scan"]
                + mode["sep"] + mode["sep_porch"] + mode["c_scan"])
    if f == "robot72":
        return (mode["sync"] + mode["porch"] + mode["y_scan"]
                + 2 * (mode["sep"] + mode["sep_porch"] + mode["c_scan"]))
    raise ValueError(f)


def vis_duration():
    return 2 * VIS_LEADER + VIS_BREAK + 10 * VIS_BIT


def nominal_duration(mode):
    d = vis_duration() + mode["h"] * line_period(mode)
    if mode["family"] == "scottie":
        d += mode["sync"]                      # Scottie "starting" sync
    return d


def sync_geometry(mode):
    """(offset of sync pulse start within a line, sync duration) in seconds."""
    f = mode["family"]
    if f == "scottie":
        return 2 * (mode["sep"] + mode["scan"]), mode["sync"]
    return 0.0, mode["sync"]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def to_int16(y):
    y = np.clip(np.asarray(y, np.float64), -1.0, 1.0)
    return (y * 32767.0).astype(np.int16)


def write_wav(path, y, fs=FS):
    wavfile.write(path, fs, to_int16(y))


def read_wav_any(path):
    """Read any WAV, return (mono float32 at FS, original_fs)."""
    fsr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    else:
        y = data.astype(np.float32)
    if fsr != FS:
        g = math.gcd(int(fsr), FS)
        y = sps.resample_poly(y, FS // g, int(fsr) // g).astype(np.float32)
    return y, fsr


def peak_norm(y, level=0.9):
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    if m < 1e-9:
        return y.astype(np.float32)
    return (y * (level / m)).astype(np.float32)


# ---------------------------------------------------------------------------
# DSP: filtering, channel simulation, resample & phase-vocoder stretch
# ---------------------------------------------------------------------------
def bandpass(y, fs, lo, hi, order=4):
    hi = min(hi, 0.45 * fs)
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def channel_simulate(y, fs=FS, bw_hz=None, snr_db=None, seed=None,
                     kind="ssb"):
    """Channel model. kind="ssb": bandpass 300 Hz..bw_hz plus in-band
    noise. kind="fm": a narrowband-FM voice path like an FRS/GMRS handheld
    or repeater — TX pre-emphasis (+6 dB/octave), a 300–3000 Hz audio
    shelf, additive noise, then RX de-emphasis, whose tilt turns flat noise
    into the low-frequency-heavy hiss FM receivers actually produce."""
    out = np.asarray(y, np.float64)
    if kind == "fm":
        # 6 dB/octave pre-emphasis around ~750 us time constant
        a = float(np.exp(-1.0 / (fs * 750e-6)))
        pre = sps.lfilter([1.0, -a], [1.0 - a], out)
        sos = sps.butter(4, [300.0, min(float(bw_hz or 3000.0), 3000.0)],
                         btype="bandpass", fs=fs, output="sos")
        ch = sps.sosfiltfilt(sos, pre)
        if snr_db is not None:
            rng = np.random.default_rng(seed)
            p = float(np.sqrt(np.mean(ch ** 2)) + 1e-12)
            n = rng.standard_normal(len(ch)) * p * (10 ** (-snr_db / 20.0))
            ch = ch + sps.sosfiltfilt(sos, n)
        out = sps.lfilter([1.0 - a], [1.0, -a], ch)   # de-emphasis
        return peak_norm(out, 0.95)
    sos = None
    if bw_hz:
        sos = sps.butter(6, [300.0, float(bw_hz)], btype="bandpass",
                         fs=fs, output="sos")
        out = sps.sosfiltfilt(sos, out)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
        noise = rng.standard_normal(len(out)) * p * (10 ** (-snr_db / 20.0))
        if sos is not None:                      # in-band noise, like a real RX
            noise = sps.sosfiltfilt(sos, noise)
        out = out + noise
    return peak_norm(out, 0.95)


def resample_rational(y, up, down):
    if up == down:
        return np.asarray(y, np.float32)
    return sps.resample_poly(np.asarray(y, np.float64), up, down).astype(np.float32)


def _hann(n):
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def _stft(x, n_fft, hop):
    x = np.asarray(x, np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    T = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(T)[:, None]
    frames = x[idx] * _hann(n_fft)[None, :]
    return np.fft.rfft(frames, axis=1).astype(np.complex64).T   # (bins, T)


def _istft(S, n_fft, hop):
    win = _hann(n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=0).real            # (n_fft, T)
    T = S.shape[1]
    out = np.zeros(n_fft + hop * (T - 1), np.float64)
    wsum = np.zeros_like(out)
    w2 = (win * win).astype(np.float64)
    wf = win.astype(np.float64)
    for i in range(T):
        a = i * hop
        out[a:a + n_fft] += frames[:, i] * wf
        wsum[a:a + n_fft] += w2
    out /= np.maximum(wsum, 1e-8)
    return out


def _burst_spectrum_probe(y, fs):
    """One FFT, two answers about the burst: (upper_edge, low_band_noisy).

    upper_edge — highest frequency still carrying real energy, i.e. where
    the transmitting channel's filter ended; demodulating wider than this
    recovers nothing and only admits hiss. low_band_noisy — whether the
    300–900 Hz region (below any SSTV carrier) contains channel noise; if
    it does, the demodulator should floor at 700 Hz, and if it is silent
    the low side can stay fully open to keep every sideband."""
    n = min(len(y), int(2.0 * fs))
    seg = np.asarray(y[len(y) // 2 - n // 2: len(y) // 2 + n // 2],
                     np.float64)
    if len(seg) < 4096:
        return 0.45 * fs, True
    win = np.hanning(len(seg))
    P = np.abs(np.fft.rfft(seg * win)) ** 2
    fr = np.fft.rfftfreq(len(seg), 1.0 / fs)
    ref = float(np.median(P[(fr > 1200) & (fr < 2400)]) + 1e-18)
    above = np.where((fr > 2000) & (P > ref / 300.0))[0]
    edge = float(fr[above[-1]]) if len(above) else 2600.0
    low = float(np.median(P[(fr > 350) & (fr < 800)]) + 1e-18)
    return edge, (low > ref / 100.0)


def fm_timescale(y, fs, up, down):
    """Time-scale an FM signal by resampling its *frequency envelope*.

    Demodulate to instantaneous frequency, resample that track by up/down
    (an exactly linear time base — no vocoder, no wander), then
    resynthesize a constant-amplitude carrier. Pitch is inherently
    preserved: the deviation range (1200–2300 Hz for SSTV) is untouched;
    only the modulation *rate* changes, which is equivalent to running the
    scan clock faster.

    The inner demodulator's bandwidth is factor-aware: a signal compressed
    k× has k× the modulation rate, so its sidebands are k× wider (Carson),
    and shaving them is exactly horizontal blur. The band opens with k, but
    never beyond the measured spectral edge of the input — demodulating
    past where the channel's filter cut off only admits noise. The burst is
    located by its amplitude envelope so silence pads scale without
    demodulating noise."""
    y = np.asarray(y, np.float64)
    n = len(y)
    env = np.abs(sps.hilbert(y, N=next_fast_len(n))[:n])
    env = ndi.uniform_filter1d(env, size=int(0.010 * fs), mode="nearest")
    thr = 0.12 * float(np.max(env) + 1e-12)
    on = np.where(env > thr)[0]
    if len(on) < int(0.05 * fs):
        return resample_rational(y, up, down)
    a = max(0, int(on[0] - 0.02 * fs))
    b = min(n, int(on[-1] + 0.02 * fs))

    k_in = max(1.0, up / down)              # input compression factor
    edge, low_noisy = _burst_spectrum_probe(y[a:b], fs)
    hi = min(1900.0 + 3600.0 * k_in, edge + 300.0, 0.45 * fs)
    # The carrier never goes below 1200 Hz. If the 300-900 Hz region is
    # silent, open the low side fully and keep every sideband; if it
    # carries channel noise, floor at 700 Hz to keep it out of the track.
    lo = 900.0 if low_noisy else 200.0
    med = 1 if (up > down and up >= 5 * down) else 3
    ft = FreqTrack(y[a:b], fs, median=med, lo=lo, hi=max(hi, 3000.0),
                   order=2)
    f2 = sps.resample_poly(ft.f, up, down)
    np.clip(f2, 400.0, 3500.0, out=f2)
    ph = 2.0 * np.pi * np.cumsum(f2) / fs
    burst = np.sin(ph) * 0.85
    nf = max(8, int(0.005 * fs))
    ramp = np.sin(0.5 * np.pi * np.arange(nf) / nf) ** 2
    burst[:nf] *= ramp
    burst[-nf:] *= ramp[::-1]
    pad0 = np.zeros(int(round(a * up / down)))
    pad1 = np.zeros(int(round((n - b) * up / down)))
    return np.concatenate([pad0, burst, pad1]).astype(np.float32)


def vox_preamble(fs, dur=0.7):
    """Keying header for VOX-triggered transmitters (FRS/GMRS handhelds,
    hands-free jacks): a steady 1900 Hz tone long enough for the VOX to
    key up and the squelch on the far end to open BEFORE the VIS leader
    starts, so the attack delay eats the sacrificial tone instead of the
    header. 1900 Hz is the leader frequency, so a decoder just sees a
    longer leader."""
    n = int(dur * fs)
    t = np.arange(n) / fs
    tone = 0.85 * np.sin(2.0 * np.pi * 1900.0 * t)
    nf = max(8, int(0.01 * fs))
    tone[:nf] *= np.sin(0.5 * np.pi * np.arange(nf) / nf) ** 2
    return tone.astype(np.float32)


def compress_audio(y, fs, factor, method):
    """Make it SHORTER by `factor` for transmission."""
    if factor <= 1.0 + 1e-9:
        return np.asarray(y, np.float32)
    if method == METHOD_RESAMPLE:
        num, den = RATIONAL[factor]
        return resample_rational(y, den, num)     # fewer samples, same fs
    num, den = RATIONAL[factor]                # METHOD_FM (default)
    return fm_timescale(y, fs, den, num)


def restore_audio(y, fs, factor, method):
    """Undo compress_audio on the RX side."""
    if factor <= 1.0 + 1e-9:
        return np.asarray(y, np.float32)
    if method == METHOD_RESAMPLE:
        num, den = RATIONAL[factor]
        return resample_rational(y, num, den)
    num, den = RATIONAL[factor]                # METHOD_FM (default)
    return fm_timescale(y, fs, num, den)


# ---------------------------------------------------------------------------
# Frequency demodulation
# ---------------------------------------------------------------------------
class FreqTrack:
    """Instantaneous-frequency track of an SSTV signal + fast window means."""

    def __init__(self, y, fs, median=5, lo=900.0, hi=2900.0, order=4):
        y = np.asarray(y, np.float64)
        y = bandpass(y, fs, lo, hi, order=order)
        n = len(y)
        a = sps.hilbert(np.asarray(y, np.float64), N=next_fast_len(n))[:n]
        # Instantaneous frequency from per-sample phase *differences*
        # (conjugate product). Never unwrap a long absolute phase: it grows
        # to ~1e6 rad over a frame and its floating-point quantization then
        # dwarfs the per-sample increment, quantizing the frequency track.
        f = np.empty(n, np.float64)
        f[:-1] = np.angle(a[1:] * np.conj(a[:-1])) * (fs / (2.0 * np.pi))
        f[-1] = f[-2] if n > 1 else 0.0
        np.clip(f, 200.0, 3600.0, out=f)
        if median and median >= 3:
            f = ndi.median_filter(f, size=median, mode="nearest")
        self.f = f.astype(np.float64)
        self.fs = fs
        self.n = n
        self.cs = np.concatenate([[0.0], np.cumsum(self.f)])

    def mean(self, a, b):
        ai = max(0, min(self.n, int(round(a))))
        bi = max(ai + 1, min(self.n, int(round(b))))
        if bi <= ai:
            return 0.0
        return (self.cs[bi] - self.cs[ai]) / (bi - ai)

    def scan(self, start, dur_samples, w, guard=0.18):
        """Mean frequency in each of `w` pixel windows -> array len w.

        `guard` trims each side of every pixel window (reads the central
        64 %), so transition smear from neighbouring pixels is excluded."""
        p = np.arange(w, dtype=np.float64)
        e0 = start + dur_samples * (p + guard) / w
        e1 = start + dur_samples * (p + 1.0 - guard) / w
        a = np.clip(np.rint(e0).astype(np.int64), 0, self.n)
        b = np.clip(np.rint(e1).astype(np.int64), 0, self.n)
        b = np.minimum(np.maximum(b, a + 1), self.n)
        a = np.minimum(a, b - 1)
        return (self.cs[b] - self.cs[a]) / np.maximum(b - a, 1)

    def coarse(self, step_s=0.001):
        step = max(1, int(round(step_s * self.fs)))
        m = self.n // step
        e = (np.arange(m + 1) * step)
        return (self.cs[e[1:]] - self.cs[e[:-1]]) / step, step


def freq_to_val(f):
    return np.clip((f - F_BLACK) * (255.0 / F_SPAN), 0.0, 255.0)


# ---------------------------------------------------------------------------
# Color math (BT.601 studio swing — the classic Robot mapping)
# ---------------------------------------------------------------------------
def rgb_to_ycrcb(arr):
    a = arr.astype(np.float64)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 16.0 + (65.738 * R + 129.057 * G + 25.064 * B) / 256.0
    Cr = 128.0 + (112.439 * R - 94.154 * G - 18.285 * B) / 256.0
    Cb = 128.0 + (-37.945 * R - 74.494 * G + 112.439 * B) / 256.0
    return Y, Cr, Cb


def ycrcb_to_rgb(Y, Cr, Cb):
    y = 1.16438 * (Y - 16.0)
    r = y + 1.59603 * (Cr - 128.0)
    g = y - 0.81297 * (Cr - 128.0) - 0.39176 * (Cb - 128.0)
    b = y + 2.01723 * (Cb - 128.0)
    out = np.stack([r, g, b], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def fit_image(img, w, h):
    """Aspect-preserving fit onto a black WxH canvas."""
    img = img.convert("RGB")
    s = min(w / img.width, h / img.height)
    nw, nh = max(1, int(img.width * s)), max(1, int(img.height * s))
    img = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(img, ((w - nw) // 2, (h - nh) // 2))
    return out


_CALLSIGN_FONTS = [
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "arialbd.ttf",            # Windows
    "segoeuib.ttf",           # Windows (Segoe UI Bold)
    "verdanab.ttf",
    "Arial Bold.ttf",         # macOS
    "LiberationSans-Bold.ttf",
]


def _callsign_font(size):
    for name in _CALLSIGN_FONTS:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return None


def overlay_callsign(img, text):
    """Callsign overlay, top-left, on a darkened strip. Uses a proper
    anti-aliased bold system font with a dark outline when one is
    available; otherwise falls back to the retro pixel font rendered at
    3x and downscaled, so even the fallback has smooth edges instead of
    hard cubes."""
    text = (text or "").strip().upper()
    if not text:
        return img
    img = img.copy()
    w, h = img.width, img.height

    size = max(14, int(h * 0.105))
    font = _callsign_font(size)
    if font is not None:
        # shrink to fit the image width
        while size > 10:
            bbox = ImageDraw.Draw(img).textbbox((0, 0), text, font=font,
                                                stroke_width=max(1,
                                                                 size // 12))
            if bbox[2] - bbox[0] <= w - 20:
                break
            size -= 2
            font = _callsign_font(size)
        bbox = ImageDraw.Draw(img).textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = 8, 6
        pad = max(4, size // 6)
        d = ImageDraw.Draw(img)
        # Solid plate, pure white text, no outline: Robot modes share
        # chroma between line pairs, and a translucent tinted plate or
        # outline shading gives the subsampled colour channel something
        # to smear. Flat black-on-white survives every mode family.
        d.rectangle((max(0, x - pad), max(0, y - pad),
                     min(w, x + tw + pad), min(h, y + th + pad)),
                    fill=(4, 4, 8))
        d.text((x - bbox[0], y - bbox[1]), text, font=font,
               fill=(255, 255, 255))
        return img

    # fallback: supersampled pixel font (smooth, not cubes)
    base = ImageFont.load_default()
    tmp = Image.new("L", (12 + 7 * len(text), 16), 0)
    ImageDraw.Draw(tmp).text((2, 2), text, font=base, fill=255)
    box = tmp.getbbox()
    if not box:
        return img
    tmp = tmp.crop(box)
    scale = max(2, h // 42)
    big = tmp.resize((tmp.width * scale * 3, tmp.height * scale * 3),
                     Image.NEAREST)
    mask = big.resize((tmp.width * scale, tmp.height * scale),
                      Image.LANCZOS)
    x, y = 8, 8
    pad = max(3, scale)
    ImageDraw.Draw(img).rectangle(
        (max(0, x - pad), max(0, y - pad),
         min(w, x + mask.width + pad), min(h, y + mask.height + pad)),
        fill=(4, 4, 8))
    white = Image.new("RGB", mask.size, (255, 255, 255))
    img.paste(white, (x, y), mask)
    return img


def test_pattern(w, h, callsign="SSTV LAB"):
    arr = np.zeros((h, w, 3), np.uint8)
    bars = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
            (255, 0, 255), (255, 0, 0), (0, 0, 255), (35, 35, 35)]
    top = int(h * 0.52)
    bw = w / len(bars)
    for i, c in enumerate(bars):
        arr[:top, int(i * bw):int((i + 1) * bw)] = c
    g0, g1 = top, int(h * 0.70)
    ramp = np.linspace(0, 255, w).astype(np.uint8)
    arr[g0:g1] = ramp[None, :, None]
    sq = max(6, w // 26)
    yy, xx = np.mgrid[g1:h, 0:w]
    chk = (((yy // sq) + (xx // sq)) % 2) * 165 + 45
    arr[g1:h] = chk[..., None].astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    cx, cy = int(w * 0.5), int((g1 + h) / 2)
    r = int((h - g1) * 0.42)
    d.ellipse([cx - r, cy - r, cx + r, cy + r],
              outline=(255, 60, 60), width=max(2, h // 90))
    d.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2],
              fill=(20, 20, 20), outline=(255, 210, 80), width=max(2, h // 120))
    return overlay_callsign(img, callsign)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
def _vis_segments(code):
    segs = [("t", 1900.0, VIS_LEADER), ("t", F_SYNC, VIS_BREAK),
            ("t", 1900.0, VIS_LEADER), ("t", F_SYNC, VIS_BIT)]
    bits = [(code >> i) & 1 for i in range(7)]
    bits.append(sum(bits) % 2)                      # even parity
    for b in bits:
        segs.append(("t", 1100.0 if b else 1300.0, VIS_BIT))
    segs.append(("t", F_SYNC, VIS_BIT))
    return segs


def _synthesize(segs, fs):
    cursor = 0.0
    chunks = []
    for kind, val, dur in segs:
        n0 = int(round(cursor * fs))
        cursor += dur
        n = int(round(cursor * fs)) - n0
        if n <= 0:
            continue
        if kind == "t":
            chunks.append(np.full(n, val, np.float64))
        else:                                       # pixel scan (staircase)
            vals = val
            idx = np.minimum((np.arange(n) * len(vals) // n), len(vals) - 1)
            chunks.append(F_BLACK + vals[idx] * (F_SPAN / 255.0))
    freq = np.concatenate(chunks)
    phase = 2.0 * np.pi * np.cumsum(freq) / fs
    y = np.sin(phase) * 0.85
    nf = int(0.005 * fs)
    ramp = np.sin(0.5 * np.pi * np.arange(nf) / nf) ** 2
    y[:nf] *= ramp
    y[-nf:] *= ramp[::-1]
    pad = np.zeros(int(PAD * fs))
    return np.concatenate([pad, y, pad]).astype(np.float32)


def encode_image(img, mode_name, fs=FS, progress=None):
    """img: PIL image (any size). Returns baseband SSTV audio float32."""
    mode = MODES[mode_name]
    w, h = mode["w"], mode["h"]
    fam = mode["family"]
    arr = np.asarray(fit_image(img, w, h), np.float64)
    segs = list(_vis_segments(mode["vis"]))

    if fam in ("martin", "scottie"):
        Gc = arr[..., 1]
        Bc = arr[..., 2]
        Rc = arr[..., 0]
        if fam == "scottie":
            segs.append(("t", F_SYNC, mode["sync"]))          # starting sync
        for l in range(h):
            if fam == "martin":
                segs += [("t", F_SYNC, mode["sync"]),
                         ("t", 1500.0, mode["porch"]),
                         ("s", Gc[l], mode["scan"]),
                         ("t", 1500.0, mode["sep"]),
                         ("s", Bc[l], mode["scan"]),
                         ("t", 1500.0, mode["sep"]),
                         ("s", Rc[l], mode["scan"]),
                         ("t", 1500.0, mode["sep"])]
            else:
                segs += [("t", 1500.0, mode["sep"]),
                         ("s", Gc[l], mode["scan"]),
                         ("t", 1500.0, mode["sep"]),
                         ("s", Bc[l], mode["scan"]),
                         ("t", F_SYNC, mode["sync"]),
                         ("t", 1500.0, mode["porch"]),
                         ("s", Rc[l], mode["scan"])]
            if progress:
                progress(l + 1, h)

    elif fam == "robot36":
        Y, Cr, Cb = rgb_to_ycrcb(arr)
        for l in range(h):
            pair = l - (l % 2)
            if l % 2 == 0:
                cvals = 0.5 * (Cr[pair] + Cr[min(pair + 1, h - 1)])
                sep_f = 1500.0
            else:
                cvals = 0.5 * (Cb[pair] + Cb[min(pair + 1, h - 1)])
                sep_f = 2300.0
            segs += [("t", F_SYNC, mode["sync"]),
                     ("t", 1500.0, mode["porch"]),
                     ("s", Y[l], mode["y_scan"]),
                     ("t", sep_f, mode["sep"]),
                     ("t", 1900.0, mode["sep_porch"]),
                     ("s", cvals, mode["c_scan"])]
            if progress:
                progress(l + 1, h)

    elif fam == "robot72":
        Y, Cr, Cb = rgb_to_ycrcb(arr)
        for l in range(h):
            segs += [("t", F_SYNC, mode["sync"]),
                     ("t", 1500.0, mode["porch"]),
                     ("s", Y[l], mode["y_scan"]),
                     ("t", 1500.0, mode["sep"]),
                     ("t", 1900.0, mode["sep_porch"]),
                     ("s", Cr[l], mode["c_scan"]),
                     ("t", 2300.0, mode["sep"]),
                     ("t", 1900.0, mode["sep_porch"]),
                     ("s", Cb[l], mode["c_scan"])]
            if progress:
                progress(l + 1, h)
    else:
        raise ValueError(fam)

    return _synthesize(segs, fs)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def find_vis(ft):
    """Look for a VIS header. Returns (code_or_None, end_sample_or_None)."""
    F, step = ft.coarse(0.001)
    if len(F) < 400:
        return None, None
    m12 = np.abs(F - 1200.0) <= 90.0
    m19 = np.abs(F - 1900.0) <= 140.0
    rising = np.where(m12[1:] & ~m12[:-1])[0] + 1
    fallback = (None, None)
    for i in rising:
        if i < 180 or i + 305 > len(F):
            continue
        if np.mean(m12[i:i + 26]) < 0.72:
            continue
        if np.mean(m19[i - 175:i - 15]) < 0.55:
            continue
        bits = []
        ok = True
        for k in range(1, 9):
            c = i + 30 * k + 15
            fm = float(np.mean(F[c - 8:c + 8]))
            if abs(fm - 1100.0) <= 90.0:
                bits.append(1)
            elif abs(fm - 1300.0) <= 90.0:
                bits.append(0)
            else:
                ok = False
                break
        if not ok:
            continue
        c = i + 30 * 9 + 15
        if abs(float(np.mean(F[c - 8:c + 8])) - 1200.0) > 90.0:
            continue
        if sum(bits) % 2 != 0:
            continue
        code = sum(bits[k] << k for k in range(7))
        end = (i + 300) * step
        if code in VIS2MODE:
            return code, end
        if fallback[0] is None:
            fallback = (code, end)
    return fallback


def _sync_score(ft, sync_dur):
    """Soft 0..1 'looks like a 1200 Hz sync' score, box-matched to the pulse."""
    sc = np.clip((1520.0 - ft.f) / 300.0, 0.0, 1.0).astype(np.float32)
    size = max(3, int(sync_dur * ft.fs))
    return ndi.uniform_filter1d(sc, size=size, mode="nearest")


def find_first_sync(ft, mode):
    off_s, sync_dur = sync_geometry(mode)
    sc = _sync_score(ft, sync_dur)
    limit = min(ft.n, int(10.0 * ft.fs))
    hits = np.where(sc[:limit] > 0.65)[0]
    if len(hits) == 0:
        return 0.0
    i0 = hits[0]
    j = i0 + int(np.argmax(sc[i0:i0 + int(2 * sync_dur * ft.fs) + 1]))
    return float(j) - (off_s + sync_dur / 2.0) * ft.fs


def _peak_frac(sc, i):
    """Parabolic sub-sample refinement of a score peak at integer index i."""
    if i <= 0 or i >= len(sc) - 1:
        return 0.0
    y0, y1, y2 = float(sc[i - 1]), float(sc[i]), float(sc[i + 1])
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-12:
        return 0.0
    return float(np.clip((y0 - y2) / (2.0 * den), -0.5, 0.5))


def sync_track(ft, mode, s0):
    """Sequential (PLL-style) tracking of the sync-pulse train.

    Instead of forcing one global straight line, follow the sync from line
    to line with an adaptive line-length estimate. This absorbs slow
    time-base wander: soundcard clock drift, tape/vocoder residue, doppler.
    Returns (line_start_positions[N], L_used, quality, slant_ppm)."""
    L_nom = line_period(mode) * ft.fs
    N = mode["h"]
    off_s, sync_dur = sync_geometry(mode)
    center = (off_s + sync_dur / 2.0) * ft.fs
    sc = _sync_score(ft, sync_dur)

    # Initial lock on LINE 1: line 0's sync can merge with the VIS stop bit
    # (both are 1200 Hz and contiguous), which would drag the anchor early.
    e1 = s0 + L_nom + center
    w0 = int(0.025 * ft.fs)
    a, b = int(max(0, e1 - w0)), int(min(ft.n, e1 + w0))
    if b - a > 8:
        j = int(np.argmax(sc[a:b]))
        if sc[a + j] > 0.35:
            e1 = a + j + _peak_frac(sc, a + j)

    win = int(max(0.005, min(0.012, line_period(mode) * 0.03)) * ft.fs)
    L_est = L_nom
    L_lo, L_hi = L_nom * 0.97, L_nom * 1.03
    e_clip = 0.0008 * ft.fs
    pos = np.zeros(N)
    conf = np.zeros(N)
    est = float(e1)
    for l in range(1, N):
        a = int(max(0, est - win))
        b = int(min(ft.n, est + win))
        if b - a < 4:
            pos[l] = est
            est += L_est
            continue
        j = int(np.argmax(sc[a:b]))
        c = float(sc[a + j])
        conf[l] = c
        if c > 0.30:
            found = a + j + _peak_frac(sc, a + j)
            err = found - est
            L_est = min(L_hi, max(L_lo,
                                  L_est + 0.08 * np.clip(err, -e_clip, e_clip)))
            pos[l] = found
            est = found + L_est
        else:
            pos[l] = est
            est += L_est

    L_head = float(np.median(np.diff(pos[1:min(N, 10)]))) \
        if N >= 4 else L_est
    if not (L_lo <= L_head <= L_hi):
        L_head = L_nom
    pos[0] = pos[1] - L_head                    # line 0 from the track
    conf[0] = 0.0

    # robust slant estimate + gentle smoothing of the tracked centers
    ls = np.arange(N, dtype=np.float64)
    good = conf > 0.45
    if good.sum() >= max(8, 0.15 * N):
        A_mat = np.stack([np.ones(int(good.sum())), ls[good]], axis=1)
        coef, *_ = np.linalg.lstsq(A_mat, pos[good], rcond=None)
        A, B = float(coef[0]), float(coef[1])
    else:
        A, B = float(pos[0]), L_nom
    if abs(B - L_nom) / L_nom > 0.05:
        B = L_nom
    resid = pos - (A + B * ls)
    # per-line sync estimates are noisy under weak signals while the true
    # time base is smooth (FM restore is exactly linear; even vocoder
    # wander is band-limited) — so smooth the residual track firmly.
    # Low-confidence lines are bridged from their neighbours first.
    bad = conf < 0.30
    if bad.any() and (~bad).sum() >= 4:
        resid[bad] = np.interp(ls[bad], ls[~bad], resid[~bad])
    resid = ndi.median_filter(resid, size=5, mode="nearest")
    resid = ndi.gaussian_filter1d(resid, sigma=2.2, mode="nearest")
    starts = (A + B * ls + resid) - center
    quality = float(np.mean(conf > 0.5))
    slant_ppm = (B / L_nom - 1.0) * 1e6
    return starts, B, quality, slant_ppm


def _row_shift_est(a, b, k=5):
    """Sub-pixel horizontal shift of row b relative to row a (gradient xcorr
    with parabolic interpolation). Returns (shift_px, confidence 0..1).
    Works on the |correlation| envelope, so an *inverted* neighbouring row
    (checkerboard band boundaries) still yields the true displacement via
    its anti-correlation peak. Confidence combines strength with peak
    sharpness, so flat plateaus (horizontal ramps are shift-invariant) are
    reported unreliable instead of confidently wrong."""
    ga = np.diff(a)
    gb = np.diff(b)
    ga = ga - ga.mean()
    gb = gb - gb.mean()
    e = float(np.sqrt((ga * ga).sum() * (gb * gb).sum()))
    if e < 1e-6:
        return 0.0, 0.0
    cc = np.correlate(np.pad(gb, (k, k)), ga, "valid")
    ac = np.abs(cc)
    i = int(np.argmax(ac))
    if i <= 0 or i >= len(ac) - 1:
        return 0.0, 0.0
    sgn = 1.0 if cc[i] >= 0 else -1.0
    y0, y1, y2 = sgn * cc[i - 1], sgn * cc[i], sgn * cc[i + 1]
    den = y0 - 2.0 * y1 + y2
    d = 0.0 if abs(den) < 1e-9 else (y0 - y2) / (2.0 * den)
    strength = ac[i] / e
    sharp = (ac[i] - max(ac[0], ac[-1])) / e
    return (i - k) + float(np.clip(d, -1.0, 1.0)), float(min(strength,
                                                             4.0 * sharp))


def jitter_repair(img, max_shift=4.5, iters=2):
    """Image-domain de-jitter for time-stretched receptions.

    Residual time-base wander serrates vertical edges, and because
    Martin/Scottie send R, G, B at different moments within a line, each
    channel jitters *independently* (colour-split edges). Cross-channel
    matching is invalid on saturated colours, so each channel is
    straightened against its own neighbouring rows: chained row-to-row
    shift estimates are integrated and HIGH-PASSED along the image (real
    diagonals, slant and geometry survive; only the wiggle goes). Rows
    whose estimate is unreliable fall back to a two-row comparison — which
    also handles alternating patterns like checkerboards, where the row two
    back is in phase — and anything still unreliable is bridged. Shifts are
    applied once, sub-pixel."""
    src_f = img.astype(np.float64)
    h, w, _ = src_f.shape
    grid = np.arange(w, dtype=np.float64)
    total = np.zeros((h, 3))
    X = src_f.copy()

    def rshift(row, s):
        return np.interp(np.clip(grid + s, 0, w - 1), grid, row)

    def roughness(ch):
        """Mean |row-to-row misalignment| where measurable — the do-no-harm
        yardstick: a correction pass must reduce this or it is rolled back."""
        vals = []
        for r in range(1, h, 2):
            e1, c1 = _row_shift_est(X[r - 1, :, ch], X[r, :, ch], 7)
            if c1 > 0.15:
                vals.append(abs(e1))
        return float(np.mean(vals)) if len(vals) >= 8 else None

    for _ in range(iters):
        for ch in range(3):
            dr = np.zeros(h)
            for r in range(1, h):
                e1, c1 = _row_shift_est(X[r - 1, :, ch], X[r, :, ch], 7)
                if c1 > 0.15:
                    dr[r] = np.clip(e1, -6.0, 6.0)
            c = np.cumsum(dr)
            corr = np.clip(c - ndi.gaussian_filter1d(c, 10.0),
                           -max_shift, max_shift)
            corr[np.abs(corr) < 0.05] = 0.0
            before = roughness(ch)
            saved = X[:, :, ch].copy()
            for r in range(h):
                if corr[r]:
                    X[r, :, ch] = rshift(X[r, :, ch], corr[r])
            after = roughness(ch)
            if before is not None and after is not None                     and after > before * 0.98:
                X[:, :, ch] = saved                     # do no harm
            else:
                total[:, ch] += corr

    out = np.empty_like(src_f)
    for r in range(h):
        for ch in range(3):
            out[r, :, ch] = (rshift(src_f[r, :, ch], total[r, ch])
                             if abs(total[r, ch]) > 0.03
                             else src_f[r, :, ch])
    return np.clip(out, 0, 255).astype(np.uint8)


def _noise_sigma(chan):
    """Robust per-channel noise estimate (gray levels), exploiting SSTV
    geometry: adjacent scan lines carry nearly identical *content* but
    statistically independent *noise*, so the median absolute vertical
    difference is a clean sigma estimator that edges barely disturb."""
    d = np.abs(np.diff(chan.astype(np.float64), axis=0))
    return float(np.median(d)) / (0.6745 * np.sqrt(2.0))


def _despeckle_lines(chan, sigma):
    """Kill impulsive line streaks: where the rows above and below agree
    with each other but the pixel disagrees with both, the pixel is demod
    shot noise, not detail — replace it with the vertical mean."""
    X = chan
    up = np.roll(X, 1, axis=0)
    dn = np.roll(X, -1, axis=0)
    vm = 0.5 * (up + dn)
    agree = np.abs(up - dn) < (3.0 * sigma + 6.0)
    imp = agree & (np.abs(X - vm) > (4.0 * sigma + 8.0))
    imp[0] = imp[-1] = False
    X[imp] = vm[imp]
    return X


def _nlm(chan, sigma, h_rel, dy, dx, patch=5):
    """Vectorised pixelwise non-local means with an anisotropic (tall)
    search window: SSTV noise is independent between lines but correlated
    along them, so the best averaging partners live above and below."""
    X = chan.astype(np.float64)
    h, w = X.shape
    pad_y, pad_x = dy + patch // 2, dx + patch // 2
    P = np.pad(X, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    h2 = (h_rel * max(sigma, 0.5)) ** 2
    acc = np.zeros_like(X)
    wsum = np.zeros_like(X)
    for u in range(-dy, dy + 1):
        for v in range(-dx, dx + 1):
            S = P[pad_y + u:pad_y + u + h, pad_x + v:pad_x + v + w]
            D = ndi.uniform_filter((X - S) ** 2, size=patch, mode="nearest")
            wgt = np.exp(-np.maximum(D - 2.0 * sigma * sigma, 0.0) / h2)
            acc += wgt * S
            wsum += wgt
    return acc / wsum


def denoise_image(img, strength=1.0):
    """Detail-preserving denoiser for off-air receptions, in the spirit of
    VE3NEA's SSTV Image Denoiser: measure the actual noise level, remove
    impulsive line streaks, then average each pixel only with patches that
    genuinely look alike (non-local means), searching mostly *vertically*
    where SSTV noise is independent. Luma is treated gently to keep detail;
    chroma, where the eye forgives smoothing, more firmly. Clean images
    pass through untouched."""
    rgb = img.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 128.0
    Cr = (R - Y) * 0.713 + 128.0

    sig = _noise_sigma(Y)
    if sig * strength < 1.6:
        return img                          # already clean — do nothing
    say = min(sig, 40.0)
    # heavy noise (e.g. horizontally-correlated streaks from high-factor
    # FM restores of a noisy channel) needs proportionally firmer
    # averaging: escalate strength with the measured level.
    strength = strength * float(np.clip(say / 10.0, 1.0, 1.8))

    Y = _despeckle_lines(Y, say)
    Cb = _despeckle_lines(Cb, say)
    Cr = _despeckle_lines(Cr, say)
    Y = _nlm(Y, say, h_rel=0.75 * strength, dy=8, dx=3)
    Cb = _nlm(Cb, say, h_rel=1.5 * strength, dy=8, dx=3)
    Cr = _nlm(Cr, say, h_rel=1.5 * strength, dy=8, dx=3)

    R2 = Y + 1.403 * (Cr - 128.0)
    B2 = Y + 1.773 * (Cb - 128.0)
    G2 = (Y - 0.299 * R2 - 0.114 * B2) / 0.587
    out = np.stack([R2, G2, B2], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def sharpen_image(img, amount=0.8, radius=1.1):
    """Constrained unsharp on luma: overshoot is clamped to the local 3x3
    min/max so edges steepen without halos, and the amount backs off as the
    measured noise level rises so grain is never amplified. Helps crispen
    hard graphics at high compression factors; on smooth photographs it is
    a matter of taste — hence a toggle, not a default."""
    f = img.astype(np.float64)
    Y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    a = amount * float(np.clip(1.0 - _noise_sigma(Y) / 22.0, 0.15, 1.0))
    hp = Y - ndi.gaussian_filter(Y, radius)
    Y2 = Y + a * hp
    lo = ndi.minimum_filter(Y, size=3)
    hi = ndi.maximum_filter(Y, size=3)
    Y2 = np.clip(Y2, lo, hi)
    out = f + (Y2 - Y)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class DecodeSink:
    """Shared buffer so a GUI can paint lines as they land."""

    def __init__(self):
        self.img = None
        self.rows = 0
        self.total = 0


def decode_signal(y, fs, ui_mode_name, sink=None, progress=None, log=None,
                  cleanup=True, denoise=True, sharpen=False):
    """Full decode: VIS detect -> sync regression -> per-line demod.

    Returns (img uint8 HxWx3, mode_name, info dict)."""
    def say(msg):
        if log:
            log(msg)

    ft = FreqTrack(y, fs, lo=700.0, hi=3200.0, order=3)   # wide: sharpest
    code, vis_end = find_vis(ft)
    if vis_end is not None:
        # noise probe on the VIS leader (known steady 1900 Hz tone): if the
        # channel is dirty, rebuild the track with the narrow noise-robust
        # band; if clean, keep the wide band for maximum edge sharpness.
        la = int(vis_end - vis_duration() * fs + 0.06 * fs)
        lb = int(la + 0.20 * fs)
        if 0 <= la < lb <= ft.n:
            sig_f = float(np.std(ft.f[la:lb]))
            if sig_f > 22.0:
                if log:
                    log(f"Leader noise σ≈{sig_f:.0f} Hz — narrow demod")
                ft = FreqTrack(y, fs)
                code2, ve2 = find_vis(ft)
                if ve2 is not None:
                    code, vis_end = code2, ve2
    if code in VIS2MODE:
        mode_name = VIS2MODE[code]
        say(f"VIS header: {mode_name} (code {code})")
    else:
        mode_name = ui_mode_name
        if code is not None:
            say(f"VIS code {code} unknown — using selected mode {mode_name}")
        else:
            say(f"No VIS found — free-running as {mode_name}")
    mode = MODES[mode_name]
    w, h, fam = mode["w"], mode["h"], mode["family"]

    if vis_end is not None:
        s0 = float(vis_end)
        if fam == "scottie":
            s0 += mode["sync"] * fs
    else:
        s0 = find_first_sync(ft, mode)

    starts, L, quality, slant_ppm = sync_track(ft, mode, s0)
    say(f"Sync lock {quality * 100:.0f}% · slant {slant_ppm:+.0f} ppm")

    # Local line rate: time-base wander changes *within* a line too, and in
    # Martin/Scottie the three colour channels are sent at different times —
    # a fixed per-line offset would shift them by different amounts (colour
    # serration on vertical edges). Interpolate the warp between consecutive
    # tracked line starts and scale each segment's nominal offset by the
    # local rate.
    Ls = np.empty(h)
    Ls[:-1] = np.diff(starts)
    Ls[-1] = Ls[-2] if h > 1 else line_period(mode) * fs
    Ls = ndi.median_filter(Ls, size=5, mode="nearest")
    Lp = line_period(mode)

    def wb(l, off_s):
        """Warped absolute position of a segment `off_s` seconds into line l."""
        return starts[l] + (off_s / Lp) * Ls[l]

    img = np.zeros((h, w, 3), np.uint8)
    if sink is not None:
        sink.img = img
        sink.total = h
        sink.rows = 0

    spx = fs  # samples per second, readability below
    if fam in ("martin", "scottie"):
        if fam == "martin":
            og = (mode["sync"] + mode["porch"])
            ob = og + mode["scan"] + mode["sep"]
            orr = ob + mode["scan"] + mode["sep"]
        else:
            og = mode["sep"]
            ob = og + mode["scan"] + mode["sep"]
            orr = ob + mode["scan"] + mode["sync"] + mode["porch"]
        sc = mode["scan"] * spx
        for l in range(h):
            g = freq_to_val(ft.scan(wb(l, og), sc, w))
            b = freq_to_val(ft.scan(wb(l, ob), sc, w))
            r = freq_to_val(ft.scan(wb(l, orr), sc, w))
            img[l, :, 0] = r
            img[l, :, 1] = g
            img[l, :, 2] = b
            if sink is not None:
                sink.rows = l + 1
            if progress:
                progress(l + 1, h)

    elif fam == "robot36":
        oy = mode["sync"] + mode["porch"]
        osep = oy + mode["y_scan"]
        oc = osep + mode["sep"] + mode["sep_porch"]
        Yb = np.zeros((h, w))
        Crows = np.zeros((h, w))
        flags = np.zeros(h, bool)          # True => this line carried B-Y
        for l in range(h):
            Yb[l] = freq_to_val(ft.scan(wb(l, oy), mode["y_scan"] * spx, w))
            sep_f = ft.mean(wb(l, osep), wb(l, osep) + mode["sep"] * spx)
            flags[l] = sep_f > 1900.0
            Crows[l] = freq_to_val(ft.scan(wb(l, oc),
                                           mode["c_scan"] * spx, w))
            # provisional live paint: assume standard parity
            k = l - (l % 2)
            cr = Crows[k]
            cb = Crows[k + 1] if (k + 1) <= l else np.full(w, 128.0)
            img[l] = ycrcb_to_rgb(Yb[l], cr, cb)
            if sink is not None:
                sink.rows = l + 1
            if progress:
                progress(l + 1, h)
        agree = float(np.mean(flags == (np.arange(h) % 2 == 1)))
        flipped = agree < 0.5
        if flipped:
            say("Robot 36 chroma parity flipped — corrected")
        for k in range(0, h - 1, 2):
            cr = Crows[k + 1] if flipped else Crows[k]
            cb = Crows[k] if flipped else Crows[k + 1]
            img[k] = ycrcb_to_rgb(Yb[k], cr, cb)
            img[k + 1] = ycrcb_to_rgb(Yb[k + 1], cr, cb)

    elif fam == "robot72":
        oy = mode["sync"] + mode["porch"]
        ocr = oy + mode["y_scan"] + mode["sep"] + mode["sep_porch"]
        ocb = ocr + mode["c_scan"] + mode["sep"] + mode["sep_porch"]
        for l in range(h):
            Yv = freq_to_val(ft.scan(wb(l, oy), mode["y_scan"] * spx, w))
            Cr = freq_to_val(ft.scan(wb(l, ocr), mode["c_scan"] * spx, w))
            Cb = freq_to_val(ft.scan(wb(l, ocb), mode["c_scan"] * spx, w))
            img[l] = ycrcb_to_rgb(Yv, Cr, Cb)
            if sink is not None:
                sink.rows = l + 1
            if progress:
                progress(l + 1, h)

    if cleanup:
        img = jitter_repair(img)
    if denoise:
        img = denoise_image(img, strength=(1.6 if denoise is True
                                           else float(denoise)))
    if sharpen:
        img = sharpen_image(img)
    if (cleanup or denoise or sharpen) and sink is not None:
        sink.img = img

    info = dict(mode=mode_name, vis=code, quality=quality,
                slant_ppm=slant_ppm, start_s=s0 / fs)
    return img, mode_name, info


# ---------------------------------------------------------------------------
# Automatic factor / method detection (tries to parse a VIS after restoring)
# ---------------------------------------------------------------------------
def auto_restore(y, fs, ui_factor, ui_method, log=None):
    """Return (restored_audio, factor, method, vis_code_or_None)."""
    def say(msg):
        if log:
            log(msg)

    head = y[:int(4.5 * fs)]
    methods = [METHOD_FM, METHOD_RESAMPLE]
    factors = [ui_factor] + [f for f in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.5, 1.25, 1.0]
                             if abs(f - ui_factor) > 1e-9]
    for meth in methods:
        for f in factors:
            try:
                r = restore_audio(head, fs, f, meth)
                code, _ = find_vis(FreqTrack(r, fs))
            except Exception:
                continue
            if code in VIS2MODE:
                say(f"Auto-detect: {f:g}x {meth} -> VIS {VIS2MODE[code]}")
                return restore_audio(y, fs, f, meth), f, meth, code
    say(f"Auto-detect failed — using selected {ui_factor:g}x {ui_method}")
    return restore_audio(y, fs, ui_factor, ui_method), ui_factor, ui_method, None


# ---------------------------------------------------------------------------
# Spectrogram rendering (the "on-air scope")
# ---------------------------------------------------------------------------
_LUT_STOPS = [(0.00, (10, 12, 20)), (0.35, (24, 62, 120)),
              (0.60, (28, 168, 178)), (0.82, (245, 195, 70)),
              (1.00, (255, 255, 255))]


def _lut():
    lut = np.zeros((256, 3), np.uint8)
    for i in range(256):
        t = i / 255.0
        for (t0, c0), (t1, c1) in zip(_LUT_STOPS, _LUT_STOPS[1:]):
            if t0 <= t <= t1:
                u = (t - t0) / (t1 - t0 + 1e-9)
                lut[i] = [int(c0[k] + u * (c1[k] - c0[k])) for k in range(3)]
                break
    return lut


_SPEC_LUT = _lut()


def spectrogram_image(y, fs=FS, fmax=5000.0, width=1000, height=170):
    y = np.asarray(y, np.float32)
    n_fft = 1024
    hop = max(n_fft // 4, len(y) // max(width, 1))
    S = np.abs(_stft(y, n_fft, hop))
    kmax = min(S.shape[0], int(n_fft * fmax / fs) + 1)
    S = S[:kmax]
    db = 20.0 * np.log10(S + 1e-6)
    db -= db.max()
    img = np.clip((db + 70.0) * (255.0 / 70.0), 0, 255).astype(np.uint8)
    rgb = _SPEC_LUT[img][::-1]                       # low freq at bottom
    pil = Image.fromarray(rgb)
    return pil.resize((width, height), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
def psnr(a, b, trim=4):
    a = np.asarray(a, np.float64)[trim:-trim, trim:-trim]
    b = np.asarray(b, np.float64)[trim:-trim, trim:-trim]
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-9 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def selftest(demo_dir=None, quick=False):
    t_all = time.time()
    rows = []

    def dump(img, name):
        if demo_dir:
            os.makedirs(demo_dir, exist_ok=True)
            Image.fromarray(img).save(os.path.join(demo_dir, name))

    print("== 1. Direct encode -> decode (codec sanity) ==")
    names = ["Robot 36", "Scottie S2"] if quick else list(MODES)
    for name in names:
        m = MODES[name]
        ref_img = test_pattern(m["w"], m["h"], "VE3 LAB")
        ref = np.asarray(fit_image(ref_img, m["w"], m["h"]))
        t0 = time.time()
        y = encode_image(ref_img, name)
        dec, got, info = decode_signal(y, FS, name)
        p = psnr(ref, dec)
        rows.append((f"{name} direct", p))
        dump(dec, f"direct_{name.replace(' ', '')}.png")
        print(f"  {name:11s} dur {len(y)/FS:6.1f}s  PSNR {p:5.1f} dB  "
              f"lock {info['quality']*100:3.0f}%  ({time.time()-t0:.1f}s)")

    print("== 2. Compress -> restore -> decode ==")
    combos = [("Martin M1", METHOD_RESAMPLE, 2.0),
              ("Martin M1", METHOD_FM, 2.0),
              ("Martin M1", METHOD_FM, 3.0),
              ("Martin M1", METHOD_FM, 4.0),
              ("Martin M1", METHOD_FM, 6.0),
              ("Martin M1", METHOD_FM, 7.0),
              ("Martin M1", METHOD_FM, 8.0),
              ("Robot 36", METHOD_FM, 2.0),
              ("Scottie S2", METHOD_FM, 2.0),
              ("Scottie S2", METHOD_RESAMPLE, 3.0)]
    if quick:
        combos = combos[3:4]
    cache = {}
    for name, meth, fac in combos:
        m = MODES[name]
        if name not in cache:
            ref_img = test_pattern(m["w"], m["h"], "VE3 LAB")
            cache[name] = (np.asarray(fit_image(ref_img, m["w"], m["h"])),
                           encode_image(ref_img, name))
        ref, y = cache[name]
        t0 = time.time()
        air = compress_audio(y, FS, fac, meth)
        back = restore_audio(air, FS, fac, meth)
        dec, got, info = decode_signal(back, FS, name)
        p = psnr(ref, dec)
        rows.append((f"{name} {meth} {fac:g}x", p))
        dump(dec, f"{meth}{fac:g}x_{name.replace(' ', '')}.png")
        print(f"  {name:11s} {meth:8s} {fac:g}x  air {len(air)/FS:6.1f}s "
              f"(was {len(y)/FS:5.1f}s)  PSNR {p:5.1f} dB  "
              f"lock {info['quality']*100:3.0f}%  ({time.time()-t0:.1f}s)")

    if not quick:
        print("== 3. Noisy SSB channel + auto-detect ==")
        name, meth, fac = "Martin M1", METHOD_FM, 3.0
        ref, y = cache[name]
        air = compress_audio(y, FS, fac, meth)
        rx = channel_simulate(air, FS, bw_hz=2700, snr_db=10, seed=7)
        back, gf, gm, code = auto_restore(rx, FS, 1.0, METHOD_FM,
                                          log=lambda s: print("   ", s))
        raw, got, info = decode_signal(back, FS, name, denoise=False)
        dec, got, info = decode_signal(back, FS, name)
        p0, p = psnr(ref, raw), psnr(ref, dec)
        print(f"  denoiser: raw {p0:5.1f} dB -> denoised {p:5.1f} dB")
        rows.append(("M1 fm 3x SSB2.7k SNR10 auto", p))
        dump(dec, "channel_M1_fm3x_snr10.png")
        print(f"  detected {gf:g}x {gm}, mode {got}, PSNR {p:5.1f} dB, "
              f"lock {info['quality']*100:.0f}%")

    print(f"== done in {time.time()-t_all:.1f}s ==")
    worst_direct = min(p for n, p in rows if "direct" in n)
    ok = worst_direct > 20.0
    for n, p in rows:
        print(f"  {'PASS' if p > 10 else 'soft':4s} {p:5.1f} dB  {n}")
    print("(direct/resample sit near 23-33 dB on this synthetic torture "
          "pattern; the FM method keeps most of that all the way to 8x.)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI encode / decode
# ---------------------------------------------------------------------------
def cli_encode(a):
    img = Image.open(a.encode)
    if a.callsign:
        m = MODES[a.mode]
        img = overlay_callsign(fit_image(img, m["w"], m["h"]), a.callsign)
    print(f"Encoding {a.mode} ...")
    y = encode_image(img, a.mode)
    fac = float(a.factor)
    air = compress_audio(y, FS, fac, a.method)
    if a.vox:
        air = np.concatenate([vox_preamble(FS), air])
    write_wav(a.out, air)
    print(f"Baseband {len(y)/FS:.1f}s -> on-air {len(air)/FS:.1f}s "
          f"({a.method} {fac:g}x)\nSaved {a.out}")


def cli_decode(a):
    y, fsr = read_wav_any(a.decode)
    print(f"Loaded {a.decode} ({len(y)/FS:.1f}s at {fsr} Hz)")
    if a.factor == "auto":
        back, f, m, code = auto_restore(y, FS, 2.0, a.method, log=print)
    else:
        f = float(a.factor)
        back = restore_audio(y, FS, f, a.method)
    img, mode, info = decode_signal(back, FS, a.mode, log=print)
    Image.fromarray(img).save(a.out)
    print(f"Decoded {mode} -> {a.out}  (sync lock {info['quality']*100:.0f}%)")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def launch_gui(smoketest=False):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import tkinter.font as tkfont
    from PIL import ImageTk

    BG = "#0d1117"
    PANEL = "#151b26"
    PANEL2 = "#1b2331"
    EDGE = "#2a3549"
    FG = "#dbe6f1"
    MUT = "#7f8fa4"
    TEAL = "#37c8c4"          # phosphor scope accent
    AMBER = "#f0a63c"         # TX / warning accent
    OK = "#49d17d"
    ERR = "#ff6b6b"

    FACTOR_LABELS = {"1x (off)": 1.0, "1.25x": 1.25, "1.5x": 1.5,
                     "2x": 2.0, "3x": 3.0, "4x": 4.0, "5x": 5.0, "6x": 6.0, "7x": 7.0, "8x": 8.0}
    METHOD_LABELS = {"FM turbo (recommended, SSB-safe)": METHOD_FM,
                     "Resample (pitch shifts, wideband)": METHOD_RESAMPLE}

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Experimental SSTV Studio")
            root.configure(bg=BG)
            root.minsize(840, 560)
            root.geometry("1280x820")
            try:
                dpi = root.winfo_fpixels("1i")
                root.tk.call("tk", "scaling", dpi / 72.0)
            except Exception:
                pass

            self.q = queue.Queue()
            self.busy = False
            self.listening = False
            self.stream = None
            self.audio_q = queue.Queue()

            self.img_src = None            # PIL, as loaded
            self.img_tx = None             # PIL, fitted+callsign (as encoded)
            self.y_base = None             # baseband SSTV audio
            self.y_air = None              # compressed on-air audio
            self.sink = DecodeSink()
            self.img_rx_final = None
            self.spec_pil = None
            self.meter_db = -90.0

            self._style()
            self._build()
            self._refresh_cfg()
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            root.after(60, self._pump)
            self._init_scaler()
            self.log("Experimental SSTV Studio ready.")
            self.log("Audio device support: "
                     + ("available" if HAVE_AUDIO else
                        "not found (pip install sounddevice for play / live RX)"))
            self._mode_changed()

        # ---------------- uniform window scaling ----------------
        DESIGN_W, DESIGN_H = 1280, 820

        def _init_scaler(self):
            """Make the whole UI scale as one unit with the window.

            Tk has no scale-the-tree transform, but almost every ttk
            widget sizes itself from its font, so scaling every font (plus
            the few pixel-sized canvases) by one common ratio shrinks or
            grows the entire interface proportionally — everything fits at
            any window size instead of clipping."""
            self._scale_fonts = []
            seen = set()
            for name in tkfont.names(self.root):
                f = tkfont.nametofont(name)
                key = str(f)
                if key not in seen:
                    seen.add(key)
                    self._scale_fonts.append((f, int(f["size"])))
            for f in (self.f_mono, self.f_title, self.f_small):
                self._scale_fonts.append((f, int(f["size"])))
            self._scale_px = [(self.cv_spec, "height", 150),
                              (self.cv_meter, "width", 140),
                              (self.cv_meter, "height", 12)]
            self._scale_cur = 1.0
            self._scale_job = None
            # The design size is authored at 96 dpi. Under Windows display
            # scaling (125 %, 150 %...) fonts grow while window pixels do
            # not, so the same nominal window holds less UI — express the
            # design size in DPI-adjusted pixels so the ratio accounts for
            # it and the interface shrinks to fit.
            try:
                self._dpi_mul = float(self.root.winfo_fpixels("1i")) / 96.0
            except Exception:
                self._dpi_mul = 1.0
            self.root.bind("<Configure>", self._on_root_configure, add="+")

        def _on_root_configure(self, e):
            if e.widget is not self.root or e.width <= 1:
                return
            s = min(e.width / (self.DESIGN_W * self._dpi_mul),
                    e.height / (self.DESIGN_H * self._dpi_mul))
            s = max(0.55, min(1.6, s))
            if abs(s - self._scale_cur) < 0.02:
                return
            if self._scale_job is not None:
                try:
                    self.root.after_cancel(self._scale_job)
                except Exception:
                    pass
            self._scale_job = self.root.after(
                80, lambda s=s: self._apply_scale(s))

        def _apply_scale(self, s):
            self._scale_job = None
            self._scale_cur = s
            for f, base in self._scale_fonts:
                mag = max(6, int(round(abs(base) * s)))
                f.configure(size=mag if base >= 0 else -mag)
            for w, opt, base in self._scale_px:
                try:
                    w.configure(**{opt: max(6, int(round(base * s)))})
                except Exception:
                    pass
            # canvases redraw via their own <Configure>; nudge the ones
            # that draw from cached images
            try:
                self._draw_spec()
            except Exception:
                pass

        # ---------------- styling ----------------
        def _style(self):
            st = ttk.Style(self.root)
            try:
                st.theme_use("clam")
            except Exception:
                pass
            base = tkfont.nametofont("TkDefaultFont")
            base.configure(size=10)
            self.f_mono = tkfont.nametofont("TkFixedFont").copy()
            self.f_mono.configure(size=10)
            self.f_title = tkfont.nametofont("TkFixedFont").copy()
            self.f_title.configure(size=17, weight="bold")
            self.f_small = self.f_mono.copy()
            self.f_small.configure(size=8)
            self.f_head = base.copy()
            self.f_head.configure(size=10, weight="bold")

            st.configure(".", background=BG, foreground=FG,
                         fieldbackground=PANEL2, bordercolor=EDGE,
                         lightcolor=EDGE, darkcolor=EDGE, troughcolor=PANEL2,
                         selectbackground=TEAL, selectforeground=BG)
            st.configure("TFrame", background=BG)
            st.configure("Panel.TFrame", background=PANEL)
            st.configure("TLabel", background=BG, foreground=FG)
            st.configure("Panel.TLabel", background=PANEL, foreground=FG)
            st.configure("Mut.TLabel", background=PANEL, foreground=MUT)
            st.configure("MutBG.TLabel", background=BG, foreground=MUT)
            st.configure("Head.TLabel", background=PANEL, foreground=TEAL,
                         font=self.f_head)
            st.configure("HeadTX.TLabel", background=PANEL, foreground=AMBER,
                         font=self.f_head)
            st.configure("TButton", background=PANEL2, foreground=FG,
                         padding=(10, 5), borderwidth=1, focusthickness=1)
            st.map("TButton",
                   background=[("active", "#26314a"), ("disabled", PANEL)],
                   foreground=[("disabled", MUT)])
            st.configure("Accent.TButton", background="#215e5c",
                         foreground="#eafffb")
            st.map("Accent.TButton", background=[("active", "#2b7f7c"),
                                                 ("disabled", PANEL)])
            st.configure("TX.TButton", background="#6e4a15",
                         foreground="#ffe9c4")
            st.map("TX.TButton", background=[("active", "#8a5d1b"),
                                             ("disabled", PANEL)])
            st.configure("TCombobox", arrowcolor=FG)
            st.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
                   foreground=[("readonly", FG)])
            st.configure("TCheckbutton", background=BG, foreground=FG)
            st.map("TCheckbutton", background=[("active", BG)])
            st.configure("TEntry", insertcolor=FG)
            st.configure("TSpinbox", arrowcolor=FG, insertcolor=FG)
            st.configure("Horizontal.TProgressbar", background=TEAL,
                         troughcolor=PANEL2, bordercolor=EDGE)
            st.configure("Horizontal.TScale", background=BG)

        # ---------------- layout ----------------
        def _build(self):
            r = self.root
            r.columnconfigure(0, weight=1)
            r.rowconfigure(3, weight=3)
            r.rowconfigure(4, weight=1)

            # header -------------------------------------------------------
            head = ttk.Frame(r)
            head.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))
            head.columnconfigure(1, weight=1)
            tk.Label(head, text="EXPERIMENTAL SSTV STUDIO", font=self.f_title,
                     bg=BG, fg=TEAL).grid(row=0, column=0, sticky="w")
            ttk.Label(head, style="MutBG.TLabel",
                      text="  time-compressed slow scan · burst TX, stretch RX"
                      ).grid(row=0, column=1, sticky="w")

            # toolbar 1: mode / compression --------------------------------
            t1 = ttk.Frame(r)
            t1.grid(row=1, column=0, sticky="ew", padx=14, pady=(6, 2))
            ttk.Label(t1, text="Mode").pack(side="left")
            self.var_mode = tk.StringVar(value="Martin M1")
            cb = ttk.Combobox(t1, textvariable=self.var_mode, width=11,
                              values=list(MODES), state="readonly")
            cb.pack(side="left", padx=(6, 14))
            cb.bind("<<ComboboxSelected>>", lambda e: self._mode_changed())
            ttk.Label(t1, text="Compression").pack(side="left")
            self.var_voxhdr = tk.BooleanVar(value=False)
            self.var_factor = tk.StringVar(value="2x")
            ttk.Combobox(t1, textvariable=self.var_factor, width=8,
                         values=list(FACTOR_LABELS), state="readonly"
                         ).pack(side="left", padx=(6, 14))
            ttk.Checkbutton(t1, text="VOX header",
                            variable=self.var_voxhdr).pack(side="left",
                                                           padx=(0, 12))
            ttk.Label(t1, text="Method").pack(side="left")
            self.var_method = tk.StringVar(value=list(METHOD_LABELS)[0])
            ttk.Combobox(t1, textvariable=self.var_method, width=30,
                         values=list(METHOD_LABELS), state="readonly"
                         ).pack(side="left", padx=(6, 14))
            ttk.Label(t1, text="Callsign").pack(side="left")
            self.var_call = tk.StringVar(value="")
            ttk.Entry(t1, textvariable=self.var_call, width=10
                      ).pack(side="left", padx=(6, 0))
            self.lbl_mode = ttk.Label(t1, style="MutBG.TLabel", text="")
            self.lbl_mode.pack(side="right")

            # toolbar 2: channel sim + RX options --------------------------
            t2 = ttk.Frame(r)
            t2.grid(row=2, column=0, sticky="ew", padx=14, pady=(2, 6))
            ttk.Label(t2, text="Channel sim").pack(side="left")
            self.var_chan = tk.StringVar(value="Off (direct)")
            ttk.Combobox(t2, textvariable=self.var_chan, width=14,
                         values=["Off (direct)", "SSB 2.7 kHz", "SSB 2.4 kHz",
                                 "FM 3 kHz (FRS)"],
                         state="readonly").pack(side="left", padx=(6, 10))
            self.var_noise = tk.BooleanVar(value=False)
            ttk.Checkbutton(t2, text="Noise, SNR", variable=self.var_noise
                            ).pack(side="left")
            self.var_snr = tk.StringVar(value="12")
            ttk.Spinbox(t2, from_=-6, to=40, textvariable=self.var_snr,
                        width=4).pack(side="left", padx=(4, 2))
            ttk.Label(t2, text="dB").pack(side="left", padx=(0, 16))
            self.var_auto = tk.BooleanVar(value=True)
            ttk.Checkbutton(t2, text="RX auto-detect rate/method",
                            variable=self.var_auto).pack(side="left")
            self.var_clean = tk.BooleanVar(value=True)
            ttk.Checkbutton(t2, text="Artifact cleanup",
                            variable=self.var_clean).pack(side="left",
                                                          padx=(12, 6))
            ttk.Label(t2, text="Denoise").pack(side="left", padx=(6, 2))
            self.var_denoise = tk.StringVar(value="Normal")
            ttk.Combobox(t2, textvariable=self.var_denoise, width=7,
                         state="readonly",
                         values=("Off", "Normal", "Strong")
                         ).pack(side="left", padx=(0, 8))
            self.var_sharp = tk.BooleanVar(value=False)
            ttk.Checkbutton(t2, text="Sharpen",
                            variable=self.var_sharp).pack(side="left",
                                                          padx=(0, 16))
            ttk.Label(t2, text="VOX").pack(side="left")
            self.var_vox = tk.DoubleVar(value=-42.0)
            ttk.Scale(t2, from_=-70, to=-15, variable=self.var_vox,
                      length=130).pack(side="left", padx=(6, 2))
            self.lbl_vox = ttk.Label(t2, style="MutBG.TLabel", text="-42 dB")
            self.lbl_vox.pack(side="left")
            self.var_vox.trace_add(
                "write", lambda *a: self.lbl_vox.configure(
                    text=f"{self.var_vox.get():.0f} dB"))

            # main panels --------------------------------------------------
            main = ttk.Frame(r)
            main.grid(row=3, column=0, sticky="nsew", padx=14)
            main.columnconfigure(0, weight=1, uniform="p")
            main.columnconfigure(1, weight=1, uniform="p")
            main.rowconfigure(0, weight=1)

            # TX panel
            txp = ttk.Frame(main, style="Panel.TFrame", padding=10)
            txp.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
            txp.columnconfigure(0, weight=1)
            txp.rowconfigure(1, weight=1)
            ttk.Label(txp, text="TRANSMIT", style="HeadTX.TLabel"
                      ).grid(row=0, column=0, sticky="w")
            self.cv_tx = tk.Canvas(txp, bg="#090d13", highlightthickness=1,
                                   highlightbackground=EDGE)
            self.cv_tx.grid(row=1, column=0, sticky="nsew", pady=6)
            self.cv_tx.bind("<Configure>",
                            lambda e: self._draw_canvas(self.cv_tx,
                                                        self._tx_display()))
            self.lbl_tx = ttk.Label(txp, style="Mut.TLabel",
                                    text="Load an image or use the test pattern.")
            self.lbl_tx.grid(row=2, column=0, sticky="w")
            bb = ttk.Frame(txp, style="Panel.TFrame")
            bb.grid(row=3, column=0, sticky="ew", pady=(8, 0))
            ttk.Button(bb, text="Load image…",
                       command=self.act_load).pack(side="left", padx=(0, 6))
            ttk.Button(bb, text="Test pattern",
                       command=self.act_pattern).pack(side="left", padx=(0, 6))
            self.b_enc = ttk.Button(bb, text="Encode + compress",
                                    style="TX.TButton", command=self.act_encode)
            self.b_enc.pack(side="left", padx=(0, 6))
            self.b_save = ttk.Button(bb, text="Save on-air WAV…",
                                     command=self.act_save, state="disabled")
            self.b_save.pack(side="left", padx=(0, 6))
            self.b_play = ttk.Button(bb, text="Play", command=self.act_play,
                                     state="disabled")
            self.b_play.pack(side="left", padx=(0, 6))
            self.b_loop = ttk.Button(bb, text="Send → RX (loopback)",
                                     style="Accent.TButton",
                                     command=self.act_loop, state="disabled")
            self.b_loop.pack(side="left")

            # RX panel
            rxp = ttk.Frame(main, style="Panel.TFrame", padding=10)
            rxp.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
            rxp.columnconfigure(0, weight=1)
            rxp.rowconfigure(1, weight=1)
            hd = ttk.Frame(rxp, style="Panel.TFrame")
            hd.grid(row=0, column=0, sticky="ew")
            ttk.Label(hd, text="RECEIVE", style="Head.TLabel").pack(side="left")
            self.cv_meter = tk.Canvas(hd, width=140, height=12, bg=PANEL2,
                                      highlightthickness=1,
                                      highlightbackground=EDGE)
            self.cv_meter.pack(side="right")
            ttk.Label(hd, text="level ", style="Mut.TLabel").pack(side="right")
            self.cv_rx = tk.Canvas(rxp, bg="#090d13", highlightthickness=1,
                                   highlightbackground=EDGE)
            self.cv_rx.grid(row=1, column=0, sticky="nsew", pady=6)
            self.cv_rx.bind("<Configure>",
                            lambda e: self._draw_canvas(self.cv_rx,
                                                        self._rx_display()))
            self.pb = ttk.Progressbar(rxp, mode="determinate", maximum=100)
            self.pb.grid(row=2, column=0, sticky="ew")
            self.lbl_rx = ttk.Label(rxp, style="Mut.TLabel", text="Idle.")
            self.lbl_rx.grid(row=3, column=0, sticky="w", pady=(3, 0))
            rb = ttk.Frame(rxp, style="Panel.TFrame")
            rb.grid(row=4, column=0, sticky="ew", pady=(8, 0))
            ttk.Button(rb, text="Open WAV + decode…",
                       command=self.act_open).pack(side="left", padx=(0, 6))
            self.b_listen = ttk.Button(rb, text="Start listening",
                                       command=self.act_listen,
                                       state=("normal" if HAVE_AUDIO
                                              else "disabled"))
            self.b_listen.pack(side="left", padx=(0, 6))
            self.b_stop = ttk.Button(rb, text="Stop", command=self.act_stop,
                                     state="disabled")
            self.b_stop.pack(side="left", padx=(0, 6))
            self.b_savepng = ttk.Button(rb, text="Save image…",
                                        command=self.act_savepng,
                                        state="disabled")
            self.b_savepng.pack(side="left")

            # bottom: scope + log -----------------------------------------
            bot = ttk.Frame(r)
            bot.grid(row=4, column=0, sticky="nsew", padx=14, pady=(10, 4))
            bot.columnconfigure(0, weight=3)
            bot.columnconfigure(1, weight=2)
            bot.rowconfigure(1, weight=1)
            ttk.Label(bot, text="ON-AIR SCOPE  ·  0–5 kHz",
                      style="MutBG.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(bot, text="LOG", style="MutBG.TLabel"
                      ).grid(row=0, column=1, sticky="w", padx=(10, 0))
            self.cv_spec = tk.Canvas(bot, bg="#0a0e15", height=150,
                                     highlightthickness=1,
                                     highlightbackground=EDGE)
            self.cv_spec.grid(row=1, column=0, sticky="nsew")
            self.cv_spec.bind("<Configure>", lambda e: self._draw_spec())
            logf = ttk.Frame(bot)
            logf.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
            logf.columnconfigure(0, weight=1)
            logf.rowconfigure(0, weight=1)
            self.txt = tk.Text(logf, height=8, bg=PANEL2, fg=FG,
                               insertbackground=FG, relief="flat",
                               font=self.f_mono, wrap="word",
                               highlightthickness=1,
                               highlightbackground=EDGE)
            self.txt.grid(row=0, column=0, sticky="nsew")
            sb = ttk.Scrollbar(logf, command=self.txt.yview)
            sb.grid(row=0, column=1, sticky="ns")
            self.txt.configure(yscrollcommand=sb.set, state="disabled")
            self.txt.tag_configure("err", foreground=ERR)
            self.txt.tag_configure("ok", foreground=OK)

            self.status = ttk.Label(r, style="MutBG.TLabel", anchor="w",
                                    text=f"fs {FS} Hz · modes: "
                                         + ", ".join(MODES))
            self.status.grid(row=5, column=0, sticky="ew", padx=14,
                             pady=(0, 8))

        # ---------------- settings snapshot (thread safety) ----------------
        # Tk variables may only be read on the main thread; workers read
        # this plain dict instead, refreshed by the UI pump.
        def _refresh_cfg(self):
            self.cfg = dict(
                mode=self.var_mode.get(),
                factor=FACTOR_LABELS[self.var_factor.get()],
                method=METHOD_LABELS[self.var_method.get()],
                call=self.var_call.get(),
                auto=self.var_auto.get(),
                clean=self.var_clean.get(),
                denoise={"Off": False, "Normal": 1.6,
                         "Strong": 2.4}[self.var_denoise.get()],
                sharpen=self.var_sharp.get(),
                vox=float(self.var_vox.get()),
            )

        # ---------------- helpers ----------------
        def log(self, msg, tag=None):
            self.q.put(("log", msg, tag))

        def _log_now(self, msg, tag=None):
            self.txt.configure(state="normal")
            ts = datetime.now().strftime("%H:%M:%S")
            self.txt.insert("end", f"[{ts}] {msg}\n", tag or ())
            self.txt.see("end")
            self.txt.configure(state="disabled")

        def factor(self):
            return FACTOR_LABELS[self.var_factor.get()]

        def method(self):
            return METHOD_LABELS[self.var_method.get()]

        def _mode_changed(self):
            m = MODES[self.var_mode.get()]
            self.lbl_mode.configure(
                text=f"{m['w']}×{m['h']} · nominal "
                     f"{nominal_duration(m):.1f} s")

        def _tx_display(self):
            return self.img_tx or self.img_src

        def _rx_display(self):
            if self.sink.img is not None and self.sink.rows > 0:
                return Image.fromarray(self.sink.img)
            if self.img_rx_final is not None:
                return Image.fromarray(self.img_rx_final)
            return None

        def _draw_canvas(self, cv, pil):
            cv.delete("all")
            w = max(2, cv.winfo_width())
            h = max(2, cv.winfo_height())
            if pil is None:
                cv.create_text(w // 2, h // 2, fill=MUT,
                               text="—", font=self.f_mono)
                return
            s = min((w - 8) / pil.width, (h - 8) / pil.height)
            nw, nh = max(1, int(pil.width * s)), max(1, int(pil.height * s))
            res = Image.NEAREST if s > 1.6 else Image.LANCZOS
            im = pil.resize((nw, nh), res)
            ph = ImageTk.PhotoImage(im)
            cv._ph = ph
            cv.create_image(w // 2, h // 2, image=ph)

        def _draw_spec(self):
            cv = self.cv_spec
            cv.delete("all")
            w = max(2, cv.winfo_width())
            h = max(2, cv.winfo_height())
            if self.spec_pil is None:
                cv.create_text(w // 2, h // 2, fill=MUT, font=self.f_mono,
                               text="encode or load a signal to see the scope")
                return
            im = self.spec_pil.resize((w, h), Image.BILINEAR)
            ph = ImageTk.PhotoImage(im)
            cv._ph = ph
            cv.create_image(w // 2, h // 2, image=ph)
            for khz in (1, 2, 3, 4):
                y = h - int(h * khz / 5.0)
                cv.create_line(0, y, w, y, fill="#233045")
                cv.create_text(4, y - 7, anchor="w", fill=MUT,
                               font=self.f_small,
                               text=f"{khz}k")

        def _draw_meter(self):
            cv = self.cv_meter
            cv.delete("all")
            w = int(cv.winfo_width())
            frac = min(1.0, max(0.0, (self.meter_db + 70.0) / 55.0))
            thr = min(1.0, max(0.0, (self.var_vox.get() + 70.0) / 55.0))
            color = TEAL if self.meter_db < self.var_vox.get() else AMBER
            cv.create_rectangle(0, 0, int(w * frac), 12, fill=color, width=0)
            x = int(w * thr)
            cv.create_line(x, 0, x, 12, fill=FG)

        def set_busy(self, b):
            self.busy = b
            state = "disabled" if b else "normal"
            for btn in (self.b_enc, self.b_loop, self.b_save, self.b_play):
                if not b and btn in (self.b_save, self.b_play, self.b_loop) \
                        and self.y_air is None:
                    btn.configure(state="disabled")
                else:
                    btn.configure(state=state)

        def bg(self, fn):
            def run():
                try:
                    fn()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.log(f"Error: {e}", "err")
                finally:
                    self.q.put(("busy", False, None))
            self.q.put(("busy", True, None))
            threading.Thread(target=run, daemon=True).start()

        # ---------------- actions ----------------
        def act_load(self):
            p = filedialog.askopenfilename(
                title="Choose an image",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                           ("All files", "*.*")])
            if not p:
                return
            try:
                self.img_src = Image.open(p).convert("RGB")
            except Exception as e:
                messagebox.showerror("Image", str(e))
                return
            self.img_tx = None
            self.y_base = self.y_air = None
            self.log(f"Loaded {os.path.basename(p)} "
                     f"({self.img_src.width}×{self.img_src.height})")
            self._draw_canvas(self.cv_tx, self._tx_display())
            self.b_save.configure(state="disabled")
            self.b_play.configure(state="disabled")
            self.b_loop.configure(state="disabled")

        def act_pattern(self):
            m = MODES[self.var_mode.get()]
            self.img_src = test_pattern(m["w"], m["h"],
                                        self.var_call.get() or "SSTV LAB")
            self.img_tx = None
            self.y_base = self.y_air = None
            self.log("Test pattern generated.")
            self._draw_canvas(self.cv_tx, self._tx_display())

        def act_encode(self):
            if self.img_src is None:
                self.act_pattern()
            mode = self.var_mode.get()
            fac, meth = self.factor(), self.method()
            call = self.var_call.get()
            voxhdr = self.var_voxhdr.get()

            def work():
                m = MODES[mode]
                img = fit_image(self.img_src, m["w"], m["h"])
                img = overlay_callsign(img, call)
                self.img_tx = img
                self.q.put(("txcanvas", None, None))
                self.q.put(("stage", f"Encoding {mode}…", None))
                y = encode_image(img, mode,
                                 progress=lambda l, n:
                                 self.q.put(("prog", 60 * l / n, None)))
                self.y_base = y
                self.q.put(("stage",
                            f"Time-compressing {fac:g}× ({meth})…", None))
                self.q.put(("prog", 70, None))
                air = compress_audio(y, FS, fac, meth)
                if voxhdr:
                    air = np.concatenate([vox_preamble(FS), air])
                self.y_air = air
                self.spec_pil = spectrogram_image(air)
                self.q.put(("spec", None, None))
                self.q.put(("prog", 100, None))
                base_s, air_s = len(y) / FS, len(air) / FS
                mb = len(air) * 2 / 1e6
                self.q.put(("txinfo",
                            f"Baseband {base_s:.1f} s → on-air {air_s:.1f} s "
                            f"({fac:g}× {meth}) · {mb:.1f} MB WAV", None))
                self.log(f"Encoded {mode}: {base_s:.1f} s → {air_s:.1f} s "
                         f"on-air ({100 * (1 - air_s / base_s):.0f}% airtime "
                         f"saved)", "ok")
                self.q.put(("txbtns", True, None))
            self.bg(work)

        def act_save(self):
            if self.y_air is None:
                return
            fac = self.factor()
            p = filedialog.asksaveasfilename(
                defaultextension=".wav",
                initialfile=f"sstv_onair_{self.var_mode.get().replace(' ', '')}"
                            f"_{fac:g}x_{self.method()}.wav",
                filetypes=[("WAV", "*.wav")])
            if not p:
                return
            write_wav(p, self.y_air)
            self.log(f"Saved {os.path.basename(p)} "
                     f"({len(self.y_air) / FS:.1f} s)", "ok")

        def act_play(self):
            if not HAVE_AUDIO or self.y_air is None:
                return
            try:
                sd.stop()
                sd.play(self.y_air, FS)
                self.log("Playing on-air audio…")
            except Exception as e:
                self.log(f"Audio error: {e}", "err")

        def act_loop(self):
            if self.y_air is None:
                return
            y = self.y_air.copy()
            chan = self.var_chan.get()
            bw = {"SSB 2.7 kHz": 2700, "SSB 2.4 kHz": 2400,
                  "FM 3 kHz (FRS)": 3000}.get(chan)
            kind = "fm" if chan.startswith("FM") else "ssb"
            snr = float(self.var_snr.get()) if self.var_noise.get() else None

            def work():
                sig = y
                if bw or snr is not None:
                    self.q.put(("stage", "Channel: filtering / noise…", None))
                    sig = channel_simulate(sig, FS, bw_hz=bw, snr_db=snr,
                                           kind=kind)
                    self.spec_pil = spectrogram_image(sig)
                    self.q.put(("spec", None, None))
                    self.log(f"Channel sim: {chan}"
                             + (f", SNR {snr:g} dB" if snr is not None else ""))
                self._rx_pipeline(sig, source="loopback")
            self.bg(work)

        def act_open(self):
            p = filedialog.askopenfilename(
                title="Open on-air WAV", filetypes=[("WAV", "*.wav"),
                                                    ("All files", "*.*")])
            if not p:
                return

            def work():
                y, fsr = read_wav_any(p)
                self.log(f"Loaded {os.path.basename(p)} "
                         f"({len(y) / FS:.1f} s @ {fsr} Hz)")
                self.spec_pil = spectrogram_image(y)
                self.q.put(("spec", None, None))
                self._rx_pipeline(y, source="file")
            self.bg(work)

        def _rx_pipeline(self, y, source):
            cfg = self.cfg
            fac, meth = cfg["factor"], cfg["method"]
            self.q.put(("rxreset", None, None))
            if cfg["auto"]:
                self.q.put(("stage", "Auto-detecting rate…", None))
                back, gf, gm, code = auto_restore(y, FS, fac, meth,
                                                  log=self.log)
            else:
                gf, gm = fac, meth
                self.q.put(("stage", f"Restoring {gf:g}× ({gm})…", None))
                back = restore_audio(y, FS, gf, gm)
            self.q.put(("prog", 15, None))
            self.q.put(("stage", "Decoding…", None))
            img, mode, info = decode_signal(
                back, FS, cfg["mode"], sink=self.sink,
                log=self.log, cleanup=cfg["clean"], denoise=cfg["denoise"],
                sharpen=cfg["sharpen"],
                progress=lambda l, n:
                self.q.put(("prog", 15 + 85 * l / n, None)))
            self.img_rx_final = img
            self.q.put(("rxdone",
                        f"{mode} · restore {gf:g}× {gm} · sync "
                        f"{info['quality'] * 100:.0f}% · slant "
                        f"{info['slant_ppm']:+.0f} ppm · {source}", None))
            self.log(f"Decode complete: {mode} "
                     f"(sync {info['quality'] * 100:.0f}%)", "ok")

        # ---------------- live listening (VOX) ----------------
        def act_listen(self):
            if not HAVE_AUDIO or self.listening:
                return
            self.listening = True
            self.b_listen.configure(state="disabled")
            self.b_stop.configure(state="normal")
            self.log("Listening… (VOX armed — waiting for signal)")

            def cb(indata, frames, t, status):
                self.audio_q.put(indata[:, 0].copy())

            try:
                self.stream = sd.InputStream(samplerate=FS, channels=1,
                                             blocksize=2048, callback=cb)
                self.stream.start()
            except Exception as e:
                self.log(f"Input error: {e}", "err")
                self.listening = False
                self.b_listen.configure(state="normal")
                self.b_stop.configure(state="disabled")
                return
            threading.Thread(target=self._vox_worker, daemon=True).start()

        def _vox_worker(self):
            ring = []
            ring_len = 0
            cap = None
            silence = 0.0
            while self.listening:
                try:
                    block = self.audio_q.get(timeout=0.3)
                except queue.Empty:
                    continue
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
                level = 20.0 * math.log10(rms)
                self.meter_db = level
                self.q.put(("meter", None, None))
                thr = self.cfg["vox"]
                if cap is None:
                    ring.append(block)
                    ring_len += len(block)
                    while ring_len > FS:                    # 1 s pre-roll
                        ring_len -= len(ring.pop(0))
                    if level > thr:
                        cap = list(ring)
                        self.q.put(("stage", "Signal! recording…", None))
                        self.log("VOX triggered — capturing burst")
                        silence = 0.0
                else:
                    cap.append(block)
                    if level < thr - 4:
                        silence += len(block) / FS
                    else:
                        silence = 0.0
                    total = sum(len(b) for b in cap) / FS
                    if silence > 1.5 or total > 240:
                        y = np.concatenate(cap).astype(np.float32)
                        cap = None
                        ring, ring_len = [], 0
                        self.log(f"Captured {len(y) / FS:.1f} s — processing")
                        self.spec_pil = spectrogram_image(y)
                        self.q.put(("spec", None, None))
                        try:
                            self._rx_pipeline(y, source="live")
                        except Exception as e:
                            self.log(f"Decode error: {e}", "err")
                        self.q.put(("stage",
                                    "Listening… (VOX armed)", None))

        def act_stop(self):
            self.listening = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            if HAVE_AUDIO:
                try:
                    sd.stop()
                except Exception:
                    pass
            self.b_listen.configure(state="normal" if HAVE_AUDIO
                                    else "disabled")
            self.b_stop.configure(state="disabled")
            self.log("Stopped.")

        def act_savepng(self):
            if self.img_rx_final is None:
                return
            p = filedialog.asksaveasfilename(
                defaultextension=".png", initialfile="sstv_decoded.png",
                filetypes=[("PNG", "*.png")])
            if not p:
                return
            Image.fromarray(self.img_rx_final).save(p)
            self.log(f"Saved {os.path.basename(p)}", "ok")

        # ---------------- UI pump ----------------
        def _pump(self):
            self._refresh_cfg()
            try:
                while True:
                    kind, a, b = self.q.get_nowait()
                    if kind == "log":
                        self._log_now(a, b)
                    elif kind == "busy":
                        self.set_busy(a)
                    elif kind == "prog":
                        self.pb["value"] = a
                    elif kind == "stage":
                        self.lbl_rx.configure(text=a)
                    elif kind == "txinfo":
                        self.lbl_tx.configure(text=a)
                    elif kind == "txcanvas":
                        self._draw_canvas(self.cv_tx, self._tx_display())
                    elif kind == "txbtns":
                        for btn in (self.b_save, self.b_play, self.b_loop):
                            btn.configure(state="normal")
                    elif kind == "spec":
                        self._draw_spec()
                    elif kind == "meter":
                        self._draw_meter()
                    elif kind == "rxreset":
                        self.sink = DecodeSink()
                        self.img_rx_final = None
                        self.pb["value"] = 0
                        self.b_savepng.configure(state="disabled")
                        self._draw_canvas(self.cv_rx, None)
                    elif kind == "rxdone":
                        self.lbl_rx.configure(text=a)
                        self.b_savepng.configure(state="normal")
                        self._draw_canvas(self.cv_rx, self._rx_display())
            except queue.Empty:
                pass
            if self.busy and self.sink.img is not None and self.sink.rows:
                self._draw_canvas(self.cv_rx, self._rx_display())
            self.root.after(60, self._pump)

        def on_close(self):
            self.listening = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception:
                    pass
            self.root.destroy()

    root = tk.Tk()
    app = App(root)
    if smoketest:
        def pump(cond, timeout):
            t0 = time.time()
            while time.time() - t0 < timeout:
                root.update()
                if cond():
                    return True
                time.sleep(0.02)
            return False

        app.var_mode.set("Robot 36")
        app._mode_changed()
        app.act_pattern()
        app.var_factor.set("2x")
        app.var_chan.set("SSB 2.7 kHz")
        app.var_noise.set(True)
        app.var_snr.set("14")
        app.act_encode()
        assert pump(lambda: app.y_air is not None and not app.busy, 120), \
            "encode did not finish"
        app.act_loop()
        assert pump(lambda: app.img_rx_final is not None and not app.busy,
                    180), "loopback decode did not finish"
        q = float(app.img_rx_final.mean())

        # ---- window scaling: everything must fit at any size ----
        def visible_controls_fit():
            rw, rh = root.winfo_width(), root.winfo_height()
            rx0, ry0 = root.winfo_rootx(), root.winfo_rooty()
            bad = []
            def walk(w):
                for c in w.winfo_children():
                    cls = c.winfo_class()
                    if cls in ("TButton", "TCheckbutton", "TCombobox",
                               "TSpinbox", "TLabel") and c.winfo_ismapped():
                        x = c.winfo_rootx() - rx0
                        y = c.winfo_rooty() - ry0
                        if (x + c.winfo_width() > rw + 2
                                or y + c.winfo_height() > rh + 2
                                or x < -2 or y < -2):
                            bad.append((cls, str(c),
                                        x, y, c.winfo_width()))
                    walk(c)
            walk(root)
            return bad
        t0_size = app.f_title["size"]
        for gw, gh in ((1600, 1000), (1100, 700), (860, 560)):
            root.geometry(f"{gw}x{gh}")
            t_end = time.time() + 3.0
            while time.time() < t_end:
                root.update()
                time.sleep(0.03)
                if abs(root.winfo_width() - gw) <= 4:
                    break
            root.update()
            time.sleep(0.2)
            root.update()
            bad = visible_controls_fit()
            assert not bad, f"clipped controls at {gw}x{gh}: {bad[:4]}"
        assert app.f_title["size"] < t0_size,             "fonts did not scale down at small window size"
        root.update()
        root.destroy()
        print(f"GUI smoketest OK (loopback decoded, mean pixel {q:.0f}; "
              f"scaling verified at 3 window sizes)")
        return
    root.mainloop()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experimental SSTV Studio")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--demo", metavar="DIR", help="dump self-test PNGs here")
    ap.add_argument("--smoketest", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--encode", metavar="IMAGE")
    ap.add_argument("--decode", metavar="WAV")
    ap.add_argument("--mode", default="Martin M1", choices=list(MODES))
    ap.add_argument("--factor", default="2")
    ap.add_argument("--method", default=METHOD_FM,
                    choices=[METHOD_FM, METHOD_RESAMPLE])
    ap.add_argument("--callsign", default="")
    ap.add_argument("--vox", action="store_true",
                    help="prepend a 0.7 s 1900 Hz keying header for "
                         "VOX-triggered transmitters (FRS handhelds)")
    ap.add_argument("--out", default="out.wav")
    a = ap.parse_args()

    if a.selftest or a.demo:
        sys.exit(selftest(demo_dir=a.demo, quick=a.quick))
    if a.encode:
        cli_encode(a)
        return
    if a.decode:
        if a.out == "out.wav":
            a.out = "decoded.png"
        cli_decode(a)
        return
    launch_gui(smoketest=a.smoketest)


if __name__ == "__main__":
    main()
