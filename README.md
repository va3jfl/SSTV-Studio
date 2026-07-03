# Experimental SSTV Studio

Time-compressed slow scan television: encode an image to SSTV audio, squeeze
the audio to a fraction of its length before it hits the air, and let the
receiving station stretch it back and decode as usual. Half (or a quarter) of
the airtime, paid for with a few seconds of number crunching on each end.

The idea comes from an old trick with audio editors: speed a recording up
3–4×, save the much shorter file, and later time-stretch it back — with a
decent algorithm the result was surprisingly close to the original. This
project asks the obvious ham question: *what happens if the "recording" is a
Martin M1 transmission?* Shorter bursts mean less exposure to QSB and QRM,
easier fits into short openings, and less time holding a frequency. The catch
is that this is not a standard mode: **both ends need this tool** (or an
equivalent external time-stretcher). Treat it as an experiment between
consenting stations, and identify your transmissions as your local
regulations require.

It's on GitHub — va3jfl/SSTV-Studio, license and all. The project has a home now, and that deserves a moment. Now, the README trick: plain Markdown image syntax can't scale, but GitHub allows HTML <img> tags with width attributes inside READMEs, and an HTML <table> gives you the grid. One wrinkle your filenames add: the spaces need to be %20-encoded in the paths. Here's a ready-to-paste block — first row shows two screenshots side by side at half width each, second row centers the third, every image clickable to open full size:
html## Screenshots

<table>
  <tr>
    <td align="center" width="50%"><b>FM turbo 8× — clean loopback</b></td>
    <td align="center" width="50%"><b>SSB 2.7 kHz + noise — 7×</b></td>
  </tr>
  <tr>
    <td><a href="screenshots/fm%20clean%208x.png"><img src="screenshots/fm%20clean%208x.png" width="100%"></a></td>
    <td><a href="screenshots/ssb%20noise%207x.png"><img src="screenshots/ssb%20noise%207x.png" width="100%"></a></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Wideband — 8×</b></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><a href="screenshots/Wideband%208x.png"><img src="screenshots/Wideband%208x.png" width="50%"></a></td>
  </tr>
</table>

## Quick start

    pip install numpy scipy Pillow sounddevice
    python3 sstv_studio.py

`sounddevice` is optional — without it everything works except Play and live
listening. On Linux you may also need the Tk package: `sudo apt install
python3-tk`. Python 3.9+.

Headless check that your install works end to end:

    python3 sstv_studio.py --selftest

## Using the studio

The window is split into a TX side and an RX side over a shared "on-air
scope" (a spectrogram of whatever was last transmitted or received) and a
log.

**Transmit:** load an image (or click *Test pattern*), pick a mode, a
compression factor and a method, optionally type a callsign to burn into the
image, then *Encode + compress*. The status line shows the trade: e.g.
Martin M1 `Baseband 115.7 s → on-air 57.9 s (2× stretch)`. Save the WAV and
play it into your rig like any SSTV audio, or click *Play* to key via VOX.

**Receive:** either open a recorded WAV, or click *Start listening* — the RX
side then waits with a VOX squelch (threshold slider in the toolbar, one
second of pre-roll is kept so the start of a burst isn't clipped), records
until the signal drops, restores the time base and decodes, painting the
image line by line as it lands. With *auto-detect* on, the decoder finds the
compression factor and method by trial-restoring the first seconds and
looking for a valid VIS header, so the RX operator doesn't need to be told
the settings in advance.

**Loopback and channel simulator:** *Send → RX (loopback)* pushes the
freshly encoded burst straight into the receive chain, optionally through a
simulated SSB filter (2.7 or 2.4 kHz) with additive noise at a chosen SNR.
This is the quickest way to get a feel for what each factor does before
going on air.

## The two methods, and the physics

**FM turbo (recommended).** SSTV is frequency modulation, so the honest way
to run it faster is to speed up the *modulation* itself: demodulate to an
instantaneous-frequency track, resample that track — an exactly linear
operation with no artifacts of its own — and resynthesize the carrier. The
deviation range (1200–2300 Hz) is untouched, so pitch is inherently
preserved and the signal fits a normal SSB voice channel; it is equivalent
to transmitting the same mode with a faster scan clock. The scaler's inner demodulator is factor-aware: a
k×-compressed signal has k× the modulation rate and therefore k×-wider
sidebands (Carson), so its passband opens with the factor — capped at the
measured spectral edge of the input, since demodulating past where the
channel's filter cut off only admits noise, and with the low side floored
at 900 Hz only when the spectrum shows actual noise below the carrier.
This single change is what makes 5–8× practical: with the old fixed narrow
band, 8× lost ~6 dB of sharpness that was sitting in the shaved sidebands
all along. What remains is
pure channel physics: compressing time multiplies the modulation rate, so
by Carson's rule the sidebands widen and a 2.7 kHz filter shaves some of
that energy — fine horizontal detail softens gradually as the factor rises.
In loopback this method loses only a few dB to a direct transmission even
at 4×, and it is also the best *receiver* for the vocoder method below.

**Resample.** The tape trick: play the audio N× faster. Round-trips
essentially losslessly — but every frequency is multiplied, so a 2×
Martin transmission occupies roughly 2400–4600 Hz and a 4× one reaches
9.2 kHz. An SSB filter will destroy it; use it only over channels that pass
the widened spectrum (wide FM audio, direct cable, digital links), or as a
lab reference.

The decoder is shared by all paths: it demodulates instantaneous frequency
with a precision-safe conjugate-product method, parses the VIS header, then
tracks the sync-pulse train line by line with a small PLL-style follower
rather than assuming a rigid line grid — absorbing residual time-base
wander and ordinary soundcard clock drift alike. The receiver measures channel noise on the VIS
leader tone and picks its demodulator bandwidth accordingly — wide for
clean signals (maximum edge sharpness), narrow for dirty ones — and the
sync tracker refines every pulse to sub-sample accuracy before smoothing
the line grid against the knowledge that the true time base is nearly
linear. An optional cleanup pass
adds content-based de-jitter (each channel straightened against its own
rows, with a do-no-harm check), and a detail-preserving denoiser — inspired
by VE3NEA's SSTV Image Denoiser — cleans off-air receptions: it measures
the actual noise level, removes impulsive line streaks, then averages each
pixel only with patches that genuinely match, searching mostly vertically
where SSTV noise is statistically independent. Luma is treated gently,
chroma firmly, and clean images pass through untouched; when the noise is
heavy — including the horizontally-streaked noise that a high-factor FM
restore makes out of channel hiss (stretching the track stretches the
noise with it) — the filter strength escalates automatically. A separate
Sharpen toggle applies a halo-clamped unsharp on luma: it measurably
crispens hard graphics at high factors and is a matter of taste on smooth
photographs, so it is off by default. Three settings in
the RX panel: Off / Normal / Strong. The channel simulator also offers an
"FM 3 kHz (FRS)" model — pre-emphasis, a 300–3000 Hz shelf, and
de-emphasised noise — for rehearsing handheld and repeater paths.

## What to expect (measured)

Numbers from `--selftest`, which round-trips a deliberately nasty synthetic
test pattern (color bars, gradient, fine checkerboard, pixel text) — real
photographs fare visibly better. PSNR against the transmitted image,
"lock" is the fraction of scan lines whose sync pulse was confidently found:

| Path | On-air time | PSNR | Sync lock |
|---|---|---|---|
| Martin M1 direct (no compression) | 115.7 s | 36.7 dB | 100 % |
| Martin M1, resample 2× | 57.9 s | 36.7 dB | 100 % |
| Martin M1, FM turbo 2× | 57.9 s | 31.1 dB | 100 % |
| Martin M1, FM turbo 3× | 38.6 s | 27.7 dB | 100 % |
| Martin M1, FM turbo 4× | 28.9 s | 27.6 dB | 100 % |
| Martin M1, FM turbo 5× | 23.1 s | 25.9 dB | 100 % |
| Martin M1, FM turbo 6× | 19.3 s | 25.2 dB | 100 % |
| Martin M1, FM turbo 7× | 16.5 s | 24.6 dB | 100 % |
| Martin M1, FM turbo 8× | 14.5 s | 23.7 dB | 100 % |
| Martin M1, FM 3× through SSB 2.7 kHz at 10 dB SNR, auto-detected | 38.6 s | 21.4 dB | 100 % |
| Martin M1, FM 5× through SSB at 10 dB SNR, auto-detected | 23.1 s | 19.0 dB | 100 % |
| Martin M1, FM 8× through SSB at 12 dB SNR, auto-detected | 14.5 s | 17.0 dB | 96 % |
| Robot 36, FM turbo 2× | 18.7 s | 22.3 dB | 100 % |
| Scottie S2, FM turbo 2× | 36.3 s | 24.8 dB | 100 % |
| Photo, FM 3× through SSB at 10 dB SNR — denoised | 38.6 s | 29.3 dB | 100 % |
| Photo, FM 3× through SSB at 6 dB SNR — raw / denoised | 38.6 s | 22.6 / 27.2 dB | 100 % |



## Command line

    python3 sstv_studio.py --encode photo.jpg --mode "Martin M1" \
        --factor 3 --method fm --callsign "VE3XYZ" --out onair.wav

    python3 sstv_studio.py --decode onair.wav --factor auto --out rx.png

`--factor auto` on decode probes factors and methods until a valid VIS
appears. `--selftest` (optionally with `--demo DIR` to dump PNGs, or
`--quick`) runs the full QA matrix and exits nonzero if the core codec
misbehaves.

## The FRS "kid lab" setup

FRS handhelds (licence-exempt walkie-talkies) are narrowband FM with a
hands-free jack — which happens to be exactly the channel this program
likes. A workable across-town picture link:

Radio speaker / hands-free output → small audio isolation transformer
(600:600 or similar; isolation matters more than exact matching — it kills
ground-loop hum) → PC line-in. PC headphone output → roughly 100:1
resistive divider (e.g. 22 kΩ series into 220 Ω across the mic pins) →
radio mic input, with the radio's VOX turned on. Then tick **VOX header**
in the TX panel (or pass `--vox` on the CLI): it prepends a 0.7 s 1900 Hz
tone so the radio's VOX attack and the far end's squelch open on a
sacrificial tone instead of eating the VIS header — in simulation, 2×–8×
all decode through an FM voice channel with 250 ms of VOX clipping, and
without the header the same transmission fails outright. Rehearse first
with the channel simulator set to "FM 3 kHz (FRS)".

Keep the radios themselves stock: FRS certification requires the integral
antenna, so the fun experimental antenna (a hand-wound UHF loop!) belongs
on a receive-only device instead — a cheap RTL-SDR dongle plus a kid-built
loop makes a superb dedicated receiver, and receiving is unrestricted.

Rules honesty: FRS is a licence-exempt *voice* service in both Canada
(ISED) and the US (FCC), and the provisions for non-voice content are
narrow and change over time — check the current wording before making it a
habit, keep transmissions short (another argument for the high factors),
identify, pick a quiet channel, listen first, and always yield to voice
traffic. Supervised kids sending 15-second pictures across town,
courteously, is very much in the spirit of why these services exist — and
it is a gateway drug to the ham licence.

## Practical notes

Feed and take audio at sane levels — the decoder is FM-based and tolerates a
lot, but clipping in a soundcard or an over-driven mic input hurts the
stretch mode disproportionately. On SSB use FM turbo. 2× and 3× are
essentially free; 4× is comfortable on any decent path; 5× through 8× lock
reliably in the simulator down to ~10–12 dB SNR — a two-minute Martin M1
frame in 14.5 seconds at 8× — but the SSB filter is eating real sideband
energy at those rates, so expect gradually softer horizontal detail and
step up only as conditions allow. Keep the receiver's filter as wide as it goes. If VOX
triggers on band noise, raise the threshold slider until the level meter's
marker sits above the idle level. The vocoder briefly allocates a few
hundred megabytes while restoring the long modes on the RX side; on a
Raspberry-Pi-class machine prefer Robot 36/72 or Martin M2. And since the
whole point is a nonstandard waveform on the air: announce what you're
doing, keep your callsign in the image and in your ID practice, and check
your local rules about experimental transmissions.

## Files

* `sstv_studio.py` — everything: GUI, encoder, decoder, vocoder, channel
  simulator, self-tests.
* `requirements.txt` — Python dependencies.
