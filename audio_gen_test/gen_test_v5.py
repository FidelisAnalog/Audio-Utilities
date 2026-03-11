#!/usr/bin/env python3
"""
Audio generation test - measure compound DAC/ADC degradation.

v5: Chirp marker alignment. Prepends three 800→1200 Hz chirps to the audio
    as alignment markers. Each generation plays [marker + music + padding],
    records, and cross-correlates each chirp against the original clean
    template for three independent offset estimates. The marker is always
    generated fresh (never degraded), so alignment quality is constant
    regardless of generation count. Sub-sample correction via FFT phase shift.

Saves milestone files at specified intervals (music only, markers stripped).

Usage:
    python gen_test_v5.py source.wav --generations 100 --milestone 10 --output results/
"""

__version__ = "0.6.0"

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import soundfile as sf
import numpy as np

from backends import get_backend, StreamConfig


def _to_mono_f64(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono float64 for alignment calculations."""
    d = audio.astype(np.float64)
    if d.ndim > 1:
        return (d[:, 0] + d[:, 1]) / 2.0 if d.shape[1] >= 2 else d[:, 0]
    return d


# ---------------------------------------------------------------------------
# Chirp marker generation
# ---------------------------------------------------------------------------

def generate_chirp(f_start: float, f_end: float, duration_s: float,
                   sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a linear frequency sweep (chirp).

    Args:
        f_start: Start frequency in Hz
        f_end: End frequency in Hz
        duration_s: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Peak amplitude (linear, 0-1). Default 0.5 ≈ -6 dBFS.

    Returns:
        Chirp signal as mono float64 numpy array
    """
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate
    # Instantaneous frequency sweeps linearly from f_start to f_end
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration_s))
    chirp = amplitude * np.sin(phase)

    # 5ms fade in/out to avoid clicks at boundaries
    fade_len = int(sample_rate * 0.005)
    if fade_len > 0 and 2 * fade_len < n_samples:
        chirp[:fade_len] *= np.linspace(0, 1, fade_len)
        chirp[-fade_len:] *= np.linspace(1, 0, fade_len)

    return chirp


def generate_marker(sample_rate: int) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    """
    Generate alignment marker: three identical 800→1200 Hz chirps with known spacing.

    Structure:
        [200ms silence] [chirp] [200ms gap] [chirp] [200ms gap] [chirp] [1s silence]
        Total: ~2.0 seconds

    Returns:
        marker_mono: Full marker signal (mono, float64)
        chirp_template: Single chirp for correlation (mono, float64)
        chirp_starts: Sample offset of each chirp start within the marker
        marker_len: Total marker length in samples
    """
    chirp_duration = 0.2    # 200ms per chirp
    gap_duration = 0.2      # 200ms between chirps
    lead_duration = 0.2     # 200ms silence before first chirp
    tail_duration = 1.0     # 1s silence after last chirp (separates marker from music)

    chirp = generate_chirp(800, 1200, chirp_duration, sample_rate)
    chirp_len = len(chirp)
    gap = np.zeros(int(sample_rate * gap_duration))
    gap_len = len(gap)
    lead = np.zeros(int(sample_rate * lead_duration))
    lead_len = len(lead)
    tail = np.zeros(int(sample_rate * tail_duration))

    marker_mono = np.concatenate([lead, chirp, gap, chirp, gap, chirp, tail])

    chirp_starts = [
        lead_len,
        lead_len + chirp_len + gap_len,
        lead_len + 2 * (chirp_len + gap_len),
    ]

    return marker_mono, chirp.copy(), chirp_starts, len(marker_mono)


def marker_to_audio(marker_mono: np.ndarray, channels: int, dtype) -> np.ndarray:
    """Convert mono float64 marker to multi-channel target dtype."""
    if dtype in (np.int16, np.int32):
        max_val = np.iinfo(dtype).max
        marker_ch = (marker_mono * max_val).astype(dtype)
    else:
        marker_ch = marker_mono.astype(dtype)

    if channels > 1:
        return np.column_stack([marker_ch] * channels)
    else:
        return marker_ch.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Chirp-based alignment
# ---------------------------------------------------------------------------

def find_marker_offset(rec_mono: np.ndarray, chirp_template: np.ndarray,
                       chirp_starts: list[int], marker_len: int,
                       sample_rate: int,
                       expected_offset: int = None) -> tuple[int, float, float, list[float]]:
    """
    Find where the music starts in a recording using chirp marker correlation.

    Cross-correlates the chirp template against the recording three times
    (once per chirp). Each correlation peak gives an independent estimate
    of the marker start position. The three estimates are averaged, with
    sub-sample refinement via parabolic interpolation on each peak.

    IMPORTANT: Search regions are tightly constrained to the marker area
    to prevent false peaks from music content correlating with the chirp
    template (especially in the 800-1200 Hz range).

    Args:
        rec_mono: Recorded audio (mono, float64)
        chirp_template: Original chirp signal to correlate against
        chirp_starts: Sample offsets of each chirp within the marker
        marker_len: Total marker length in samples
        sample_rate: Sample rate in Hz
        expected_offset: Expected music start (from gen 1), narrows search

    Returns:
        Tuple of (music_start_int, frac_offset, chirp_spread, chirp_details)
        - music_start_int: integer sample where music starts in recording
        - frac_offset: sub-sample correction (fractional part)
        - chirp_spread: max difference between chirp estimates (samples)
        - chirp_details: per-chirp marker-start estimates (for diagnostics)
    """
    chirp_len = len(chirp_template)

    # Determine search region — must NOT extend into music content.
    # The recording structure is: [silence/latency] [marker] [music] [...]
    # We only want to search where the marker could be.
    if expected_offset is not None:
        # Gen 2+: we know where the marker is (±small variation)
        est_marker_start = expected_offset - marker_len
        search_start = max(0, est_marker_start - sample_rate // 4)  # 250ms before marker
        search_end = min(len(rec_mono), expected_offset + 500)       # tiny overshoot past marker end
    else:
        # Gen 1: marker is within first (max_latency + marker_len).
        # Assume max hardware latency of 1.5s (very generous).
        search_start = 0
        search_end = min(len(rec_mono), int(sample_rate * 1.5) + marker_len)

    search_region = rec_mono[search_start:search_end].copy()
    search_region -= np.mean(search_region)  # Remove DC bias from ADC

    # FFT cross-correlation: chirp template vs search region
    n = len(search_region) + chirp_len - 1
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    CHIRP = np.fft.rfft(chirp_template, fft_size)
    REC = np.fft.rfft(search_region, fft_size)
    xcorr = np.fft.irfft(CHIRP.conj() * REC, fft_size)

    # Only valid lags (chirp fully within search region)
    valid_len = len(search_region) - chirp_len + 1
    if valid_len < 1:
        valid_len = len(search_region)

    # Find each chirp's position.
    # Chirp 1: broader search (but still within marker region).
    # Chirps 2,3: very tight search — spacing within the marker is exact,
    # so the only variation is correlation peak jitter (< few samples).
    marker_start_estimates = []

    for i, cs in enumerate(chirp_starts):
        if i == 0:
            if expected_offset is not None:
                # Gen 2+: narrow search for chirp 1 around known position
                est_pos = int(expected_offset - marker_len - search_start) + cs
                region_start = max(0, est_pos - sample_rate // 10)  # ±100ms
                region_end = min(valid_len, est_pos + sample_rate // 10)
            else:
                # Gen 1: first chirp within first ~1.5s of recording
                region_start = 0
                region_end = min(valid_len, int(sample_rate * 1.5))
        else:
            # Chirps 2,3: exact spacing from chirp 1, ±100 samples for peak jitter
            est_marker_in_search = marker_start_estimates[0] - search_start
            expected_lag = int(est_marker_in_search) + cs
            region_start = max(0, expected_lag - 100)
            region_end = min(valid_len, expected_lag + 100)

        # Safety: ensure valid search region
        if region_end <= region_start:
            region_start = max(0, region_start - 500)
            region_end = min(valid_len, region_end + 500)

        region = xcorr[region_start:region_end]
        peak_in_region = np.argmax(region)
        peak_idx = region_start + peak_in_region

        # Parabolic sub-sample interpolation on correlation peak
        frac = 0.0
        if 0 < peak_idx < valid_len - 1:
            y_l = xcorr[peak_idx - 1]
            y_c = xcorr[peak_idx]
            y_r = xcorr[peak_idx + 1]
            denom = y_l - 2 * y_c + y_r
            if abs(denom) > 1e-12 and y_c > y_l and y_c > y_r:
                frac = 0.5 * (y_l - y_r) / denom

        # This chirp was found at (search_start + peak_idx + frac) in the recording.
        # The marker starts at that position minus this chirp's offset within the marker.
        chirp_pos_in_recording = search_start + peak_idx + frac
        marker_start = chirp_pos_in_recording - cs
        marker_start_estimates.append(marker_start)

    # Average the three marker-start estimates
    estimates = np.array(marker_start_estimates)
    spread = float(np.max(estimates) - np.min(estimates))

    # If spread is suspiciously large (> 5 samples), an outlier slipped through.
    # Use the two closest estimates and reject the outlier.
    if spread > 5.0 and len(estimates) == 3:
        diffs = [
            abs(estimates[0] - estimates[1]),
            abs(estimates[0] - estimates[2]),
            abs(estimates[1] - estimates[2]),
        ]
        best_pair = [(0, 1), (0, 2), (1, 2)][int(np.argmin(diffs))]
        mean_marker_start = float(np.mean(estimates[list(best_pair)]))
        spread = float(min(diffs))
    else:
        mean_marker_start = float(np.mean(estimates))

    # Music starts after the marker
    music_start = mean_marker_start + marker_len
    int_music_start = int(round(music_start))
    frac_offset = music_start - int_music_start

    return int_music_start, frac_offset, spread, [float(e) for e in estimates]


# ---------------------------------------------------------------------------
# FFT cross-correlation (kept for calibration only)
# ---------------------------------------------------------------------------

def fft_xcorr_offset(ref_mono: np.ndarray, test_mono: np.ndarray,
                     sample_rate: int, max_offset: int = 0) -> int:
    """
    FFT cross-correlation for rough sample offset.
    Used for calibration alignment (single-pass, not degraded).
    """
    if max_offset <= 0:
        max_offset = sample_rate // 2

    margin = sample_rate
    usable = min(len(ref_mono), len(test_mono)) - margin * 2 - max_offset
    if usable < sample_rate:
        margin = min(len(ref_mono), len(test_mono)) // 8
        usable = min(len(ref_mono), len(test_mono)) - margin * 2 - max_offset

    seg_len = max(usable, sample_rate)
    start = margin

    ref_seg = ref_mono[start:start + seg_len].copy()
    t_start = max(0, start - max_offset)
    t_end = min(len(test_mono), start + seg_len + max_offset)
    test_seg = test_mono[t_start:t_end].copy()

    ref_seg -= np.mean(ref_seg)
    test_seg -= np.mean(test_seg)

    n = len(ref_seg) + len(test_seg) - 1
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    REF = np.fft.rfft(ref_seg, fft_size)
    TEST = np.fft.rfft(test_seg, fft_size)
    xcorr = np.fft.irfft(REF.conj() * TEST, fft_size)

    peak_idx = np.argmax(xcorr[:len(test_seg)])
    offset = peak_idx - (start - t_start)
    return offset


# ---------------------------------------------------------------------------
# MSE (for degradation tracking, not alignment)
# ---------------------------------------------------------------------------

def mse_at_offset(a_mono: np.ndarray, b_mono: np.ndarray, offset: int, margin: int = 0) -> float:
    """
    Compute MSE between two signals at a given offset.
    Used to track degradation over generations (not for alignment).
    """
    a_start = margin
    b_start = margin + offset

    if b_start < 0:
        a_start += -b_start
        b_start = 0

    available = min(len(a_mono) - a_start, len(b_mono) - b_start) - margin
    if available <= 0:
        return np.inf

    a_seg = a_mono[a_start:a_start + available]
    b_seg = b_mono[b_start:b_start + available]
    return float(np.mean((a_seg - b_seg) ** 2))


# ---------------------------------------------------------------------------
# Sub-sample correction
# ---------------------------------------------------------------------------

def apply_fractional_shift(audio: np.ndarray, frac_samples: float) -> np.ndarray:
    """
    Shift audio by a fractional number of samples using FFT phase rotation.
    Exact for bandlimited signals. Preserves original dtype.
    """
    if abs(frac_samples) < 1e-6:
        return audio

    orig_dtype = audio.dtype
    was_int = orig_dtype in (np.int16, np.int32)

    if was_int:
        max_val = np.iinfo(orig_dtype).max
        work = audio.astype(np.float64) / max_val
    else:
        work = audio.astype(np.float64)

    if work.ndim == 1:
        work = work.reshape(-1, 1)
        was_1d = True
    else:
        was_1d = False

    n = work.shape[0]
    freqs = np.fft.rfftfreq(n)
    phase_shift = np.exp(-2j * np.pi * freqs * frac_samples)

    result = np.zeros_like(work)
    for ch in range(work.shape[1]):
        spectrum = np.fft.rfft(work[:, ch])
        result[:, ch] = np.fft.irfft(spectrum * phase_shift, n)

    if was_1d:
        result = result[:, 0]

    if was_int:
        max_val = np.iinfo(orig_dtype).max
        min_val = np.iinfo(orig_dtype).min
        result = np.clip(result * max_val, min_val, max_val).astype(orig_dtype)
    else:
        result = result.astype(orig_dtype)

    return result


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def extract_aligned_audio(recorded: np.ndarray, offset: int, target_len: int) -> np.ndarray:
    """Extract aligned audio from recording starting at offset."""
    extracted = recorded[offset:offset + target_len]

    # Pad if needed (shouldn't happen with sufficient padding)
    if len(extracted) < target_len:
        pad_len = target_len - len(extracted)
        channels = extracted.shape[1] if extracted.ndim > 1 else 1
        pad = np.zeros((pad_len, channels), dtype=extracted.dtype)
        extracted = np.vstack([extracted, pad])

    return extracted


def measure_levels(audio: np.ndarray) -> tuple[float, float]:
    """Measure RMS and peak levels of audio."""
    if audio.dtype in (np.int16, np.int32):
        max_val = np.iinfo(audio.dtype).max
        audio_float = audio.astype(np.float64) / max_val
    else:
        audio_float = audio.astype(np.float64)

    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))
    return rms, peak


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """Apply linear gain to audio, preserving dtype and clipping if needed."""
    if audio.dtype in (np.int16, np.int32):
        max_val = np.iinfo(audio.dtype).max
        min_val = np.iinfo(audio.dtype).min
        scaled = audio.astype(np.float64) * gain
        clipped = np.clip(scaled, min_val, max_val)
        return clipped.astype(audio.dtype)
    else:
        return audio * gain


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_io_gain(backend, config, sample_rate: int, channels: int, dtype) -> float:
    """
    Measure I/O gain by playing pink noise and comparing levels.
    Uses simple FFT xcorr for alignment (single pass, no degradation).
    """
    from scipy import signal as sig

    duration = 2.0
    num_samples = int(sample_rate * duration)

    # Generate pink noise (Voss-McCartney)
    num_octaves = 16
    pink = np.zeros(num_samples)
    for i in range(num_octaves):
        step = 2 ** i
        noise = np.random.randn((num_samples // step) + 1)
        pink += np.repeat(noise, step)[:num_samples]

    # Bandlimit 20Hz-20kHz
    nyquist = sample_rate / 2
    low = 20 / nyquist
    high = min(20000 / nyquist, 0.99)
    sos = sig.butter(4, [low, high], btype='band', output='sos')
    pink = sig.sosfilt(sos, pink)

    # Normalize to -20dBFS RMS
    pink = pink / np.sqrt(np.mean(pink ** 2))
    amplitude = 10 ** (-20 / 20)
    pink = pink * amplitude

    # Convert to target dtype
    if dtype in (np.int16, np.int32):
        max_val = np.iinfo(dtype).max
        test_signal = (pink * max_val).astype(dtype)
    else:
        test_signal = pink.astype(dtype)

    # Make multi-channel
    if channels > 1:
        test_signal = np.column_stack([test_signal] * channels)
    else:
        test_signal = test_signal.reshape(-1, 1)

    # Pad for latency
    pad_samples = sample_rate // 2
    padding = np.zeros((pad_samples, channels), dtype=dtype)
    padded_signal = np.vstack([test_signal, padding])

    # Play and record
    recorded = backend.playrec(padded_signal, config)

    # Simple FFT xcorr alignment (fine for single-pass calibration)
    ref_mono = _to_mono_f64(test_signal)
    rec_mono = _to_mono_f64(recorded)
    offset = fft_xcorr_offset(ref_mono, rec_mono, sample_rate)
    aligned = extract_aligned_audio(recorded, offset, num_samples)

    # Compare RMS
    ref_rms, _ = measure_levels(test_signal)
    rec_rms, _ = measure_levels(aligned)

    if rec_rms > 0:
        gain = ref_rms / rec_rms
        gain_db = 20 * np.log10(gain)
        print(f"  Calibration: ref={20*np.log10(ref_rms):.4f}dBFS, "
              f"rec={20*np.log10(rec_rms):.4f}dBFS, correction={gain_db:+.4f}dB")
        return gain
    else:
        print(f"  Calibration failed: no signal recorded")
        return 1.0


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def list_devices():
    """List available audio devices and return categorized lists."""
    backend = get_backend()
    devices = backend.enumerate_devices()

    outputs = [d for d in devices if d.max_channels_out > 0]
    inputs = [d for d in devices if d.max_channels_in > 0]

    print("\nOUTPUT DEVICES:")
    for i, d in enumerate(outputs):
        print(f"  [{i}] {d.name} ({d.max_channels_out}ch)")

    print("\nINPUT DEVICES:")
    for i, d in enumerate(inputs):
        print(f"  [{i}] {d.name} ({d.max_channels_in}ch)")

    return outputs, inputs


def select_device(devices: list, prompt: str, preselect: int = None) -> int:
    """Interactive device selection."""
    if preselect is not None and 0 <= preselect < len(devices):
        print(f"{prompt}: [{preselect}] {devices[preselect].name}")
        return preselect

    while True:
        try:
            idx = int(input(f"{prompt}: "))
            if 0 <= idx < len(devices):
                return idx
            print(f"Please enter 0-{len(devices) - 1}")
        except ValueError:
            print("Please enter a valid number")


# ---------------------------------------------------------------------------
# Main generation test
# ---------------------------------------------------------------------------

def run_generation_test(
    source_file: str,
    output_dir: str,
    generations: int,
    milestone_interval: int,
    output_device_idx: int = None,
    input_device_idx: int = None,
    compensate: str = None,
    shared_mode: bool = False,
    output_channels: list = None,
    input_channels: list = None,
    per_channel: bool = False,
):
    """
    Run the generation test with chirp marker alignment.

    Each generation:
      1. Prepend fresh (clean) chirp marker to current music
      2. Play marker+music+padding through DAC, record through ADC
      3. Cross-correlate each chirp against original template → 3 offset estimates
      4. Average offsets, extract music portion (marker stripped)
      5. Apply sub-sample correction, level compensation
      6. Save milestone if applicable; feed music to next generation
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read source file
    print(f"\nReading source: {source_file}")
    info = sf.info(source_file)

    subtype = info.subtype
    if 'PCM_16' in subtype:
        bit_depth = 16
        read_dtype = 'int16'
    elif 'PCM_24' in subtype:
        bit_depth = 24
        read_dtype = 'int32'
    elif 'PCM_32' in subtype:
        bit_depth = 32
        read_dtype = 'int32'
    elif 'FLOAT' in subtype:
        bit_depth = 32
        read_dtype = 'float32'
    else:
        bit_depth = 24
        read_dtype = 'int32'

    data, sample_rate = sf.read(source_file, dtype=read_dtype)

    channels = data.shape[1] if data.ndim > 1 else 1
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    duration = len(data) / sample_rate

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Bit depth: {bit_depth}")
    print(f"  Channels: {channels}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Generations: {generations}")
    if milestone_interval > 0:
        print(f"  Milestones: every {milestone_interval} generations")

    # Reference levels
    ref_rms, ref_peak = measure_levels(data)
    ref_rms_db = 20 * np.log10(ref_rms) if ref_rms > 0 else -np.inf
    ref_peak_db = 20 * np.log10(ref_peak) if ref_peak > 0 else -np.inf
    print(f"  Reference RMS: {ref_rms_db:.4f} dBFS")
    print(f"  Reference Peak: {ref_peak_db:.4f} dBFS")

    # Per-channel reference RMS
    ref_rms_per_ch = None
    if per_channel and channels > 1:
        ref_rms_per_ch = []
        for ch in range(channels):
            ch_rms, _ = measure_levels(data[:, ch:ch+1])
            ref_rms_per_ch.append(ch_rms)
            ch_rms_db = 20 * np.log10(ch_rms) if ch_rms > 0 else -np.inf
            print(f"  Reference RMS ch{ch+1}: {ch_rms_db:.4f} dBFS")

    # Generate alignment marker
    marker_mono, chirp_template, chirp_starts, marker_len = generate_marker(int(sample_rate))
    marker_duration = marker_len / sample_rate
    marker_audio = marker_to_audio(marker_mono, channels, data.dtype)

    print(f"  Alignment: chirp marker ({marker_duration:.2f}s, 3x 800→1200Hz @ -6dBFS)")
    print(f"  Sub-sample correction: enabled")
    if compensate:
        print(f"  Level compensation: {compensate.upper()}{' (per-channel)' if per_channel else ''}")

    # List and select devices
    outputs, inputs = list_devices()

    out_idx = select_device(outputs, "\nSelect OUTPUT device", output_device_idx)
    in_idx = select_device(inputs, "Select INPUT device", input_device_idx)

    output_device = outputs[out_idx]
    input_device = inputs[in_idx]

    # Configure stream
    config = StreamConfig(
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
    )

    # Create run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gen_{Path(source_file).stem}_{run_timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(exist_ok=True)

    # Save metadata
    meta_file = run_dir / "metadata.txt"
    with open(meta_file, 'w') as f:
        f.write(f"Version: {__version__}\n")
        f.write(f"Source: {source_file}\n")
        f.write(f"Sample rate: {sample_rate}\n")
        f.write(f"Bit depth: {bit_depth}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Samples: {len(data)}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Generations: {generations}\n")
        f.write(f"Milestone interval: {milestone_interval}\n")
        f.write(f"Alignment: chirp marker (3x 800-1200Hz, {marker_duration:.3f}s, -6dBFS)\n")
        f.write(f"Sub-sample correction: enabled\n")
        f.write(f"Level compensation: {compensate}\n")
        f.write(f"Reference RMS: {ref_rms_db:.4f} dBFS\n")
        f.write(f"Reference Peak: {ref_peak_db:.4f} dBFS\n")
        f.write(f"Output device: {output_device.name}\n")
        f.write(f"Input device: {input_device.name}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")

    # Save source (music only, no marker)
    source_copy = run_dir / f"gen_000_{Path(source_file).stem}.wav"
    sf.write(source_copy, data, sample_rate, subtype=info.subtype)
    print(f"\nSource saved: {source_copy}")

    # Run generations
    backend = get_backend()
    backend.shared_mode = shared_mode
    backend.output_channels = output_channels if output_channels else list(range(1, channels + 1))
    backend.input_channels = input_channels if input_channels else list(range(1, channels + 1))
    current_music = data.copy()

    try:
        print(f"\nConfiguring devices...")
        backend.configure_output(output_device, config)
        backend.configure_input(input_device, config)

        # Padding at end to ensure full music capture despite hardware latency
        pad_samples = sample_rate  # 1 second (generous for high-latency setups)
        padding = np.zeros((pad_samples, channels), dtype=data.dtype)

        # Calibrate I/O gain if using calibrated mode
        io_gain = 1.0
        if compensate == 'calibrated':
            print(f"\nCalibrating I/O gain...")
            io_gain = calibrate_io_gain(backend, config, int(sample_rate), channels, data.dtype)
            with open(meta_file, 'a') as f:
                f.write(f"Calibrated I/O gain: {20*np.log10(io_gain):+.4f} dB\n")

        estimated_total = generations * (duration + marker_duration)
        print(f"\nStarting {generations} generations (~{estimated_total/60:.1f} min of audio)")
        print("-" * 80)

        with open(meta_file, 'a') as f:
            f.write(f"\nStarting {generations} generations\n")
            f.write("-" * 80 + "\n")

        target_len = len(data)
        original_data = data.copy()  # Preserved gen 0 for MSE tracking
        initial_music_offset = None  # From gen 1, used to narrow subsequent searches

        for gen in range(1, generations + 1):
            # Build playback signal: fresh marker + current music + padding
            playback = np.vstack([marker_audio, current_music, padding])

            # Play and record
            recorded = backend.playrec(playback, config)

            # Find alignment via chirp markers
            rec_mono = _to_mono_f64(recorded)
            music_offset, frac_offset, chirp_spread, chirp_details = find_marker_offset(
                rec_mono, chirp_template, chirp_starts, marker_len,
                int(sample_rate),
                expected_offset=initial_music_offset
            )

            if initial_music_offset is None:
                initial_music_offset = music_offset

            # Extract music portion (marker stripped)
            aligned = extract_aligned_audio(recorded, music_offset, target_len)

            # Sub-sample correction via FFT phase shift
            if abs(frac_offset) > 1e-6:
                aligned = apply_fractional_shift(aligned, -frac_offset)

            # Measure levels
            gen_rms, gen_peak = measure_levels(aligned)
            gen_rms_db = 20 * np.log10(gen_rms) if gen_rms > 0 else -np.inf
            gen_peak_db = 20 * np.log10(gen_peak) if gen_peak > 0 else -np.inf
            rms_delta = gen_rms_db - ref_rms_db

            # MSE vs original (degradation tracking)
            ref_mono = _to_mono_f64(original_data)
            aligned_mono = _to_mono_f64(aligned)
            track_mse = mse_at_offset(ref_mono, aligned_mono, 0, int(sample_rate))

            # Level compensation
            gain_db = 0.0
            gains_per_ch = None
            if compensate == 'dynamic' and gen_rms > 0:
                if per_channel and ref_rms_per_ch and channels > 1:
                    gains_per_ch = []
                    for ch in range(channels):
                        ch_rms, _ = measure_levels(aligned[:, ch:ch+1])
                        if ch_rms > 0:
                            ch_gain = ref_rms_per_ch[ch] / ch_rms
                            aligned[:, ch] = apply_gain(aligned[:, ch:ch+1], ch_gain).flatten()
                            gains_per_ch.append(20*np.log10(ch_gain))
                        else:
                            gains_per_ch.append(0.0)
                    gain_db = sum(gains_per_ch) / len(gains_per_ch)
                else:
                    gain = ref_rms / gen_rms
                    aligned = apply_gain(aligned, gain)
                    gain_db = 20*np.log10(gain)
            elif compensate == 'calibrated':
                aligned = apply_gain(aligned, io_gain)
                gain_db = 20*np.log10(io_gain)

            # Check milestones
            is_milestone = milestone_interval > 0 and gen % milestone_interval == 0
            is_final = gen == generations
            filename = None

            # Console output
            if compensate == 'dynamic':
                rms_str = f"rms:{gen_rms_db:.2f}dB"
                if gains_per_ch:
                    gain_str = f"  gain:{'/'.join(f'{g:+.2f}' for g in gains_per_ch)}"
                else:
                    gain_str = f"  gain:{gain_db:+.2f}"
            elif compensate == 'calibrated':
                rms_str = f"rms:{gen_rms_db:.2f}dB({rms_delta:+.2f})"
                gain_str = f"  gain:{gain_db:+.2f}"
            else:
                rms_str = f"rms:{gen_rms_db:.2f}dB({rms_delta:+.2f})"
                gain_str = ""

            console_line = (
                f"{gen:3d}/{generations}  {rms_str}  pk:{gen_peak_db:.2f}{gain_str}"
                f"  off:{music_offset}{frac_offset:+.3f} sprd:{chirp_spread:.2f} mse:{track_mse:.1f}"
            )

            if is_milestone or is_final:
                filename = f"gen_{gen:03d}.wav"
                filepath = run_dir / filename
                sf.write(filepath, aligned, sample_rate, subtype=info.subtype)
                console_line += f"  {filename}"

            print(console_line)

            # Metadata log (full precision)
            if compensate:
                if gains_per_ch:
                    gain_str_meta = '/'.join(f'{g:+.4f}' for g in gains_per_ch)
                    meta_line = (
                        f"{gen:3d}/{generations}  rms:{gen_rms_db:.4f}dB({rms_delta:+.4f})"
                        f"  pk:{gen_peak_db:.4f}  gain:{gain_str_meta}"
                        f"  off:{music_offset}{frac_offset:+.4f}"
                        f" sprd:{chirp_spread:.4f} mse:{track_mse:.4f}"
                    )
                else:
                    meta_line = (
                        f"{gen:3d}/{generations}  rms:{gen_rms_db:.4f}dB({rms_delta:+.4f})"
                        f"  pk:{gen_peak_db:.4f}  gain:{gain_db:+.4f}"
                        f"  off:{music_offset}{frac_offset:+.4f}"
                        f" sprd:{chirp_spread:.4f} mse:{track_mse:.4f}"
                    )
            else:
                meta_line = (
                    f"{gen:3d}/{generations}  rms:{gen_rms_db:.4f}dB({rms_delta:+.4f})"
                    f"  pk:{gen_peak_db:.4f}"
                    f"  off:{music_offset}{frac_offset:+.4f}"
                    f" sprd:{chirp_spread:.4f} mse:{track_mse:.4f}"
                )
            if filename:
                meta_line += f"  {filename}"
            with open(meta_file, 'a') as f:
                f.write(meta_line + "\n")

            # Feed music to next generation
            current_music = aligned

        print("-" * 80)
        print(f"\nTest complete! Results saved to: {run_dir}")

        with open(meta_file, 'a') as f:
            f.write("-" * 80 + "\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at generation {gen}")
        interrupt_file = run_dir / f"gen_{gen:03d}_interrupted.wav"
        sf.write(interrupt_file, current_music, sample_rate, subtype=info.subtype)
        print(f"Current state saved: {interrupt_file}")

        with open(meta_file, 'a') as f:
            f.write("-" * 80 + "\n")
            f.write(f"Interrupted at generation: {gen}\n")
            f.write(f"Interrupted: {datetime.now().isoformat()}\n")

    finally:
        backend.release()

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audio generation test for DAC/ADC degradation (v5: chirp marker alignment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 100 generations, save every 10
    python gen_test_v5.py sweep.wav -g 100 -m 10 -o results/

    # Save every generation (for detailed analysis)
    python gen_test_v5.py sweep.wav -g 100 -m 1 -o results/

    # Run with dynamic level compensation
    python gen_test_v5.py sweep.wav -g 100 -m 10 -c dynamic -o results/

    # Run with calibrated level compensation
    python gen_test_v5.py sweep.wav -g 100 -m 10 -c calibrated -o results/
        """
    )

    parser.add_argument('source', help="Source WAV file")
    parser.add_argument('-g', '--generations', type=int, default=10,
                        help="Number of generations (default: 10)")
    parser.add_argument('-m', '--milestone', type=int, default=0,
                        help="Save milestone every N generations (default: 0 = none)")
    parser.add_argument('-o', '--output', default='./results',
                        help="Output directory (default: ./results)")
    parser.add_argument('--output-device', type=int, default=None,
                        help="Output device index (skip selection)")
    parser.add_argument('--input-device', type=int, default=None,
                        help="Input device index (skip selection)")
    parser.add_argument('--list', '-l', action='store_true',
                        help="List devices and exit")
    parser.add_argument('-c', '--compensate', choices=['dynamic', 'calibrated'], default=None,
                        help="Level compensation: 'dynamic' (per-iteration RMS) or 'calibrated' (fixed I/O gain)")
    parser.add_argument('--shared', action='store_true',
                        help="Use WASAPI shared mode (Windows only, for virtual devices)")
    parser.add_argument('--output-channels', type=str, default=None,
                        help="Output channel mapping, e.g. '11,12' (default: 1,2)")
    parser.add_argument('--input-channels', type=str, default=None,
                        help="Input channel mapping, e.g. '11,12' (default: 1,2)")
    parser.add_argument('-p', '--per-channel', action='store_true',
                        help="Apply level compensation per channel (for unlinked hardware gains)")
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    if args.list:
        list_devices()
        return 0

    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return 1

    # Parse channel mappings
    output_channels = None
    if args.output_channels:
        output_channels = [int(c) for c in args.output_channels.split(',')]

    input_channels = None
    if args.input_channels:
        input_channels = [int(c) for c in args.input_channels.split(',')]

    run_generation_test(
        source_file=args.source,
        output_dir=args.output,
        generations=args.generations,
        milestone_interval=args.milestone,
        output_device_idx=args.output_device,
        input_device_idx=args.input_device,
        compensate=args.compensate,
        shared_mode=args.shared,
        output_channels=output_channels,
        input_channels=input_channels,
        per_channel=args.per_channel,
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
