#!/usr/bin/env python3
"""
FLAC null test - empirical demonstration that FLAC decodes bit-identically to WAV.

Plays a WAV file and a FLAC file (encoded from the same WAV) through a DAC,
captures the loopback via ADC, and nulls the two captures. A control pass
(WAV played twice) establishes the analog noise floor baseline.

All three passes use callback-based streaming playback to simulate real-world
progressive decode behavior.

Usage:
    python flac_null_test.py source.wav source.flac -o results/
"""

__version__ = "0.1.0"

import argparse
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf

from backends import get_backend, StreamConfig


# --- Forked from gen_test.py ---

def find_alignment_offset(reference: np.ndarray, recorded: np.ndarray, sample_rate: int, cumulative_subsample: float = 0.0) -> tuple[int, float]:
    """
    Find exact sample offset between reference and recorded using cross-correlation
    with sub-sample interpolation.

    Args:
        reference: Original audio played
        recorded: Audio recorded (should be longer due to padding)
        sample_rate: For determining search window
        cumulative_subsample: Running total of sub-sample offsets from previous generations

    Returns:
        Tuple of (integer_offset, new_cumulative_subsample)
    """
    from scipy import signal

    # Use first channel
    ref_mono = reference[:, 0] if reference.ndim > 1 else reference
    rec_mono = recorded[:, 0] if recorded.ndim > 1 else recorded

    # Convert to float for correlation
    ref_float = ref_mono.astype(np.float64)
    rec_float = rec_mono.astype(np.float64)

    # Use full correlation
    correlation = signal.correlate(rec_float, ref_float, mode='full')

    # The peak position relative to where zero-lag would be
    # In 'full' mode, zero-lag is at index len(ref) - 1
    peak_idx = np.argmax(correlation)
    integer_offset = peak_idx - (len(ref_float) - 1)

    # Parabolic interpolation for sub-sample refinement
    # Standard formula: d = (y_plus - y_minus) / (2 * (2*y_peak - y_minus - y_plus))
    if peak_idx > 0 and peak_idx < len(correlation) - 1:
        y_minus = correlation[peak_idx - 1]
        y_peak = correlation[peak_idx]
        y_plus = correlation[peak_idx + 1]

        denom = y_minus - 2*y_peak + y_plus
        if abs(denom) > 1e-10:
            # Positive subsample = true peak is to the right of integer peak
            # Negative subsample = true peak is to the left
            subsample = 0.5 * (y_plus - y_minus) / (-denom)
        else:
            subsample = 0.0
    else:
        subsample = 0.0

    # Add this generation's sub-sample offset to cumulative total
    new_cumulative = cumulative_subsample + subsample

    # When cumulative exceeds +/-0.5, adjust integer offset
    # Note: empirically the subsample detection has inverted sign,
    # so we invert the adjustment direction
    adjustment = 0
    if new_cumulative <= -0.5:
        adjustment = +1  # Trim more (audio arriving later than detected)
        new_cumulative += 1.0
    elif new_cumulative >= 0.5:
        adjustment = -1  # Trim less
        new_cumulative -= 1.0

    final_offset = max(0, integer_offset + adjustment)

    return final_offset, new_cumulative


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
    """
    Measure RMS and peak levels of audio.

    Returns:
        Tuple of (rms, peak) as linear values
    """
    # Convert to float if needed
    if audio.dtype in (np.int16, np.int32):
        max_val = np.iinfo(audio.dtype).max
        audio_float = audio.astype(np.float64) / max_val
    else:
        audio_float = audio.astype(np.float64)

    rms = np.sqrt(np.mean(audio_float ** 2))
    peak = np.max(np.abs(audio_float))

    return rms, peak


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


# --- New functions ---

def get_sd_dtype(bit_depth: int) -> str:
    """Map bit depth to sounddevice dtype string."""
    if bit_depth == 16:
        return 'int16'
    else:
        return 'int32'  # 24-bit packed into int32, or native 32-bit


def streaming_playrec(
    file_path: str,
    sd_output_id: int,
    sd_input_id: int,
    config: StreamConfig,
    blocksize: int = 2048,
    extra_settings=None,
) -> tuple[np.ndarray, int]:
    """
    Play an audio file via callback-based streaming and record simultaneously.

    The callback reads from sf.SoundFile in chunks, simulating real-world
    progressive decode. For FLAC files, each callback invocation triggers
    on-the-fly FLAC frame decoding via libsndfile.

    Args:
        file_path: Path to WAV or FLAC file
        sd_output_id: sounddevice output device index
        sd_input_id: sounddevice input device index
        config: StreamConfig with sample_rate, bit_depth, channels
        blocksize: Callback buffer size in frames
        extra_settings: Optional sounddevice extra_settings (e.g. WasapiSettings)

    Returns:
        Tuple of (recorded audio array, xrun count)
    """
    dtype = get_sd_dtype(config.bit_depth)
    pad_frames = int(config.sample_rate * 0.5)  # 0.5s padding after EOF

    # Mutable state accessed by callback
    recording_chunks = []
    file_exhausted = False
    padding_remaining = pad_frames
    xrun_count = 0
    finished_event = threading.Event()

    sound_file = sf.SoundFile(file_path)

    def callback(indata, outdata, frames, time_info, status):
        nonlocal file_exhausted, padding_remaining, xrun_count

        if status:
            xrun_count += 1
            print(f"  Stream status: {status}", file=sys.stderr)

        # Always record input
        recording_chunks.append(indata.copy())

        # Output: read from file or zero-pad
        if not file_exhausted:
            data = sound_file.read(frames, dtype=dtype)
            n = len(data)
            if n == 0:
                # EOF reached
                file_exhausted = True
                outdata[:] = 0
                padding_remaining -= frames
            elif n < frames:
                # Partial read (final chunk before EOF)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                outdata[:n] = data
                outdata[n:] = 0
                file_exhausted = True
                padding_remaining -= (frames - n)
            else:
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                outdata[:] = data
        else:
            # Zero-padding phase
            outdata[:] = 0
            padding_remaining -= frames
            if padding_remaining <= 0:
                raise sd.CallbackStop

    stream = sd.Stream(
        samplerate=int(config.sample_rate),
        blocksize=blocksize,
        device=(sd_input_id, sd_output_id),
        channels=config.channels,
        dtype=dtype,
        callback=callback,
        finished_callback=finished_event.set,
        extra_settings=extra_settings,
    )
    try:
        with stream:
            finished_event.wait()
    finally:
        sound_file.close()

    # Concatenate all recorded chunks
    if recording_chunks:
        return np.concatenate(recording_chunks, axis=0), xrun_count
    else:
        return np.zeros((0, config.channels), dtype=dtype), xrun_count


def compute_null(capture_a: np.ndarray, capture_b: np.ndarray) -> np.ndarray:
    """Subtract two aligned captures, return normalized float64 residual in [-1.0, 1.0]."""
    a = capture_a.astype(np.float64)
    b = capture_b.astype(np.float64)

    # Normalize to [-1.0, 1.0] if integer format
    if capture_a.dtype in (np.int16, np.int32):
        max_val = np.iinfo(capture_a.dtype).max
        a = a / max_val
        b = b / max_val

    return a - b


def measure_null_depth(null_signal: np.ndarray, reference_signal: np.ndarray) -> float:
    """
    Measure null depth in dB relative to reference signal level.

    Both signals should be normalized float64 in [-1.0, 1.0].

    Returns:
        Null depth in dB (negative number; more negative = deeper null).
    """
    ref_rms = np.sqrt(np.mean(reference_signal ** 2))
    null_rms = np.sqrt(np.mean(null_signal ** 2))

    if null_rms == 0:
        return -np.inf  # Perfect null
    if ref_rms == 0:
        return 0.0  # No signal

    return 20 * np.log10(null_rms / ref_rms)


def parse_bit_depth(subtype: str) -> tuple[int, str]:
    """
    Parse bit depth and read dtype from soundfile subtype string.

    Returns:
        Tuple of (bit_depth, read_dtype)
    """
    if 'PCM_16' in subtype:
        return 16, 'int16'
    elif 'PCM_24' in subtype:
        return 24, 'int32'
    elif 'PCM_32' in subtype:
        return 32, 'int32'
    elif 'FLOAT' in subtype:
        return 32, 'float32'
    else:
        return 24, 'int32'


def run_null_test(
    wav_file: str,
    flac_file: str,
    output_dir: str,
    output_device_idx: int = None,
    input_device_idx: int = None,
    blocksize: int = 2048,
    shared_mode: bool = False,
):
    """
    Run the FLAC null test.

    Three passes:
      1. Play WAV  -> capture (wav1)
      2. Play FLAC -> capture (flac)
      3. Play WAV  -> capture (wav2, control)

    Two nulls:
      test_null    = wav1 - flac  (should be just DAC/ADC noise)
      control_null = wav1 - wav2  (establishes noise floor baseline)
    """
    # Validate and read file info
    wav_info = sf.info(wav_file)
    flac_info = sf.info(flac_file)

    wav_bits, wav_dtype = parse_bit_depth(wav_info.subtype)
    flac_bits, _ = parse_bit_depth(flac_info.subtype)

    # Validate format match
    if wav_info.samplerate != flac_info.samplerate:
        print(f"Error: Sample rate mismatch: WAV={wav_info.samplerate}, FLAC={flac_info.samplerate}")
        return 1
    if wav_info.channels != flac_info.channels:
        print(f"Error: Channel count mismatch: WAV={wav_info.channels}, FLAC={flac_info.channels}")
        return 1
    if wav_bits != flac_bits:
        print(f"Error: Bit depth mismatch: WAV={wav_bits}, FLAC={flac_bits}")
        return 1
    if wav_info.frames != flac_info.frames:
        print(f"Warning: Frame count mismatch: WAV={wav_info.frames}, FLAC={flac_info.frames}")
        print(f"  Files may not have been encoded from the same source.")

    sample_rate = wav_info.samplerate
    channels = wav_info.channels
    bit_depth = wav_bits
    duration = wav_info.frames / sample_rate

    print(f"\nFLAC Null Test")
    print(f"  WAV source:   {wav_file}")
    print(f"  FLAC source:  {flac_file}")
    print(f"  Sample rate:  {sample_rate} Hz")
    print(f"  Bit depth:    {bit_depth}")
    print(f"  Channels:     {channels}")
    print(f"  Duration:     {duration:.2f}s")
    print(f"  Blocksize:    {blocksize} frames")

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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"null_test_{run_timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(exist_ok=True)

    # Configure backend
    backend = get_backend()
    backend.shared_mode = shared_mode

    try:
        print(f"\nConfiguring devices...")
        backend.configure_output(output_device, config)
        backend.configure_input(input_device, config)

        sd_output_id = backend._sd_output_id
        sd_input_id = backend._sd_input_id

        # Determine extra_settings for Windows WASAPI
        extra_settings = None
        if sys.platform == 'win32':
            use_asio = getattr(backend, '_use_asio', False)
            if not use_asio and not shared_mode:
                extra_settings = sd.WasapiSettings(exclusive=True)

        # --- Three streaming passes ---
        pass_files = [
            ("WAV",  wav_file),
            ("FLAC", flac_file),
            ("WAV",  wav_file),
        ]

        captures = []
        offsets = []
        subsample_vals = []
        xrun_totals = []

        # Load reference for alignment (WAV source into memory)
        ref_data, _ = sf.read(wav_file, dtype=wav_dtype)
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(-1, 1)
        target_len = len(ref_data)

        print(f"\nStarting 3 passes (estimated time: {3 * duration / 60:.1f} minutes)")
        print("-" * 50)

        for i, (label, fpath) in enumerate(pass_files):
            print(f"Pass {i+1}/3: Playing {label}...", end="", flush=True)

            recorded, xruns = streaming_playrec(
                fpath, sd_output_id, sd_input_id, config,
                blocksize=blocksize, extra_settings=extra_settings,
            )

            # Align to reference
            offset, subsample = find_alignment_offset(ref_data, recorded, int(sample_rate), 0.0)
            aligned = extract_aligned_audio(recorded, offset, target_len)

            captures.append(aligned)
            offsets.append(offset)
            subsample_vals.append(subsample)
            xrun_totals.append(xruns)

            xrun_note = f"  xruns:{xruns}" if xruns > 0 else ""
            print(f" done (offset: +{offset}, subsample: {subsample:+.2f}{xrun_note})")

        wav1_aligned, flac_aligned, wav2_aligned = captures

        # --- Compute nulls ---
        test_null = compute_null(wav1_aligned, flac_aligned)
        control_null = compute_null(wav1_aligned, wav2_aligned)

        # --- Measure ---
        sig_rms, sig_peak = measure_levels(wav1_aligned)
        sig_rms_db = 20 * np.log10(sig_rms) if sig_rms > 0 else -np.inf
        sig_peak_db = 20 * np.log10(sig_peak) if sig_peak > 0 else -np.inf

        # Normalize reference for null depth measurement (nulls are already normalized)
        if wav1_aligned.dtype in (np.int16, np.int32):
            ref_normalized = wav1_aligned.astype(np.float64) / np.iinfo(wav1_aligned.dtype).max
        else:
            ref_normalized = wav1_aligned.astype(np.float64)

        test_depth = measure_null_depth(test_null, ref_normalized)
        control_depth = measure_null_depth(control_null, ref_normalized)

        # --- Report ---
        print("-" * 50)
        print(f"\nResults:")
        print(f"  Signal level:             {sig_rms_db:.2f} dBFS (RMS)  /  {sig_peak_db:.2f} dBFS (peak)")
        print(f"  Test null (WAV-FLAC):     {test_depth:.1f} dB")
        print(f"  Control null (WAV-WAV):   {control_depth:.1f} dB")

        if any(x > 0 for x in xrun_totals):
            print(f"\n  Warning: Buffer underruns detected (xruns: {xrun_totals}). Results may be affected.")

        # --- Save outputs ---
        sf.write(run_dir / "wav1_capture.wav", wav1_aligned, sample_rate, subtype=wav_info.subtype)
        sf.write(run_dir / "flac_capture.wav", flac_aligned, sample_rate, subtype=wav_info.subtype)
        sf.write(run_dir / "wav2_capture.wav", wav2_aligned, sample_rate, subtype=wav_info.subtype)
        sf.write(run_dir / "test_null.wav", test_null.astype(np.float32), sample_rate, subtype='FLOAT')
        sf.write(run_dir / "control_null.wav", control_null.astype(np.float32), sample_rate, subtype='FLOAT')

        # --- Write metadata ---
        meta_file = run_dir / "metadata.txt"
        with open(meta_file, 'w') as f:
            f.write("FLAC Null Test\n")
            f.write("=" * 50 + "\n")
            f.write(f"WAV source:     {wav_file}\n")
            f.write(f"FLAC source:    {flac_file}\n")
            f.write(f"Sample rate:    {sample_rate} Hz\n")
            f.write(f"Bit depth:      {bit_depth}\n")
            f.write(f"Channels:       {channels}\n")
            f.write(f"Duration:       {duration:.2f}s\n")
            f.write(f"Blocksize:      {blocksize} frames\n")
            f.write(f"Output device:  {output_device.name}\n")
            f.write(f"Input device:   {input_device.name}\n")
            f.write(f"Started:        {run_timestamp}\n")
            f.write("\n")
            f.write("Passes:\n")
            f.write("-" * 50 + "\n")
            labels = ["WAV  (pass 1)", "FLAC (pass 2)", "WAV  (pass 3)"]
            for i, label in enumerate(labels):
                f.write(f"  {label}:  offset={offsets[i]}  subsample={subsample_vals[i]:+.4f}  xruns={xrun_totals[i]}\n")
            f.write("\n")
            f.write("Results:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Signal level:             {sig_rms_db:.4f} dBFS (RMS)  /  {sig_peak_db:.4f} dBFS (peak)\n")
            f.write(f"  Test null (WAV-FLAC):     {test_depth:.4f} dB\n")
            f.write(f"  Control null (WAV-WAV):   {control_depth:.4f} dB\n")
            f.write("\n")
            f.write("Files:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  wav1_capture.wav    Pass 1 capture ({bit_depth}-bit PCM)\n")
            f.write(f"  flac_capture.wav    Pass 2 capture ({bit_depth}-bit PCM)\n")
            f.write(f"  wav2_capture.wav    Pass 3 capture ({bit_depth}-bit PCM)\n")
            f.write(f"  test_null.wav       wav1 - flac (32-bit float)\n")
            f.write(f"  control_null.wav    wav1 - wav2 (32-bit float)\n")
            f.write("\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")

        print(f"\nOutput: {run_dir}/")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted.")
    finally:
        backend.release()


def main():
    parser = argparse.ArgumentParser(
        description="FLAC null test - empirical demonstration of lossless transparency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic null test
    python flac_null_test.py music.wav music.flac -o results/

    # With pre-selected devices and custom blocksize
    python flac_null_test.py music.wav music.flac --output-device 0 --input-device 1 --blocksize 4096
        """
    )

    parser.add_argument('wav_file', help="Source WAV file")
    parser.add_argument('flac_file', help="FLAC file (encoded from same WAV)")
    parser.add_argument('-o', '--output', default='./results',
                        help="Output directory (default: ./results)")
    parser.add_argument('--output-device', type=int, default=None,
                        help="Output device index (skip selection)")
    parser.add_argument('--input-device', type=int, default=None,
                        help="Input device index (skip selection)")
    parser.add_argument('--blocksize', type=int, default=2048,
                        help="Callback buffer size in frames (default: 2048)")
    parser.add_argument('-l', '--list', action='store_true',
                        help="List devices and exit")
    parser.add_argument('--shared', action='store_true',
                        help="Use WASAPI shared mode (Windows only)")
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    args = parser.parse_args()

    if args.list:
        list_devices()
        return 0

    if not os.path.exists(args.wav_file):
        print(f"Error: WAV file not found: {args.wav_file}")
        return 1

    if not os.path.exists(args.flac_file):
        print(f"Error: FLAC file not found: {args.flac_file}")
        return 1

    run_null_test(
        wav_file=args.wav_file,
        flac_file=args.flac_file,
        output_dir=args.output,
        output_device_idx=args.output_device,
        input_device_idx=args.input_device,
        blocksize=args.blocksize,
        shared_mode=args.shared,
    )

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    finally:
        # Catch-all: release any devices that may still be hogged
        try:
            backend = get_backend()
            backend.release()
        except Exception:
            pass
