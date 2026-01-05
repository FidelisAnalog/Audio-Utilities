#!/usr/bin/env python3
"""
Audio generation test - measure compound DAC/ADC degradation.

Plays audio through DAC, records through ADC, and repeats for N generations.
Saves milestone files at specified intervals.

Usage:
    python gen_test.py source.wav --generations 100 --milestone 10 --output results/
"""

__version__ = "0.1.0"

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import soundfile as sf
import numpy as np

from backends import get_backend, StreamConfig


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
    # Standard formula: δ = (y_plus - y_minus) / (2 * (2*y_peak - y_minus - y_plus))
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
    
    # When cumulative exceeds ±0.5, adjust integer offset
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
    """
    Extract aligned audio from recording starting at offset.
    """
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


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """
    Apply linear gain to audio, preserving dtype and clipping if needed.
    """
    if audio.dtype in (np.int16, np.int32):
        max_val = np.iinfo(audio.dtype).max
        min_val = np.iinfo(audio.dtype).min
        scaled = audio.astype(np.float64) * gain
        clipped = np.clip(scaled, min_val, max_val)
        return clipped.astype(audio.dtype)
    else:
        return audio * gain


def calibrate_io_gain(backend, config, sample_rate: int, channels: int, dtype) -> float:
    """
    Measure I/O gain by playing pink noise and comparing levels.
    
    Uses bandlimited (20Hz-20kHz) pink noise at -20dBFS for 2 seconds.
    Cross-correlates to align, then compares RMS.
    
    Returns:
        Linear gain factor to compensate for I/O loss
    """
    from scipy import signal as sig
    
    duration = 2.0  # seconds
    num_samples = int(sample_rate * duration)
    
    # Generate pink noise using the Voss-McCartney algorithm
    num_octaves = 16
    pink = np.zeros(num_samples)
    
    for i in range(num_octaves):
        step = 2 ** i
        noise = np.random.randn((num_samples // step) + 1)
        pink += np.repeat(noise, step)[:num_samples]
    
    # Bandlimit to 20Hz-20kHz
    nyquist = sample_rate / 2
    low = 20 / nyquist
    high = min(20000 / nyquist, 0.99)
    sos = sig.butter(4, [low, high], btype='band', output='sos')
    pink = sig.sosfilt(sos, pink)
    
    # Normalize to -20dBFS RMS
    pink = pink / np.sqrt(np.mean(pink ** 2))
    amplitude = 10 ** (-20 / 20)  # -20dBFS
    pink = pink * amplitude
    
    # Convert to target dtype
    if dtype in (np.int16, np.int32):
        max_val = np.iinfo(dtype).max
        test_signal = (pink * max_val).astype(dtype)
    else:
        test_signal = pink.astype(dtype)
    
    # Make stereo if needed
    if channels > 1:
        test_signal = np.column_stack([test_signal] * channels)
    else:
        test_signal = test_signal.reshape(-1, 1)
    
    # Add padding for latency (like generation loop does)
    pad_samples = sample_rate // 2
    padding = np.zeros((pad_samples, channels), dtype=dtype)
    padded_signal = np.vstack([test_signal, padding])
    
    # Play and record
    recorded = backend.playrec(padded_signal, config)
    
    # Use cross-correlation to find alignment (like generation loop)
    offset, _ = find_alignment_offset(test_signal, recorded, sample_rate, 0.0)
    aligned = extract_aligned_audio(recorded, offset, num_samples)
    
    # Measure RMS of both
    ref_rms, _ = measure_levels(test_signal)
    rec_rms, _ = measure_levels(aligned)
    
    if rec_rms > 0:
        gain = ref_rms / rec_rms
        gain_db = 20 * np.log10(gain)
        print(f"  Calibration: ref={20*np.log10(ref_rms):.4f}dBFS, rec={20*np.log10(rec_rms):.4f}dBFS, correction={gain_db:+.4f}dB")
        return gain
    else:
        print(f"  Calibration failed: no signal recorded")
        return 1.0


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


def run_generation_test(
    source_file: str,
    output_dir: str,
    generations: int,
    milestone_interval: int,
    output_device_idx: int = None,
    input_device_idx: int = None,
    compensate: str = None,
    shared_mode: bool = False,
):
    """
    Run the generation test.
    
    Args:
        source_file: Path to source WAV file
        output_dir: Directory for output files
        generations: Number of generations to run
        milestone_interval: Save a file every N generations (0 = no milestones)
        output_device_idx: Pre-selected output device index
        input_device_idx: Pre-selected input device index
        compensate: Compensation mode - None, 'dynamic' (per-iteration RMS), or 'calibrated' (fixed I/O gain)
        shared_mode: Use WASAPI shared mode instead of exclusive (Windows only, for virtual devices)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get file info first to determine native format
    print(f"\nReading source: {source_file}")
    info = sf.info(source_file)
    
    # Parse bit depth from subtype and determine read dtype
    subtype = info.subtype
    if 'PCM_16' in subtype:
        bit_depth = 16
        read_dtype = 'int16'
    elif 'PCM_24' in subtype:
        bit_depth = 24
        read_dtype = 'int32'  # 24-bit gets packed into int32
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
    
    # Measure reference levels
    ref_rms, ref_peak = measure_levels(data)
    ref_rms_db = 20 * np.log10(ref_rms) if ref_rms > 0 else -np.inf
    ref_peak_db = 20 * np.log10(ref_peak) if ref_peak > 0 else -np.inf
    print(f"  Reference RMS: {ref_rms_db:.4f} dBFS")
    print(f"  Reference Peak: {ref_peak_db:.4f} dBFS")
    if compensate:
        print(f"  Level compensation: {compensate.upper()}")
    
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
    
    # Create run metadata
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"gen_{Path(source_file).stem}_{run_timestamp}"
    run_dir = output_path / run_name
    run_dir.mkdir(exist_ok=True)
    
    # Save metadata
    meta_file = run_dir / "metadata.txt"
    with open(meta_file, 'w') as f:
        f.write(f"Source: {source_file}\n")
        f.write(f"Sample rate: {sample_rate}\n")
        f.write(f"Bit depth: {bit_depth}\n")
        f.write(f"Channels: {channels}\n")
        f.write(f"Duration: {duration:.2f}s\n")
        f.write(f"Generations: {generations}\n")
        f.write(f"Milestone interval: {milestone_interval}\n")
        f.write(f"Level compensation: {compensate}\n")
        f.write(f"Reference RMS: {ref_rms_db:.4f} dBFS\n")
        f.write(f"Reference Peak: {ref_peak_db:.4f} dBFS\n")
        f.write(f"Output device: {output_device.name}\n")
        f.write(f"Input device: {input_device.name}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
    
    # Copy source to run directory
    source_copy = run_dir / f"gen_000_{Path(source_file).stem}.wav"
    sf.write(source_copy, data, sample_rate, subtype=info.subtype)
    print(f"\nSource saved: {source_copy}")
    
    # Run generations
    backend = get_backend()
    backend.shared_mode = shared_mode  # For WASAPI: use shared mode instead of exclusive
    current_data = data.copy()
    
    try:
        print(f"\nConfiguring devices...")
        backend.configure_output(output_device, config)
        backend.configure_input(input_device, config)
        
        # Generous padding to ensure we capture full audio (0.5s should cover any latency)
        pad_samples = sample_rate // 2
        padding = np.zeros((pad_samples, channels), dtype=data.dtype)
        
        # Calibrate I/O gain if using calibrated mode
        io_gain = 1.0
        if compensate == 'calibrated':
            print(f"\nCalibrating I/O gain...")
            io_gain = calibrate_io_gain(backend, config, int(sample_rate), channels, data.dtype)
            with open(meta_file, 'a') as f:
                f.write(f"Calibrated I/O gain: {20*np.log10(io_gain):+.4f} dB\n")
        
        estimated_total = generations * duration
        print(f"\nStarting {generations} generations (estimated time: {estimated_total/60:.1f} minutes)")
        print("-" * 50)
        
        # Log generation header to metadata
        with open(meta_file, 'a') as f:
            f.write(f"\nStarting {generations} generations\n")
            f.write("-" * 50 + "\n")
        
        target_len = len(data)
        cumulative_subsample = 0.0
        
        for gen in range(1, generations + 1):
            # Pad current data with silence at end
            padded_data = np.vstack([current_data, padding])
            
            # Play padded audio and record
            recorded = backend.playrec(padded_data, config)
            
            # Find exact alignment using cross-correlation with sub-sample tracking
            offset, cumulative_subsample = find_alignment_offset(
                current_data, recorded, int(sample_rate), cumulative_subsample
            )
            
            # Extract aligned audio
            aligned = extract_aligned_audio(recorded, offset, target_len)
            
            # Measure levels
            gen_rms, gen_peak = measure_levels(aligned)
            gen_rms_db = 20 * np.log10(gen_rms) if gen_rms > 0 else -np.inf
            gen_peak_db = 20 * np.log10(gen_peak) if gen_peak > 0 else -np.inf
            rms_delta = gen_rms_db - ref_rms_db
            
            # Apply compensation if enabled
            gain_db = 0.0
            if compensate == 'dynamic' and gen_rms > 0:
                # Dynamic: adjust to match original RMS each iteration
                gain = ref_rms / gen_rms
                aligned = apply_gain(aligned, gain)
                gain_db = 20*np.log10(gain)
            elif compensate == 'calibrated':
                # Calibrated: apply fixed I/O gain correction
                aligned = apply_gain(aligned, io_gain)
                gain_db = 20*np.log10(io_gain)
            
            # Check if this is a milestone
            is_milestone = milestone_interval > 0 and gen % milestone_interval == 0
            is_final = gen == generations
            
            # Build console output (compact, <=80 chars)
            if compensate == 'dynamic':
                # In dynamic mode, delta is redundant with gain
                rms_str = f"rms:{gen_rms_db:.2f}dB"
                gain_str = f"  gain:{gain_db:+.2f}"
            elif compensate == 'calibrated':
                # Calibrated mode shows both
                rms_str = f"rms:{gen_rms_db:.2f}dB({rms_delta:+.2f})"
                gain_str = f"  gain:{gain_db:+.2f}"
            else:
                # No compensation: show delta, no gain
                rms_str = f"rms:{gen_rms_db:.2f}dB({rms_delta:+.2f})"
                gain_str = ""
            
            console_line = f"{gen:3d}/{generations}  {rms_str}  pk:{gen_peak_db:.2f}{gain_str}  off:{offset} sub:{cumulative_subsample:+.2f}"
            
            if is_milestone or is_final:
                filename = f"gen_{gen:03d}.wav"
                filepath = run_dir / filename
                sf.write(filepath, aligned, sample_rate, subtype=info.subtype)
                console_line += f"  {filename}"
            
            print(console_line)
            
            # Log to metadata file (full precision)
            if compensate:
                meta_line = f"{gen:3d}/{generations}  rms:{gen_rms_db:.4f}dB({rms_delta:+.4f})  pk:{gen_peak_db:.4f}  gain:{gain_db:+.4f}  off:{offset} sub:{cumulative_subsample:+.4f}"
            else:
                meta_line = f"{gen:3d}/{generations}  rms:{gen_rms_db:.4f}dB({rms_delta:+.4f})  pk:{gen_peak_db:.4f}  off:{offset} sub:{cumulative_subsample:+.4f}"
            if is_milestone or is_final:
                meta_line += f"  {filename}"
            with open(meta_file, 'a') as f:
                f.write(meta_line + "\n")
            
            # Use aligned data as input for next generation
            current_data = aligned
        
        print("-" * 50)
        print(f"\nTest complete! Results saved to: {run_dir}")
        
        # Update metadata with completion
        with open(meta_file, 'a') as f:
            f.write("-" * 50 + "\n")
            f.write(f"Completed: {datetime.now().isoformat()}\n")
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted at generation {gen}")
        # Save current state
        interrupt_file = run_dir / f"gen_{gen:03d}_interrupted.wav"
        sf.write(interrupt_file, current_data, sample_rate, subtype=info.subtype)
        print(f"Current state saved: {interrupt_file}")
        
        with open(meta_file, 'a') as f:
            f.write("-" * 50 + "\n")
            f.write(f"Interrupted at generation: {gen}\n")
            f.write(f"Interrupted: {datetime.now().isoformat()}\n")
    
    finally:
        backend.release()
    
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Audio generation test for DAC/ADC degradation measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 100 generations, save every 10 (no compensation)
    python gen_test.py sweep.wav -g 100 -m 10 -o results/

    # Run with dynamic level compensation (per-iteration RMS matching)
    python gen_test.py sweep.wav -g 100 -m 10 -c dynamic -o results/
    
    # Run with calibrated level compensation (fixed I/O gain correction)
    python gen_test.py sweep.wav -g 100 -m 10 -c calibrated -o results/
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
                        help="Level compensation mode: 'dynamic' (per-iteration RMS) or 'calibrated' (fixed I/O gain)")
    parser.add_argument('--shared', action='store_true',
                        help="Use WASAPI shared mode instead of exclusive (Windows only, for virtual devices)")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    if args.list:
        list_devices()
        return 0
    
    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return 1
    
    run_generation_test(
        source_file=args.source,
        output_dir=args.output,
        generations=args.generations,
        milestone_interval=args.milestone,
        output_device_idx=args.output_device,
        input_device_idx=args.input_device,
        compensate=args.compensate,
        shared_mode=args.shared,
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
