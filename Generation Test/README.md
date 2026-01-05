# Audio Generation Test

Measures cumulative audio degradation through repeated DAC/ADC conversion cycles. Plays audio through your interface, records it back, and repeats—revealing subtle losses that accumulate over many generations.

## Quick Start

```bash
# Install dependencies
pip install sounddevice numpy scipy soundfile

# Run 100 generations, save every 10
python gen_test.py music.wav -g 100 -m 10 -o results/

# With dynamic level compensation
python gen_test.py music.wav -g 100 -m 10 -c dynamic -o results/
```

## Usage

```
python gen_test.py <source.wav> [options]

Arguments:
  source              Source WAV file (16/24/32-bit PCM)

Options:
  -g, --generations N Number of generations (default: 10)
  -m, --milestone N   Save WAV every N generations (default: 0 = final only)
  -o, --output DIR    Output directory (default: ./results)
  -c, --compensate    Level compensation: 'dynamic' or 'calibrated'
  --shared            WASAPI shared mode (Windows, for virtual devices)
  --output-device N   Skip device selection, use device N for output
  --input-device N    Skip device selection, use device N for input
  -l, --list          List audio devices and exit
```

## Output

Each run creates a timestamped folder containing:

- `gen_000_<source>.wav` - Original source (unmodified)
- `gen_010.wav`, `gen_020.wav`, ... - Milestone generations
- `gen_100.wav` - Final generation
- `metadata.txt` - Run parameters and per-generation measurements

Console output (one line per generation):

```
# No compensation - shows delta, no gain
  1/100  rms:-24.74dB(-0.10)  pk:-6.63  off:12541 sub:+0.23
 10/100  rms:-25.74dB(-1.00)  pk:-6.63  off:12541 sub:-0.25  gen_010.wav

# Dynamic compensation - shows gain, no delta
  1/100  rms:-24.74dB  pk:-6.63  gain:+0.10  off:12541 sub:+0.23
 10/100  rms:-24.74dB  pk:-6.63  gain:+0.10  off:12541 sub:-0.25  gen_010.wav
```

Fields:
- `rms` - RMS level in dBFS (with delta from reference when not using dynamic compensation)
- `pk` - Peak level in dBFS
- `gain` - Applied gain compensation in dB (only shown when compensation enabled)
- `off` - Alignment offset in samples (I/O latency)
- `sub` - Cumulative sub-sample drift

## Platforms

### macOS (CoreAudio)
- Uses hog mode for exclusive device access
- Sets device sample rate and bit depth directly
- Tested with RME ADI-2 Pro

### Windows (ASIO / WASAPI)
- ASIO preferred for pro audio interfaces
- Falls back to WASAPI Exclusive for consumer hardware
- Use `--shared` flag for virtual devices (VB-Cable, etc.)

## How It Works

### Generation Loop

1. **Play + Record**: Source audio plays through DAC while simultaneously recording through ADC
2. **Align**: Cross-correlation finds exact sample offset between played and recorded audio
3. **Extract**: Recorded audio is trimmed to match source length at correct alignment
4. **Measure**: RMS and peak levels compared to original reference
5. **Compensate** (optional): Gain adjustment applied to maintain consistent levels
6. **Repeat**: Recorded audio becomes input for next generation

### Sample Alignment

The recorded audio is delayed by I/O latency (typically 100-150ms). Cross-correlation precisely locates the start of the actual audio within the recording buffer.

Sub-sample drift is tracked across generations using parabolic interpolation on the correlation peak. This catches clock rate mismatches between playback and recording paths.

Padding (0.5s silence) is appended before playback to ensure the full audio is captured despite latency.

### Level Measurement

RMS and peak are measured on the aligned audio before any compensation:
- Integer formats (int16/int32): Normalized by max integer value
- Float formats: Used directly

### Level Compensation

**Dynamic mode** (`-c dynamic`): Each generation is gain-adjusted to match the original RMS. This isolates spectral/distortion changes from cumulative level loss.

**Calibrated mode** (`-c calibrated`): Measures I/O loss once using pink noise, applies fixed correction. Currently experimental.

### Bit Depth Handling

- 16-bit: Native int16
- 24-bit: Packed into int32 (left-justified)
- 32-bit: Native int32 or float32

The tool preserves the source file's bit depth throughout the chain.

## Requirements

- Python 3.10+
- sounddevice
- numpy
- scipy
- soundfile

## Hardware Setup

Connect your DAC output to ADC input with appropriate cables:
- Balanced (XLR or TRS) preferred for lowest noise
- Match levels to avoid clipping or excessive noise floor
- Verify signal path with a quick test run

For interfaces with loopback routing (digital), use physical cables to test actual DAC/ADC conversion.

## Interpreting Results

**RMS drift**: Cumulative level loss reveals converter gain accuracy.

**Spectral changes**: Compare FFTs of gen_000 vs gen_100 to see frequency-dependent losses.

**Distortion**: Examine difference signal (gen_100 minus gen_000 aligned) for added harmonics or noise.

**Sub-sample drift**: Tracked via parabolic interpolation on correlation peak.
