# Audio Generation Test

Measures cumulative audio degradation through repeated DAC/ADC conversion cycles. Plays audio through your interface, records it back, aligns, and repeats.

Two versions are included:

- **`gen_test.py`** (v0.2.0) — Music-based alignment using FFT cross-correlation + MSE refinement. Works well for short runs but alignment degrades over many generations as the audio itself degrades.
- **`gen_test_v5.py`** (v0.6.0) — Chirp marker alignment. Prepends known alignment tones to each playback. Alignment quality is constant regardless of generation count. Recommended for 100+ generation runs.

## Quick Start

```bash
# Install dependencies
pip install sounddevice numpy scipy soundfile

# Run 100 generations, save every generation (v5 / chirp alignment)
python gen_test_v5.py music.wav -g 100 -m 1 -o results/

# With dynamic level compensation (per-channel)
python gen_test_v5.py music.wav -g 100 -m 1 -c dynamic -p -o results/

# Original music-based alignment (v1)
python gen_test.py music.wav -g 100 -m 10 -o results/
```

## Usage

### gen_test_v5.py (chirp marker alignment)

```
python gen_test_v5.py <source.wav> [options]

Arguments:
  source              Source WAV file (16/24/32-bit PCM)

Options:
  -g, --generations N Number of generations (default: 10)
  -m, --milestone N   Save WAV every N generations (default: 0 = final only)
  -o, --output DIR    Output directory (default: ./results)
  -c, --compensate    Level compensation: 'dynamic' or 'calibrated'
  -p, --per-channel   Apply level compensation per channel independently
  --shared            WASAPI shared mode (Windows, for virtual devices)
  --output-device N   Skip device selection, use device N for output
  --input-device N    Skip device selection, use device N for input
  --output-channels   Output channel mapping, e.g. '11,12'
  --input-channels    Input channel mapping, e.g. '11,12'
  -l, --list          List audio devices and exit
  -V, --version       Show version and exit
```

### gen_test.py (music-based alignment)

Same options as above, plus:

```
  -a, --align-ref     Alignment reference: 'original' (default) or 'previous'
  -w, --align-window  Alignment search ±samples after gen 1 (default: 64, 0 = full)
```

## Output

Each run creates a timestamped folder containing:

- `gen_000_<source>.wav` — Original source (unmodified)
- `gen_001.wav`, `gen_002.wav`, ... — Milestone generations
- `gen_100.wav` — Final generation
- `metadata.txt` — Run parameters and per-generation measurements

### Console Output (v5)

```
  1/100  rms:-16.57dB  pk:0.00  gain:-0.41/-0.40  off:109433-0.201 sprd:0.00 mse:73477.4  gen_001.wav
  2/100  rms:-16.57dB  pk:0.00  gain:-0.41/-0.40  off:109433-0.201 sprd:0.00 mse:166372.2  gen_002.wav
```

Fields:
- `rms` — RMS level in dBFS (with delta from reference when not using dynamic compensation)
- `pk` — Peak level in dBFS
- `gain` — Applied gain compensation in dB (shown when compensation enabled; `L/R` format with `-p`)
- `off` — Music start offset in samples, including fractional sub-sample correction
- `sprd` — Chirp spread: max disagreement between the three chirp offset estimates (0.00 = perfect agreement)
- `mse` — Mean squared error vs. original source (degradation tracking, not alignment quality)

## Chirp Marker Alignment (v5)

### The Problem

Earlier versions aligned each generation by cross-correlating the recorded audio against the original source (or previous generation). This works when the audio is clean, but as the signal degrades over many DAC/ADC passes, different parts of the waveform suggest different alignments. By generation 30-40, alignment error of 2-3 samples was unavoidable, and a single bad alignment corrupts all subsequent generations.

### The Solution

Instead of aligning on degraded music content, v5 prepends a known alignment marker to each playback. The marker is generated mathematically each generation — it never passes through the DAC/ADC chain and never degrades. Alignment is performed by cross-correlating the recorded marker against the original clean template.

### Marker Structure

```
[200ms silence] [chirp] [200ms gap] [chirp] [200ms gap] [chirp] [1s silence] [music...]

Total marker: ~2.0 seconds
```

Each chirp is a 200ms linear frequency sweep from 800 Hz to 1200 Hz at -6 dBFS, with 5ms fade-in/out to avoid clicks. Three identical chirps provide three independent alignment estimates. The 1-second tail silence ensures a clean separation between the marker and the music content.

### How Alignment Works

1. **Generate marker**: Three identical 800→1200 Hz chirps at known positions within a ~2s marker signal.

2. **Prepend to music**: Each generation plays `[fresh marker] + [current music] + [padding]`. The marker is always clean — generated from the mathematical formula, not carried forward from the previous generation.

3. **Record**: The recording captures `[silence/latency] [marker] [music] [...]`.

4. **Cross-correlate**: The single chirp template is FFT cross-correlated against the recording. Each chirp produces a correlation peak at its position in the recording. The search region is tightly constrained to the marker area to prevent false matches with music content (which often has energy in the 800-1200 Hz range).

5. **Three estimates**: Each chirp's peak gives an independent estimate of where the marker starts. Since the chirp positions within the marker are exact (mathematically defined), the three estimates should agree. The spread between them (`sprd:` in the output) serves as a confidence metric — 0.00 means all three agree perfectly.

6. **Sub-sample refinement**: Parabolic interpolation on each correlation peak provides sub-sample accuracy. The fractional correction is applied via FFT phase rotation on the extracted audio.

7. **Extract music**: The music starts at `marker_start + marker_length`. The aligned music is extracted, the marker is stripped, and only the music is saved to milestone files.

### Why This Works

The key insight is that alignment quality is decoupled from signal degradation. In music-based alignment, the reference degrades along with the signal — you're correlating two noisy signals. With chirp markers:

- The played marker is always pristine (generated fresh each generation)
- The recorded marker has only passed through one DAC/ADC cycle
- Cross-correlation against the clean template always produces a sharp, unambiguous peak
- Alignment quality at generation 100 is identical to generation 1

### Search Region Constraints

The chirp frequency range (800-1200 Hz) overlaps with common musical content, particularly piano. To prevent false correlation peaks from music:

- The search region is limited to the marker area of the recording (never extends into music)
- After generation 1 establishes the hardware latency, subsequent generations search a narrow window around the known position
- Chirps 2 and 3 search ±100 samples around their expected position (the spacing is exact within the marker)
- If the three chirp estimates disagree by more than 5 samples, the outlier is rejected and the closest pair is averaged
- The 1-second tail silence provides a buffer zone between the last chirp and the music

## Music-Based Alignment (v1)

The original alignment approach, retained in `gen_test.py`:

1. **FFT cross-correlation** finds the rough sample offset between played and recorded audio
2. **MSE refinement** over ±search_radius samples around the rough offset selects the best integer alignment
3. **Sub-sample drift** is tracked using parabolic interpolation on the MSE minimum
4. After generation 1 establishes the offset, subsequent generations search a narrow window (default ±64 samples) to prevent false peaks

This works well for runs under ~20 generations. Beyond that, cumulative degradation makes alignment progressively less reliable.

## Level Compensation

**Dynamic mode** (`-c dynamic`): Each generation is gain-adjusted to match the original RMS level. This isolates spectral/distortion changes from cumulative level loss. Use `-p` for per-channel compensation when hardware gains are unlinked between channels.

**Calibrated mode** (`-c calibrated`): Measures I/O loss once using pink noise, applies a fixed correction each generation. Experimental.

## Platforms

### macOS (CoreAudio)
- Uses hog mode for exclusive device access
- Sets device sample rate and bit depth directly
- Tested with RME ADI-2 Pro

### Windows (ASIO / WASAPI)
- ASIO preferred for pro audio interfaces
- Falls back to WASAPI Exclusive for consumer hardware
- Use `--shared` flag for virtual devices (VB-Cable, etc.)

## Hardware Setup

Connect your DAC output to ADC input with appropriate cables:
- Match levels to avoid clipping or excessive noise floor
- Verify signal path with a quick test run

For interfaces with loopback routing (digital), use physical cables to test actual DAC/ADC conversion.

## Requirements

- Python 3.10+
- sounddevice
- numpy
- scipy
- soundfile
