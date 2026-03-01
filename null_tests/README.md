# FLAC Null Test
0.1.0

## Overview
Empirical demonstration that FLAC decodes bit-identically to WAV through a real DAC/ADC signal chain. Plays both a WAV and a FLAC file (encoded from the same WAV) through your hardware, captures the loopback, and nulls the two captures. A control pass (WAV played twice) establishes the analog noise floor baseline.

All three passes use callback-based streaming playback via `sd.Stream`, simulating real-world progressive decode behavior rather than batch-decoding to memory first.

## Quick Start

```bash
# Install dependencies
pip install sounddevice numpy scipy soundfile

# Encode a FLAC from your WAV source
ffmpeg -i source.wav -c:a flac -compression_level 8 source.flac

# Run the null test
python flac_null_test.py source.wav source.flac -o results/
```

## Usage

```
python flac_null_test.py <source.wav> <source.flac> [options]

Arguments:
  wav_file            Source WAV file (16/24/32-bit PCM)
  flac_file           FLAC file (encoded from same WAV)

Options:
  -o, --output DIR    Output directory (default: ./results)
  --output-device N   Skip device selection, use device N for output
  --input-device N    Skip device selection, use device N for input
  --blocksize N       Callback buffer size in frames (default: 2048)
  --shared            WASAPI shared mode (Windows, for virtual devices)
  -l, --list          List audio devices and exit
  -v, --version       Show version and exit
```

## Output

Each run creates a timestamped folder containing:

- `wav1_capture.wav` - Pass 1 capture (WAV source, native bit depth)
- `flac_capture.wav` - Pass 2 capture (FLAC source, native bit depth)
- `wav2_capture.wav` - Pass 3 capture (WAV source, control)
- `test_null.wav` - wav1 minus flac (32-bit float, natural level)
- `control_null.wav` - wav1 minus wav2 (32-bit float, natural level)
- `metadata.txt` - Configuration, per-pass alignment data, and null depths

Console output:

```
Pass 1/3: Playing WAV... done (offset: +1247, subsample: +0.12)
Pass 2/3: Playing FLAC... done (offset: +1248, subsample: +0.09)
Pass 3/3: Playing WAV... done (offset: +1247, subsample: +0.15)
--------------------------------------------------

Results:
  Signal level:             -6.02 dBFS (RMS)  /  -0.10 dBFS (peak)
  Test null (WAV-FLAC):     -118.3 dB
  Control null (WAV-WAV):   -117.9 dB
```

## Interpreting Results

The test null (WAV-FLAC) and control null (WAV-WAV) should be statistically indistinguishable -- both sitting at the analog noise floor of your hardware. If FLAC playback introduced anything (noise from CPU load, decoding artifacts, etc.), the test null would be measurably shallower than the control. It won't be.

With a high-quality interface (e.g., RME ADI-2 Pro), expect null depths around -115 to -120 dB.

## How It Works

1. **Validate**: Confirms WAV and FLAC have matching sample rate, bit depth, and channel count
2. **Configure**: Sets up audio interface via platform backend (CoreAudio hog mode / WASAPI exclusive)
3. **Play + Record** (3 passes): Each pass streams the file through `sd.Stream` with a callback that reads chunks on-demand from `sf.SoundFile`. For FLAC, this triggers real-time frame decoding via libsndfile. Recorded input is accumulated in the callback.
4. **Align**: Cross-correlation with sub-sample interpolation finds the exact offset between the source reference and each capture
5. **Null**: Captures are normalized to float64 and subtracted
6. **Measure**: Null depth reported as RMS of the residual relative to signal level

## Platforms

### macOS (CoreAudio)
- Uses hog mode for exclusive device access
- Sets device sample rate and bit depth directly

### Windows (ASIO / WASAPI)
- ASIO preferred for pro audio interfaces
- Falls back to WASAPI Exclusive for consumer hardware
- Use `--shared` flag for virtual devices

## Requirements

- Python 3.10+
- sounddevice
- numpy
- scipy
- soundfile

## Hardware Setup

Connect DAC output to ADC input. For interfaces with both (e.g., RME ADI-2 Pro), use analog cables between the output and input to ensure the full DAC/ADC conversion path is exercised.
