# Korf Audio Compliance/Effective Mass Calculator

Python implementation of the cartridge compliance calculator from [korfaudio.com/calculator](https://korfaudio.com/calculator), reverse-engineered by JP / sjplot.com with Claude analysis (Feb 2026).

## What it does

Computes and plots the frequency response of a phono cartridge/tonearm system: headshell excursion (mm) and acceleration (g) as a function of frequency, given the tonearm effective mass and cartridge compliance. A green zone reference curve shows the Korf-recommended acceleration boundary.

## Physics

The model treats the tonearm + cartridge as a single-degree-of-freedom damped harmonic oscillator.

**Carlson natural frequency** — derived from Hooke's law for a mass on a spring, where the cartridge compliance acts as the spring constant:

    f₀ = 159.15 / √(m × C)

where m is total effective mass (grams) and C is compliance (µm/mN).

**Valencia-Campo transfer function** — the textbook driven damped harmonic oscillator displacement transmissibility:

    H(f) = 1 / √((1 − r²)² + (r/Q)²)

where r = f/f₀ and Q is the quality factor (default 1.0, user-adjustable).

Headshell excursion is the input displacement (0.1 mm) multiplied by H(f). Acceleration is derived from excursion via ω²×displacement, converted to g.

## Example output

![Sample plot — 25g effective mass, 13 µm/mN compliance](examples/KorfCompCalc_25g_13um.png)

## Usage

Basic usage — shows an interactive plot:

    python3 korf_calculator.py 35 15

With custom Q factor:

    python3 korf_calculator.py 20 5 0.75

Save plot to PNG:

    python3 korf_calculator.py 35 15 --save

Save to a specific file:

    python3 korf_calculator.py 35 15 --save myplot.png

Text output only (no plot window):

    python3 korf_calculator.py 35 15 --no-plot

CSV output:

    python3 korf_calculator.py 35 15 --csv

### Arguments

| Argument | Description |
|---|---|
| `mass` | Tonearm effective mass in grams |
| `compliance` | Cartridge compliance in µm/mN |
| `Q` | Quality factor (optional, default 1.0) |
| `-c` | Cartridge body weight in grams |
| `-s` | Headshell/hardware weight in grams |
| `--save [path]` | Save plot as PNG |
| `--no-plot` | Suppress interactive plot window |
| `--csv` | Output frequency/excursion/acceleration as CSV |
| `--ascii` | Show ASCII acceleration plot in terminal |

### Saved filenames

Auto-generated filenames follow the pattern `KorfCompCalc_<mass>g_<compliance>um.png`, for example `KorfCompCalc_35g_15um.png` or `KorfCompCalc_20g_12.5um.png`.

## API usage

The calculator can also be used as a Python module:

```python
from korf_calculator import calculate, plot_results

result = calculate(tonearm_mass=35, compliance=15, Q=0.8)
print(f"f₀ = {result['f0']:.2f} Hz")
print(f"Peak excursion: {result['exc_peak_mm']:.4f} mm @ {result['exc_peak_hz']:.1f} Hz")
print(f"Peak acceleration: {result['acc_peak_g']:.4f} g @ {result['acc_peak_hz']:.1f} Hz")

plot_results(result, save=True)
```

## Green zone

The acceleration plot includes a green zone reference curve from the Korf server's "dataEtalon" — the acceleration response of a reference system (20g effective mass, 12.25 µm/mN compliance, Q = 1). Systems whose acceleration curve stays below this reference are considered well-matched.

## Notes on the model

The Valencia-Campo formula used here is the force-excited oscillator transfer function. The physically correct formula for a record groove (base excitation) includes an additional damping term in the numerator:

    H(f) = √(1 + (r/Q)²) / √((1 − r²)² + (r/Q)²)

This affects the curve shape at high frequencies. The Korf calculator uses the simpler force-excitation form.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
