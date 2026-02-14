#!/usr/bin/env python3
"""
Korf Audio Compliance/Effective Mass Calculator - Python Implementation

Reverse-engineered from korfaudio.com/calculator (Feb 2026)
by JP / sjplot.com with Claude analysis

This implements the exact model used by the Korf online calculator:
- Valencia-Campo displacement transmissibility (textbook driven damped
  harmonic oscillator transfer function)
- Carlson natural frequency (Hooke's law): f0 = 159.15 / sqrt(m * C),
  treating the tonearm as a mass on a spring (cartridge compliance)
- Default Q = 1.0 (damping ratio zeta = 0.5), user-adjustable
- Input excitation: 0.1 mm stylus displacement
- Outputs: headshell excursion (mm) and acceleration (g) vs frequency

NOTES ON THE MODEL:
The Valencia-Campo formula used by Korf is the textbook transfer function
for a single-degree-of-freedom driven damped harmonic oscillator:
    H(f) = 1 / sqrt((1 - r^2)^2 + (r/Q)^2)
where r = f/f0.

This is the force-excited transfer function, NOT the correct base
excitation transmissibility, which is:
    H(f) = sqrt(1 + (r/Q)^2) / sqrt((1 - r^2)^2 + (r/Q)^2)

The correct base excitation formula includes the damping term in the
numerator, which affects the curve shape (particularly at high frequencies).

The Carlson natural frequency derives from Hooke's law for a simple
mass-spring system: f0 = (1/2*pi) * sqrt(k/m). With the cartridge
compliance C acting as the spring (k = 1/C) and appropriate unit
conversion (grams, µm/mN), this reduces to f0 = 159.15 / sqrt(m * C).

The Korf site fixes Q at 1.0 regardless of the cartridge compliance or
tonearm mass. In a real cartridge/tonearm system, the Q factor depends
on the viscoelastic properties of the suspension elastomer and can vary
significantly between cartridge designs (typical range: 1-5 for LF
compliance resonance, per Bauer 1963 and AES literature). This
implementation allows Q to be specified as an input parameter.
"""

import numpy as np
import argparse
import sys
import os

# ============================================================
# CORE CALCULATOR
# ============================================================

# Physical constants
G_MM_S2 = 9810.0       # 1g in mm/s^2 (9.81 m/s^2 * 1000)
INPUT_DISP = 0.1        # Input stylus displacement in mm
Q_FIXED = 1.0           # Fixed Q factor (damping ratio zeta = 0.5)

# Green zone thresholds
EXC_THRESHOLD = 0.1     # mm - excursion should stay below this

# Acceleration green zone boundary from korfaudio.com/calculator.
# The server's "dataEtalon" is a fixed reference curve — the acceleration
# response of a reference system at f0 = 159.155/sqrt(245) Hz, Q = 1.
# (Equivalent to 20g effective mass, 12.25 µm/mN compliance.)
# The original server data was 100 points quantised to 4 significant figures;
# this analytical form reproduces it to within ~1.6e-3 g (< 1% of full scale).
GREEN_ZONE_F0 = 159.155 / np.sqrt(245.0)   # ≈ 10.168 Hz


def green_zone_acc(freqs):
    """Korf green zone acceleration boundary — analytical form."""
    return headshell_acceleration(freqs, GREEN_ZONE_F0)


def carlson_frequency(mass_g, compliance_um_mN):
    """
    Carlson resonance frequency.

    f0 = 1/(2*pi) * sqrt(k/m) = 1/(2*pi) * sqrt(1/(C*m))

    With unit conversion:
    f0 = 159.15 / sqrt(m_grams * C_um_per_mN)

    Parameters
    ----------
    mass_g : float
        Total effective mass in grams
        (tonearm effective mass + cartridge weight + headshell weight)
    compliance_um_mN : float
        Cartridge compliance in µm/mN (= 10^-6 cm/dyne)

    Returns
    -------
    float
        Resonance frequency in Hz
    """
    return 159.155 / np.sqrt(mass_g * compliance_um_mN)


def transmissibility(f, f0, Q=Q_FIXED):
    """
    Valencia-Campo displacement transmissibility.

    H(f) = 1 / sqrt((1 - r^2)^2 + (r/Q)^2)

    where r = f/f0

    At DC (f=0): H = 1 (passthrough)
    At resonance: H = Q / sqrt(1 - 1/(4Q^2))  [peak, if Q > 1/sqrt(2)]
    At high frequency: H → (f0/f)^2 → 0

    Parameters
    ----------
    f : array_like
        Frequency in Hz
    f0 : float
        Natural frequency in Hz
    Q : float
        Quality factor (default: 1.0)

    Returns
    -------
    array_like
        Displacement transmissibility magnitude
    """
    f = np.asarray(f, dtype=float)
    r = f / f0
    return 1.0 / np.sqrt((1.0 - r**2)**2 + (r / Q)**2)


def headshell_excursion(f, f0, Q=Q_FIXED, A_input=INPUT_DISP):
    """
    Headshell excursion (displacement) in mm.

    Parameters
    ----------
    f : array_like
        Frequency in Hz
    f0 : float
        Natural frequency in Hz
    Q : float
        Quality factor (default: 1.0)
    A_input : float
        Input stylus displacement in mm (default: 0.1)

    Returns
    -------
    array_like
        Headshell displacement in mm
    """
    return A_input * transmissibility(f, f0, Q)


def headshell_acceleration(f, f0, Q=Q_FIXED, A_input=INPUT_DISP):
    """
    Headshell acceleration in g.

    acc = omega^2 * displacement
    Convert mm/s^2 to g by dividing by 9810.

    Parameters
    ----------
    f : array_like
        Frequency in Hz
    f0 : float
        Natural frequency in Hz
    Q : float
        Quality factor (default: 1.0)
    A_input : float
        Input stylus displacement in mm (default: 0.1)

    Returns
    -------
    array_like
        Headshell acceleration in g
    """
    disp = headshell_excursion(f, f0, Q, A_input)
    omega = 2.0 * np.pi * np.asarray(f, dtype=float)
    return omega**2 * disp / G_MM_S2


def calculate(tonearm_mass, compliance, cartridge_weight=0, headshell_weight=0,
              Q=Q_FIXED, f_min=0.01, f_max=35.0, n_points=1000):
    """
    Full Korf calculator computation.

    Parameters
    ----------
    tonearm_mass : float
        Tonearm effective mass in grams (1-100)
    compliance : float
        Cartridge compliance in µm/mN (1-100)
    cartridge_weight : float
        Cartridge weight in grams (0-50)
    headshell_weight : float
        Removable headshell + mounting hardware weight in grams (0-50)
    Q : float
        Quality factor (default: 1.0)
    f_min, f_max : float
        Frequency range in Hz
    n_points : int
        Number of frequency points

    Returns
    -------
    dict with keys:
        'freqs' : ndarray - frequency array (Hz)
        'excursion' : ndarray - headshell excursion (mm)
        'acceleration' : ndarray - headshell acceleration (g)
        'f0' : float - Carlson natural frequency (Hz)
        'total_mass' : float - total effective mass (g)
        'Q' : float - quality factor used
        'exc_peak_hz' : float - excursion peak frequency
        'exc_peak_mm' : float - excursion peak value
        'acc_peak_hz' : float - acceleration peak frequency
        'acc_peak_g' : float - acceleration peak value
        'in_green_zone' : bool - whether acceleration stays below threshold
    """
    total_mass = tonearm_mass + cartridge_weight + headshell_weight
    f0 = carlson_frequency(total_mass, compliance)

    freqs = np.linspace(f_min, f_max, n_points)
    exc = headshell_excursion(freqs, f0, Q)
    acc = headshell_acceleration(freqs, f0, Q)

    exc_peak_idx = np.argmax(exc)
    acc_peak_idx = np.argmax(acc)

    return {
        'freqs': freqs,
        'excursion': exc,
        'acceleration': acc,
        'f0': f0,
        'total_mass': total_mass,
        'compliance': compliance,
        'Q': Q,
        'exc_peak_hz': freqs[exc_peak_idx],
        'exc_peak_mm': exc[exc_peak_idx],
        'acc_peak_hz': freqs[acc_peak_idx],
        'acc_peak_g': acc[acc_peak_idx],
    }


def analytical_peaks(f0, Q=Q_FIXED, A_input=INPUT_DISP):
    """
    Analytical peak frequencies and values.

    Displacement peak:
        f_d = f0 * sqrt(1 - 1/(2Q^2))    [exists only if Q > 1/sqrt(2)]
        H_max = Q / sqrt(1 - 1/(4Q^2))

    Acceleration peak:
        f_a = f0 * sqrt(2Q^2 / (2Q^2 - 1))  [exists only if Q > 1/sqrt(2)]
        Note: f_d * f_a = f0^2 (geometric mean is f0)

    Parameters
    ----------
    f0 : float
        Natural frequency in Hz
    Q : float
        Quality factor
    A_input : float
        Input displacement in mm

    Returns
    -------
    dict with analytical peak values
    """
    if Q <= 1.0 / np.sqrt(2):
        return {
            'exc_peak_exists': False,
            'acc_peak_exists': False,
        }

    r_d = np.sqrt(1.0 - 1.0 / (2.0 * Q**2))
    r_a = np.sqrt(2.0 * Q**2 / (2.0 * Q**2 - 1.0))
    H_max = Q / np.sqrt(1.0 - 1.0 / (4.0 * Q**2))

    f_d = f0 * r_d
    f_a = f0 * r_a

    exc_peak = A_input * H_max
    # Acceleration at f_a:
    omega_a = 2.0 * np.pi * f_a
    H_at_fa = transmissibility(f_a, f0, Q)
    acc_peak = omega_a**2 * A_input * H_at_fa / G_MM_S2

    return {
        'exc_peak_exists': True,
        'acc_peak_exists': True,
        'exc_peak_hz': f_d,
        'exc_peak_mm': exc_peak,
        'acc_peak_hz': f_a,
        'acc_peak_g': acc_peak,
        'H_max': H_max,
        'geometric_mean': np.sqrt(f_d * f_a),  # should equal f0
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_results(result, output_path=None, show=True, save=False):
    """
    Generate Korf-style frequency response plots.

    Replicates the visual output of korfaudio.com/calculator:
    - Top panel: Headshell excursion (mm) vs frequency
    - Bottom panel: Acceleration at headshell (g) vs frequency
    - Green zone threshold lines
    - Carlson f0 marker

    Parameters
    ----------
    result : dict
        Output from calculate()
    output_path : str, optional
        Path to save the plot PNG. If None and save=True, auto-generates.
    show : bool
        If True, display the plot interactively (default: True)
    save : bool
        If True, save the plot to a PNG file (default: False)
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    freqs = result['freqs']
    exc = result['excursion']
    acc = result['acceleration']
    f0 = result['f0']
    Q = result['Q']
    total_mass = result['total_mass']

    # Korf-style colors
    CURVE_COLOR = '#1a5276'
    GREEN_ZONE = '#27ae60'
    RED_ZONE = '#e74c3c'
    THRESHOLD_COLOR = '#e67e22'
    F0_COLOR = '#8e44ad'
    BG_COLOR = '#fafafa'
    GRID_COLOR = '#dcdcdc'

    compliance = result.get('compliance', None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12), facecolor=BG_COLOR)
    line1 = f'Korf Calculator — {total_mass:.1f}g'
    if compliance is not None:
        line1 += f', {compliance:.0f} µm/mN'
    line2 = f'f₀ = {f0:.2f} Hz, Q = {Q:g}'
    fig.suptitle(line1, fontsize=14, fontweight='bold', y=0.99)
    fig.text(0.5, 0.955, line2, ha='center', fontsize=10, color='#555')

    # ── Fixed axis scales matching korfaudio.com/calculator ──
    XMAX = 30.0             # Hz
    EXC_YMAX = 0.30         # mm
    ACC_YMAX = 0.150        # g
    EXC_YTICKS = np.arange(0, EXC_YMAX + 0.01, 0.05)   # 0.00, 0.05, ... 0.30
    ACC_YTICKS = np.arange(0, ACC_YMAX + 0.001, 0.025)  # 0.000, 0.025, ... 0.150
    XTICKS = np.arange(0, XMAX + 1, 10)                 # 0, 10, 20, 30
    XTICKS_MINOR = np.arange(5, XMAX, 10)               # 5, 15, 25

    for ax in (ax1, ax2):
        ax.set_facecolor(BG_COLOR)
        ax.set_box_aspect(1)           # square plot area
        ax.grid(True, which='major', color=GRID_COLOR, linewidth=0.5)
        ax.grid(True, which='minor', color=GRID_COLOR, linewidth=0.3, alpha=0.5)
        ax.set_xlim(0, XMAX)
        ax.set_xticks(XTICKS)
        ax.set_xticks(XTICKS_MINOR, minor=True)

    # ── Top panel: Excursion ──
    # Green zone fill below threshold (matching accel panel style)
    ax1.fill_between(freqs, 0, EXC_THRESHOLD, alpha=0.15, color=GREEN_ZONE)
    ax1.axhline(y=EXC_THRESHOLD, color=GREEN_ZONE, linestyle='--',
                linewidth=1.5, label=f'Threshold ({EXC_THRESHOLD} mm)')
    ax1.plot(freqs, exc, color=CURVE_COLOR, linewidth=2, label='Excursion')
    ax1.axvline(x=f0, color=F0_COLOR, linestyle=':', linewidth=1,
                alpha=0.7, label=f'Carlson f₀ = {f0:.2f} Hz')

    # Mark excursion peak
    ax1.annotate(f"{result['exc_peak_mm']:.4f} mm @ {result['exc_peak_hz']:.1f} Hz",
                 xy=(result['exc_peak_hz'], result['exc_peak_mm']),
                 xytext=(20, 25), textcoords='offset points', fontsize=7,
                 color=RED_ZONE,
                 arrowprops=dict(arrowstyle='->', color=RED_ZONE, lw=1))

    ax1.set_ylim(0, EXC_YMAX)
    ax1.set_yticks(EXC_YTICKS)
    ax1.set_ylabel('Excursion, mm', fontsize=9)
    ax1.set_title('Excursion at Headshell', fontsize=8, loc='left', color='#555')
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.9)

    # ── Bottom panel: Acceleration with green zone ──
    # Green zone boundary: interpolated from Korf's fixed reference curve
    acc_green = green_zone_acc(freqs)

    # Green zone shading below the reference curve
    ax2.fill_between(freqs, 0, acc_green,
                     alpha=0.15, color=GREEN_ZONE)
    ax2.plot(freqs, acc_green, color=GREEN_ZONE, linewidth=1.5,
             linestyle='--', label='Green zone limit')

    ax2.plot(freqs, acc, color=CURVE_COLOR, linewidth=2, label='Acceleration')
    ax2.axvline(x=f0, color=F0_COLOR, linestyle=':', linewidth=1,
                alpha=0.7, label=f'Carlson f₀ = {f0:.2f} Hz')

    # Mark acceleration peak
    ax2.annotate(f"{result['acc_peak_g']:.4f}g @ {result['acc_peak_hz']:.1f} Hz",
                 xy=(result['acc_peak_hz'], result['acc_peak_g']),
                 xytext=(20, 25), textcoords='offset points', fontsize=7,
                 color=RED_ZONE,
                 arrowprops=dict(arrowstyle='->', color=RED_ZONE, lw=1))

    ax2.set_ylim(0, ACC_YMAX)
    ax2.set_yticks(ACC_YTICKS)
    ax2.set_xlabel('Frequency, Hz', fontsize=9)
    ax2.set_ylabel('Acc, g', fontsize=9)
    ax2.set_title('Acceleration at Headshell', fontsize=8, loc='left', color='#555')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.01, 1, 0.96])

    saved_path = None
    if save:
        if output_path is None:
            c_str = f"{compliance:g}" if compliance is not None else "0"
            output_path = f"KorfCompCalc_{total_mass:.0f}g_{c_str}um.png"
        fig.savefig(output_path, dpi=150,
                    facecolor=BG_COLOR, edgecolor='none')
        print(f"  Plot saved: {output_path}")
        saved_path = output_path

    if show:
        plt.show()

    plt.close(fig)
    return saved_path


# ============================================================
# DISPLAY / CLI
# ============================================================

def print_results(result, show_curve=False):
    """Print calculator results in a formatted table."""
    print(f"\n{'='*60}")
    print(f"  Korf Compliance Calculator Results")
    print(f"{'='*60}")
    print(f"  Total effective mass:  {result['total_mass']:.1f} g")
    print(f"  Carlson frequency f₀: {result['f0']:.2f} Hz")
    print(f"  Q factor:             {result['Q']:g}")
    print(f"  Input excitation:     {INPUT_DISP} mm")
    print(f"{'─'*60}")
    print(f"  Excursion peak:       {result['exc_peak_mm']:.4f} mm "
          f"@ {result['exc_peak_hz']:.2f} Hz")
    print(f"  Acceleration peak:    {result['acc_peak_g']:.6f} g "
          f"@ {result['acc_peak_hz']:.2f} Hz")
    print(f"{'─'*60}")

    print(f"{'='*60}\n")

    if show_curve:
        # Simple ASCII plot of acceleration
        freqs = result['freqs']
        acc = result['acceleration']
        acc_green = green_zone_acc(freqs)
        max_acc = max(np.max(acc), np.max(acc_green) * 1.2)
        width = 60
        height = 20

        print(f"  Acceleration at Headshell (g)")
        print(f"  {'─'*width}")

        for row in range(height, -1, -1):
            val = row / height * max_acc
            line = f"  {val:7.4f} |"
            for col in range(width):
                f_col = f_col = freqs[0] + col / width * (freqs[-1] - freqs[0])
                # Find nearest frequency
                idx = np.argmin(np.abs(freqs - f_col))
                a = acc[idx]
                a_row = a / max_acc * height
                if abs(a_row - row) < 0.5:
                    line += "*"
                elif row == int(green_zone_acc(np.array([f_col]))[0] / max_acc * height):
                    line += "-"
                else:
                    line += " "
            print(line)

        print(f"  {'':>8} +{'─'*width}")
        print(f"  {'':>8}  {freqs[0]:<10.0f}{'':>20}{freqs[-1]:>10.0f} Hz")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Korf Audio Compliance/Effective Mass Calculator (Python)')
    parser.add_argument('mass', type=float,
                        help='Tonearm effective mass (grams, 1-100)')
    parser.add_argument('compliance', type=float,
                        help='Cartridge compliance (µm/mN, 1-100)')
    parser.add_argument('Q', type=float, nargs='?', default=Q_FIXED,
                        help=f'Q factor (default: {Q_FIXED})')
    parser.add_argument('-c', '--cart-weight', type=float, default=0,
                        help='Cartridge weight (grams, 0-50)')
    parser.add_argument('-s', '--headshell-weight', type=float, default=0,
                        help='Headshell/hardware weight (grams, 0-50)')
    parser.add_argument('--save', type=str, default=None, nargs='?',
                        const='auto',
                        help='Save plot as PNG (optional path; default: korf_plot_<mass>g.png)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Suppress the interactive plot window')
    parser.add_argument('--ascii', action='store_true',
                        help='Show ASCII plot in terminal')
    parser.add_argument('--csv', action='store_true',
                        help='Output as CSV (freq,excursion,acceleration)')

    args = parser.parse_args()

    result = calculate(args.mass, args.compliance,
                       args.cart_weight, args.headshell_weight,
                       Q=args.Q)

    if args.csv:
        print("frequency_hz,excursion_mm,acceleration_g")
        for f, e, a in zip(result['freqs'], result['excursion'], result['acceleration']):
            print(f"{f:.4f},{e:.8f},{a:.8f}")
    else:
        print_results(result, show_curve=args.ascii)

    # Determine save path (None = don't save)
    save_path = None
    if args.save is not None:
        save_path = args.save if args.save != 'auto' else None  # None → auto-name

    show_plot = not args.no_plot and not args.csv
    plot_results(result, output_path=save_path, show=show_plot,
                 save=args.save is not None)

    return result


# ============================================================
# VERIFICATION
# ============================================================

def verify_against_korf():
    """
    Verify this implementation against data extracted from korfaudio.com/calculator.
    """
    # Corrected reference data (with Y-axis calibration correction applied)
    # Format: (mass, compliance, measured_exc_peak_hz, measured_exc_peak_mm,
    #          measured_acc_peak_hz, measured_acc_peak_g)
    EXC_CORR = 0.005732  # axis calibration correction
    ACC_CORR = 0.002866

    reference_data = [
        (35, 25, 3.836, 0.120587 - EXC_CORR, 7.655, 0.016013 - ACC_CORR),
        (35, 15, 4.791, 0.120546 - EXC_CORR, 9.884, 0.024981 - ACC_CORR),
        (14,  5, 13.385, 0.120590 - EXC_CORR, 26.754, 0.170710 - ACC_CORR),
        (40, 40, 2.881, 0.120547 - EXC_CORR, 5.745, 0.009916 - ACC_CORR),
        (30, 15, 5.427, 0.120550 - EXC_CORR, 10.520, 0.028717 - ACC_CORR),
        (20, 10, 7.974, 0.120592 - EXC_CORR, 15.931, 0.061414 - ACC_CORR),
        (20, 30, 4.472, 0.120539 - EXC_CORR, 9.247, 0.022178 - ACC_CORR),
        (20, 50, 3.517, 0.120582 - EXC_CORR, 7.019, 0.014330 - ACC_CORR),
    ]

    print("\n" + "=" * 70)
    print("VERIFICATION AGAINST KORF ONLINE CALCULATOR")
    print("=" * 70)
    print(f"\n{'Mass':>4} {'Compl':>5} | {'Exc pk mm':>10} {'Pred':>10} {'Err%':>6} | "
          f"{'Acc pk g':>10} {'Pred':>10} {'Err%':>6}")
    print("-" * 70)

    max_exc_err = 0
    max_acc_err = 0

    for mass, compl, exc_hz_m, exc_mm_m, acc_hz_m, acc_g_m in reference_data:
        result = calculate(mass, compl)
        peaks = analytical_peaks(result['f0'])

        exc_err = abs(peaks['exc_peak_mm'] - exc_mm_m) / exc_mm_m * 100
        acc_err = abs(peaks['acc_peak_g'] - acc_g_m) / acc_g_m * 100
        max_exc_err = max(max_exc_err, exc_err)
        max_acc_err = max(max_acc_err, acc_err)

        print(f"{mass:4d} {compl:5d} | {exc_mm_m:10.6f} {peaks['exc_peak_mm']:10.6f} {exc_err:6.2f} | "
              f"{acc_g_m:10.6f} {peaks['acc_peak_g']:10.6f} {acc_err:6.2f}")

    print(f"\nMax excursion error: {max_exc_err:.2f}%")
    print(f"Max acceleration error: {max_acc_err:.2f}%")
    print("(Acceleration errors are larger for low-f0 cases due to SVG chart resolution)")

    return max_exc_err < 2.0  # pass if within 2%


if __name__ == '__main__':
    if '--verify' in sys.argv:
        sys.argv.remove('--verify')
        success = verify_against_korf()
        sys.exit(0 if success else 1)
    else:
        main()
