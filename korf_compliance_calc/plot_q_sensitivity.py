#!/usr/bin/env python3
"""
Q Sensitivity Plot

Shows how Q factor affects excursion and acceleration curves for a given
arm/cartridge system. Draws a smooth trace through the peaks computed at
fine Q increments to visualise how peak frequency and amplitude converge
toward f₀ as Q increases.

Usage:
    python plot_q_sensitivity.py                  # defaults: 17g, 26 µm/mN
    python plot_q_sensitivity.py 25 15            # custom mass & compliance
    python plot_q_sensitivity.py 25 15 -o out.png # custom output path
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from korf_calculator import transmissibility, carlson_frequency, INPUT_DISP, G_MM_S2


def plot_q_sensitivity(mass, compliance, output_path=None):
    f0 = carlson_frequency(mass, compliance)
    freqs = np.linspace(0.5, 30, 1000)

    Q_plot = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Fine Q steps for smooth peak trace
    Q_fine = np.linspace(0.8, 6.0, 200)
    exc_pf, exc_pv = [], []
    acc_pf, acc_pv = [], []

    for Q in Q_fine:
        H = transmissibility(freqs, f0, Q)
        exc = INPUT_DISP * H
        omega = 2.0 * np.pi * freqs
        acc = omega**2 * exc / G_MM_S2
        ei = np.argmax(exc)
        ai = np.argmax(acc)
        exc_pf.append(freqs[ei])
        exc_pv.append(exc[ei])
        acc_pf.append(freqs[ai])
        acc_pv.append(acc[ai])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Q_plot)))

    exc_dots_f, exc_dots_v = [], []
    acc_dots_f, acc_dots_v = [], []

    for Q, color in zip(Q_plot, colors):
        H = transmissibility(freqs, f0, Q)
        exc = INPUT_DISP * H
        omega = 2.0 * np.pi * freqs
        acc = omega**2 * exc / G_MM_S2

        ei = np.argmax(exc)
        ai = np.argmax(acc)
        exc_dots_f.append(freqs[ei])
        exc_dots_v.append(exc[ei])
        acc_dots_f.append(freqs[ai])
        acc_dots_v.append(acc[ai])

        ax1.plot(freqs, exc, color=color, linewidth=1.8, label=f'Q = {Q:g}')
        ax2.plot(freqs, acc, color=color, linewidth=1.8, label=f'Q = {Q:g}')

    # Smooth peak trace
    ax1.plot(exc_pf, exc_pv, 'r--', linewidth=1.5, alpha=0.7, zorder=10)
    ax1.scatter(exc_dots_f, exc_dots_v, color='red', s=16, zorder=11)

    ax2.plot(acc_pf, acc_pv, 'r--', linewidth=1.5, alpha=0.7, zorder=10)
    ax2.scatter(acc_dots_f, acc_dots_v, color='red', s=16, zorder=11)

    ax1.set_ylabel('Excursion (mm)')
    ax1.set_title(f'Q Sensitivity — {mass:g}g, {compliance:g} µm/mN, f₀ = {f0:.2f} Hz',
                   fontsize=11)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Acceleration (g)')
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'Q_sensitivity.png')
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Q Sensitivity Plot')
    parser.add_argument('mass', type=float, nargs='?', default=17.0,
                        help='Total effective mass in grams (default: 17)')
    parser.add_argument('compliance', type=float, nargs='?', default=26.0,
                        help='Compliance in µm/mN (default: 26)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output PNG path (default: Q_sensitivity.png)')
    args = parser.parse_args()
    plot_q_sensitivity(args.mass, args.compliance, args.output)


if __name__ == '__main__':
    main()
