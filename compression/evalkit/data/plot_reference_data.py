#!/usr/bin/env python3
"""
Plot experimental reference data for all EMB diameters.
Creates juxtaposed plots showing data in both DPD units and real physical units.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import convertToDiameterFromDPDUnits, convertToForceFromDPDUnits


def load_dpd_data(diameter_um):
    """Load experimental data in DPD units"""
    here = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(here, f"compression_data_{diameter_um}um.dat")

    if not os.path.exists(data_file):
        print(f"Warning: Data file not found: {data_file}")
        return None, None

    data = np.loadtxt(data_file, skiprows=1, ndmin=2)
    disp_dpd = data[:, 0]  # Displacement in DPD units
    force_dpd = data[:, 1]  # Force in DPD units

    return disp_dpd, force_dpd


def convert_to_real_units(disp_dpd, force_dpd, diameter_um):
    """Convert DPD units to physical units (nm, nN)"""
    # Save current directory and change to project root for conversion functions
    saved_cwd = os.getcwd()

    # Find project root (where init_compression_X directories are)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    os.chdir(project_root)

    try:
        # Convert each point individually
        disp_nm = np.array([convertToDiameterFromDPDUnits(d, diameter_um) for d in disp_dpd])
        force_nN = np.array([convertToForceFromDPDUnits(f, diameter_um) for f in force_dpd])
    finally:
        # Restore original directory
        os.chdir(saved_cwd)

    return disp_nm, force_nN


def plot_reference_data():
    """Create juxtaposed plots of reference data in DPD and real units"""

    diameters = [2.1, 2.9, 3.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green
    markers = ["o", "s", "^"]

    # Create figure with two subplots
    fig, (ax_dpd, ax_real) = plt.subplots(1, 2, figsize=(14, 6))

    print("=" * 80)
    print("Plotting Reference Data for All EMB Diameters")
    print("=" * 80)
    print()

    for diameter, color, marker in zip(diameters, colors, markers):
        print(f"Loading {diameter} μm data...")

        # Load DPD data
        disp_dpd, force_dpd = load_dpd_data(diameter)

        if disp_dpd is None:
            print(f"  ⚠️  Skipping {diameter} μm (no data)")
            continue

        # Convert to real units
        disp_nm, force_nN = convert_to_real_units(disp_dpd, force_dpd, diameter)

        print(f"  ✓ Loaded {len(disp_dpd)} points")
        print(
            f"    DPD units:  disp = [{disp_dpd.min():.3f}, {disp_dpd.max():.3f}], "
            f"force = [{force_dpd.min():.1f}, {force_dpd.max():.1f}]"
        )
        print(
            f"    Real units: disp = [{disp_nm.min():.1f}, {disp_nm.max():.1f}] nm, "
            f"force = [{force_nN.min():.2f}, {force_nN.max():.2f}] nN"
        )
        print()

        # Plot in DPD units (left panel)
        ax_dpd.plot(
            disp_dpd,
            force_dpd,
            marker=marker,
            color=color,
            markersize=7,
            linewidth=2,
            alpha=0.8,
            label=f"{diameter} μm ({len(disp_dpd)} pts)",
        )

        # Plot in real units (right panel)
        ax_real.plot(
            disp_nm,
            force_nN,
            marker=marker,
            color=color,
            markersize=7,
            linewidth=2,
            alpha=0.8,
            label=f"{diameter} μm ({len(disp_nm)} pts)",
        )

    # Format DPD units plot
    ax_dpd.set_xlabel("Displacement (DPD units)", fontsize=13, fontweight="bold")
    ax_dpd.set_ylabel("Force (DPD units)", fontsize=13, fontweight="bold")
    ax_dpd.set_title("Reference Data - DPD Units", fontsize=14, fontweight="bold")
    ax_dpd.grid(True, alpha=0.3, linestyle="--")
    ax_dpd.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax_dpd.set_xlim(left=0)
    ax_dpd.set_ylim(bottom=0)

    # Format real units plot
    ax_real.set_xlabel("Displacement (nm)", fontsize=13, fontweight="bold")
    ax_real.set_ylabel("Force (nN)", fontsize=13, fontweight="bold")
    ax_real.set_title("Reference Data - Real Units", fontsize=14, fontweight="bold")
    ax_real.grid(True, alpha=0.3, linestyle="--")
    ax_real.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax_real.set_xlim(left=0)
    ax_real.set_ylim(bottom=0)

    # Add overall title
    plt.suptitle(
        "Experimental Compression Data for All EMB Diameters",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    here = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(here, "reference_data_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved to: {output_file}")

    # Also save as PDF for publication quality
    output_pdf = os.path.join(here, "reference_data_comparison.pd")
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"✓ PDF saved to: {output_pdf}")

    plt.show()

    print()
    print("=" * 80)
    print("Plotting complete!")
    print("=" * 80)


def print_data_summary():
    """Print summary statistics for all datasets"""

    diameters = [2.1, 2.9, 3.0]

    print()
    print("=" * 80)
    print("REFERENCE DATA SUMMARY")
    print("=" * 80)
    print()

    for diameter in diameters:
        disp_dpd, force_dpd = load_dpd_data(diameter)

        if disp_dpd is None:
            continue

        disp_nm, force_nN = convert_to_real_units(disp_dpd, force_dpd, diameter)

        print(f"{diameter} μm EMB:")
        print(f"  Data points: {len(disp_dpd)}")
        print("  DPD units:")
        print(f"    Displacement range: [{disp_dpd.min():.4f}, {disp_dpd.max():.4f}]")
        print(f"    Force range:        [{force_dpd.min():.2f}, {force_dpd.max():.2f}]")
        print("  Real units:")
        print(f"    Displacement range: [{disp_nm.min():.2f}, {disp_nm.max():.2f}] nm")
        print(f"    Force range:        [{force_nN.min():.3f}, {force_nN.max():.3f}] nN")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot experimental reference data in DPD and real units"
    )
    parser.add_argument("--summary", action="store_true", help="Print data summary statistics")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save plots without displaying (for compute nodes)",
    )

    args = parser.parse_args()

    # Set matplotlib backend for non-interactive mode
    if args.no_display:
        import matplotlib

        matplotlib.use("Agg")

    # Print summary if requested
    if args.summary:
        print_data_summary()

    # Generate plots
    plot_reference_data()


if __name__ == "__main__":
    main()
