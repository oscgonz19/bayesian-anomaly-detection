#!/usr/bin/env python3
"""
Full Visualization Report

Generate a complete visualization report for the entire BSAD pipeline.

Usage:
    python dataviz/06_full_report.py --data data/events.parquet --model outputs/model.nc --scores outputs/scores.parquet --output outputs/figures
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def run_visualization_script(script_name: str, args: list) -> bool:
    """Run a visualization script and return success status."""
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: {script_name} had errors:")
            print(result.stderr[:500] if result.stderr else "No error output")
            return False
        return True
    except Exception as e:
        print(f"  Error running {script_name}: {e}")
        return False


def create_report_cover(output_dir: Path) -> None:
    """Create a report cover page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    # Title
    ax.text(0.5, 0.7, "Bayesian Security\nAnomaly Detection",
            ha="center", va="center", fontsize=32, fontweight="bold",
            transform=ax.transAxes)

    ax.text(0.5, 0.5, "Visualization Report",
            ha="center", va="center", fontsize=24,
            transform=ax.transAxes, color="gray")

    # Date
    ax.text(0.5, 0.3, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            ha="center", va="center", fontsize=14,
            transform=ax.transAxes)

    # Description
    description = """
    This report contains visualizations for:

    1. Data Exploration - Raw event data analysis
    2. Feature Engineering - Modeling table and feature distributions
    3. Model Diagnostics - MCMC convergence and posterior analysis
    4. Anomaly Results - Detection results and rankings
    5. Evaluation Metrics - PR-AUC, ROC-AUC, Recall@K
    """
    ax.text(0.5, 0.1, description,
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes, family="monospace")

    plt.tight_layout()
    fig.savefig(output_dir / "00_report_cover.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_section_header(section_name: str, output_dir: Path, section_num: int) -> None:
    """Create a section header image."""
    fig, ax = plt.subplots(figsize=(11, 2))
    ax.axis("off")

    ax.text(0.5, 0.5, f"Section {section_num}: {section_name}",
            ha="center", va="center", fontsize=24, fontweight="bold",
            transform=ax.transAxes)

    ax.axhline(0.2, color="steelblue", lw=3, xmin=0.1, xmax=0.9)

    plt.tight_layout()
    fig.savefig(output_dir / f"section_{section_num:02d}_header.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine_to_pdf(output_dir: Path, pdf_name: str = "full_report.pdf") -> None:
    """Combine all PNG images into a single PDF."""
    try:
        from PIL import Image

        # Gather all PNG files in order
        png_files = sorted(output_dir.rglob("*.png"))

        if not png_files:
            print("  No PNG files found to combine into PDF")
            return

        # Create PDF
        pdf_path = output_dir / pdf_name
        images = []

        for png_file in png_files:
            img = Image.open(png_file)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        if images:
            images[0].save(pdf_path, save_all=True, append_images=images[1:])
            print(f"\n  Created PDF report: {pdf_path}")

    except ImportError:
        print("  Note: Install Pillow to generate PDF report: pip install Pillow")
    except Exception as e:
        print(f"  Error creating PDF: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Full Visualization Report")
    parser.add_argument("--data", "-d", type=str, default="data/events.parquet", help="Events data file")
    parser.add_argument("--model", "-m", type=str, default="outputs/model.nc", help="Model trace file")
    parser.add_argument("--scores", "-s", type=str, default="outputs/scores.parquet", help="Scores file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures", help="Output directory")
    parser.add_argument("--pdf", action="store_true", help="Generate combined PDF report")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BSAD - FULL VISUALIZATION REPORT")
    print(f"{'='*60}\n")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create report cover
    print("Creating report cover...")
    create_report_cover(output_dir)

    # Track which sections succeeded
    sections = []

    # Section 1: Data Exploration
    print("\n" + "="*40)
    print("Section 1: Data Exploration")
    print("="*40)
    data_path = Path(args.data)
    if data_path.exists():
        create_section_header("Data Exploration", output_dir, 1)
        success = run_visualization_script("01_data_exploration.py", [
            "--input", str(data_path),
            "--output", str(output_dir / "exploration")
        ])
        sections.append(("Data Exploration", success))
    else:
        print(f"  Skipping: {data_path} not found")
        sections.append(("Data Exploration", False))

    # Section 2: Feature Analysis
    print("\n" + "="*40)
    print("Section 2: Feature Analysis")
    print("="*40)
    if data_path.exists():
        create_section_header("Feature Analysis", output_dir, 2)
        success = run_visualization_script("02_feature_analysis.py", [
            "--input", str(data_path),
            "--output", str(output_dir / "features")
        ])
        sections.append(("Feature Analysis", success))
    else:
        print(f"  Skipping: {data_path} not found")
        sections.append(("Feature Analysis", False))

    # Section 3: Model Diagnostics
    print("\n" + "="*40)
    print("Section 3: Model Diagnostics")
    print("="*40)
    model_path = Path(args.model)
    if model_path.exists():
        create_section_header("Model Diagnostics", output_dir, 3)
        success = run_visualization_script("03_model_diagnostics.py", [
            "--model", str(model_path),
            "--output", str(output_dir / "diagnostics")
        ])
        sections.append(("Model Diagnostics", success))
    else:
        print(f"  Skipping: {model_path} not found")
        sections.append(("Model Diagnostics", False))

    # Section 4: Anomaly Results
    print("\n" + "="*40)
    print("Section 4: Anomaly Results")
    print("="*40)
    scores_path = Path(args.scores)
    if scores_path.exists():
        create_section_header("Anomaly Results", output_dir, 4)
        success = run_visualization_script("04_anomaly_results.py", [
            "--scores", str(scores_path),
            "--output", str(output_dir / "results")
        ])
        sections.append(("Anomaly Results", success))
    else:
        print(f"  Skipping: {scores_path} not found")
        sections.append(("Anomaly Results", False))

    # Section 5: Evaluation Metrics
    print("\n" + "="*40)
    print("Section 5: Evaluation Metrics")
    print("="*40)
    if scores_path.exists():
        create_section_header("Evaluation Metrics", output_dir, 5)
        success = run_visualization_script("05_evaluation_plots.py", [
            "--scores", str(scores_path),
            "--output", str(output_dir / "evaluation")
        ])
        sections.append(("Evaluation Metrics", success))
    else:
        print(f"  Skipping: {scores_path} not found")
        sections.append(("Evaluation Metrics", False))

    # Summary
    print("\n" + "="*60)
    print("REPORT GENERATION SUMMARY")
    print("="*60)
    for section_name, success in sections:
        status = "✓" if success else "✗"
        print(f"  {status} {section_name}")

    successful = sum(1 for _, s in sections if s)
    print(f"\n  Completed: {successful}/{len(sections)} sections")
    print(f"  Output directory: {output_dir}")

    # Generate PDF if requested
    if args.pdf:
        print("\nGenerating PDF report...")
        combine_to_pdf(output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
