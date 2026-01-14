#!/usr/bin/env python3
"""
Model Diagnostics Visualizations

Visualize MCMC diagnostics, posterior distributions, and model checks.

Usage:
    python dataviz/03_model_diagnostics.py --model outputs/model.nc --output outputs/figures/diagnostics
"""

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")


def plot_trace_plots(trace: az.InferenceData, output_dir: Path) -> None:
    """Plot MCMC trace plots for key parameters."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    params = ["mu", "alpha", "phi"]

    for idx, param in enumerate(params):
        # Trace plot
        ax_trace = axes[idx, 0]
        for chain in range(trace.posterior.dims["chain"]):
            values = trace.posterior[param].sel(chain=chain).values
            ax_trace.plot(values, alpha=0.7, lw=0.5, label=f"Chain {chain}")
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_ylabel(param)
        ax_trace.set_title(f"Trace: {param}")
        if idx == 0:
            ax_trace.legend(loc="upper right", fontsize=8)

        # Posterior histogram
        ax_hist = axes[idx, 1]
        all_values = trace.posterior[param].values.flatten()
        ax_hist.hist(all_values, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        ax_hist.axvline(np.mean(all_values), color="red", linestyle="--", lw=2, label=f"Mean: {np.mean(all_values):.3f}")
        ax_hist.axvline(np.median(all_values), color="green", linestyle="-", lw=2, label=f"Median: {np.median(all_values):.3f}")
        ax_hist.set_xlabel(param)
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(f"Posterior: {param}")
        ax_hist.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "01_trace_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 01_trace_plots.png")


def plot_posterior_distributions(trace: az.InferenceData, output_dir: Path) -> None:
    """Plot detailed posterior distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # μ posterior with prior overlay
    ax1 = axes[0, 0]
    mu_samples = trace.posterior["mu"].values.flatten()
    ax1.hist(mu_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white", label="Posterior")

    # Prior: Exponential(0.1)
    x_prior = np.linspace(0, mu_samples.max(), 100)
    prior_pdf = 0.1 * np.exp(-0.1 * x_prior)
    ax1.plot(x_prior, prior_pdf, "r--", lw=2, label="Prior: Exp(0.1)")

    ax1.set_xlabel("μ (Population Mean Rate)")
    ax1.set_ylabel("Density")
    ax1.set_title("Posterior Distribution of μ")
    ax1.legend()

    # α posterior
    ax2 = axes[0, 1]
    alpha_samples = trace.posterior["alpha"].values.flatten()
    ax2.hist(alpha_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white", label="Posterior")

    # Prior: HalfNormal(2)
    x_prior = np.linspace(0, alpha_samples.max(), 100)
    prior_pdf = 2 * np.exp(-x_prior**2 / (2 * 4)) / np.sqrt(2 * np.pi * 4)
    ax2.plot(x_prior, prior_pdf, "r--", lw=2, label="Prior: HalfNormal(2)")

    ax2.set_xlabel("α (Concentration)")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior Distribution of α")
    ax2.legend()

    # φ posterior
    ax3 = axes[1, 0]
    phi_samples = trace.posterior["phi"].values.flatten()
    ax3.hist(phi_samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white", label="Posterior")

    ax3.set_xlabel("φ (Overdispersion)")
    ax3.set_ylabel("Density")
    ax3.set_title("Posterior Distribution of φ")
    ax3.legend()

    # Joint posterior μ vs α
    ax4 = axes[1, 1]
    ax4.scatter(mu_samples[::10], alpha_samples[::10], alpha=0.3, s=5)
    ax4.set_xlabel("μ")
    ax4.set_ylabel("α")
    ax4.set_title("Joint Posterior: μ vs α")

    plt.tight_layout()
    fig.savefig(output_dir / "02_posterior_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 02_posterior_distributions.png")


def plot_entity_rates(trace: az.InferenceData, output_dir: Path, n_entities_to_show: int = 50) -> None:
    """Plot entity-level rate posteriors."""
    theta = trace.posterior["theta"].values

    # Flatten chains
    theta_flat = theta.reshape(-1, theta.shape[-1])
    n_entities = theta_flat.shape[1]

    # Compute summary statistics
    theta_mean = np.mean(theta_flat, axis=0)
    theta_std = np.std(theta_flat, axis=0)
    theta_lower = np.percentile(theta_flat, 5, axis=0)
    theta_upper = np.percentile(theta_flat, 95, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Distribution of entity means
    ax1 = axes[0, 0]
    ax1.hist(theta_mean, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax1.axvline(np.mean(theta_mean), color="red", linestyle="--", label=f"Mean: {np.mean(theta_mean):.2f}")
    ax1.set_xlabel("Entity Rate (θ)")
    ax1.set_ylabel("Number of Entities")
    ax1.set_title("Distribution of Entity Rate Posteriors")
    ax1.legend()

    # Uncertainty vs mean
    ax2 = axes[0, 1]
    ax2.scatter(theta_mean, theta_std, alpha=0.5, s=20)
    ax2.set_xlabel("Posterior Mean θ")
    ax2.set_ylabel("Posterior Std θ")
    ax2.set_title("Uncertainty vs Rate Level")

    # Sample of entity posteriors (credible intervals)
    ax3 = axes[1, 0]
    n_show = min(n_entities_to_show, n_entities)
    sorted_idx = np.argsort(theta_mean)[::-1][:n_show]

    y_pos = np.arange(n_show)
    ax3.errorbar(
        theta_mean[sorted_idx],
        y_pos,
        xerr=[theta_mean[sorted_idx] - theta_lower[sorted_idx],
              theta_upper[sorted_idx] - theta_mean[sorted_idx]],
        fmt="o",
        markersize=4,
        capsize=2,
        color="steelblue",
        alpha=0.7,
    )
    ax3.set_ylabel("Entity (sorted by mean)")
    ax3.set_xlabel("Rate θ (90% CI)")
    ax3.set_title(f"Entity Rate Posteriors (Top {n_show})")
    ax3.invert_yaxis()

    # Shrinkage visualization
    ax4 = axes[1, 1]
    population_mean = np.mean(trace.posterior["mu"].values)

    # Calculate shrinkage (how much each entity shrinks toward population mean)
    shrinkage = (theta_mean - population_mean) / (theta_mean.max() - population_mean + 0.01)

    ax4.hist(shrinkage, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax4.axvline(0, color="red", linestyle="--", label="Population Mean")
    ax4.set_xlabel("Relative Distance from Population Mean")
    ax4.set_ylabel("Number of Entities")
    ax4.set_title("Partial Pooling: Shrinkage toward Population Mean")
    ax4.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "03_entity_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 03_entity_rates.png")


def plot_convergence_diagnostics(trace: az.InferenceData, output_dir: Path) -> None:
    """Plot convergence diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])

    # R-hat values
    ax1 = axes[0, 0]
    r_hat_values = summary["r_hat"].values
    params = summary.index.tolist()
    colors = ["green" if r < 1.01 else "orange" if r < 1.05 else "red" for r in r_hat_values]
    ax1.barh(params, r_hat_values, color=colors, alpha=0.7)
    ax1.axvline(1.0, color="black", linestyle="-", lw=1)
    ax1.axvline(1.01, color="green", linestyle="--", lw=1, label="Good (<1.01)")
    ax1.axvline(1.05, color="red", linestyle="--", lw=1, label="Warning (>1.05)")
    ax1.set_xlabel("R-hat")
    ax1.set_title("Convergence: R-hat Statistics")
    ax1.legend()

    # ESS values
    ax2 = axes[0, 1]
    ess_bulk = summary["ess_bulk"].values
    ess_tail = summary["ess_tail"].values

    x = np.arange(len(params))
    width = 0.35
    ax2.barh(x - width/2, ess_bulk, width, label="ESS Bulk", alpha=0.7)
    ax2.barh(x + width/2, ess_tail, width, label="ESS Tail", alpha=0.7)
    ax2.axvline(400, color="green", linestyle="--", lw=1, label="Min recommended (400)")
    ax2.set_yticks(x)
    ax2.set_yticklabels(params)
    ax2.set_xlabel("Effective Sample Size")
    ax2.set_title("Convergence: Effective Sample Size")
    ax2.legend()

    # Divergences
    ax3 = axes[1, 0]
    if "diverging" in trace.sample_stats:
        divergences = trace.sample_stats["diverging"].values.flatten()
        n_divergent = divergences.sum()
        n_total = len(divergences)

        ax3.pie(
            [n_divergent, n_total - n_divergent],
            labels=[f"Divergent ({n_divergent})", f"OK ({n_total - n_divergent})"],
            colors=["red", "green"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax3.set_title(f"Divergent Transitions ({n_divergent}/{n_total})")
    else:
        ax3.text(0.5, 0.5, "No divergence data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Divergent Transitions")

    # Energy diagnostic
    ax4 = axes[1, 1]
    if "energy" in trace.sample_stats:
        energy = trace.sample_stats["energy"].values.flatten()
        ax4.hist(energy, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax4.set_xlabel("Energy")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Energy Distribution (HMC Diagnostic)")
    else:
        ax4.text(0.5, 0.5, "No energy data", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Energy Distribution")

    plt.tight_layout()
    fig.savefig(output_dir / "04_convergence_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 04_convergence_diagnostics.png")


def plot_posterior_predictive_check(trace: az.InferenceData, output_dir: Path) -> None:
    """Plot posterior predictive checks."""
    if "posterior_predictive" not in trace.groups() or "y" not in trace.posterior_predictive:
        print("  Skipping posterior predictive check (no data)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    y_obs = trace.observed_data["y_obs"].values
    y_pred = trace.posterior_predictive["y"].values

    # Flatten predictions
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    # Predicted vs observed distribution
    ax1 = axes[0, 0]
    ax1.hist(y_obs, bins=50, density=True, alpha=0.7, label="Observed", color="steelblue", edgecolor="white")

    # Sample from predictions
    pred_samples = y_pred_flat[np.random.choice(len(y_pred_flat), 1000, replace=False), :]
    for sample in pred_samples[:50]:
        ax1.hist(sample, bins=50, density=True, alpha=0.02, color="red")
    ax1.hist(y_pred_flat.mean(axis=0), bins=50, density=True, alpha=0.5, label="Predicted (mean)", color="red")

    ax1.set_xlabel("Event Count")
    ax1.set_ylabel("Density")
    ax1.set_title("Posterior Predictive Check: Distribution")
    ax1.legend()

    # Predicted mean vs observed
    ax2 = axes[0, 1]
    y_pred_mean = y_pred_flat.mean(axis=0)
    ax2.scatter(y_obs, y_pred_mean, alpha=0.3, s=10)
    max_val = max(y_obs.max(), y_pred_mean.max())
    ax2.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect prediction")
    ax2.set_xlabel("Observed")
    ax2.set_ylabel("Predicted (mean)")
    ax2.set_title("Observed vs Predicted")
    ax2.legend()

    # Residuals
    ax3 = axes[1, 0]
    residuals = y_obs - y_pred_mean
    ax3.hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="red", linestyle="--", lw=2)
    ax3.set_xlabel("Residual (Observed - Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title(f"Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})")

    # Coverage
    ax4 = axes[1, 1]
    y_pred_lower = np.percentile(y_pred_flat, 5, axis=0)
    y_pred_upper = np.percentile(y_pred_flat, 95, axis=0)
    covered = (y_obs >= y_pred_lower) & (y_obs <= y_pred_upper)
    coverage = covered.mean()

    ax4.bar(["Covered", "Not Covered"], [covered.sum(), (~covered).sum()],
            color=["green", "red"], alpha=0.7)
    ax4.set_ylabel("Number of Observations")
    ax4.set_title(f"90% Credible Interval Coverage: {coverage:.1%}")

    plt.tight_layout()
    fig.savefig(output_dir / "05_posterior_predictive_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 05_posterior_predictive_check.png")


def plot_autocorrelation(trace: az.InferenceData, output_dir: Path) -> None:
    """Plot autocorrelation for key parameters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = ["mu", "alpha", "phi"]

    for idx, param in enumerate(params):
        ax = axes[idx]
        samples = trace.posterior[param].values[0, :]  # First chain

        # Calculate autocorrelation
        n = len(samples)
        max_lag = min(100, n // 4)
        acf = np.correlate(samples - samples.mean(), samples - samples.mean(), mode="full")
        acf = acf[n-1:n-1+max_lag] / acf[n-1]

        ax.bar(range(max_lag), acf, color="steelblue", alpha=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.axhline(0.05, color="red", linestyle="--", lw=1, alpha=0.5)
        ax.axhline(-0.05, color="red", linestyle="--", lw=1, alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Autocorrelation: {param}")

    plt.tight_layout()
    fig.savefig(output_dir / "06_autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: 06_autocorrelation.png")


def main():
    parser = argparse.ArgumentParser(description="Model Diagnostics Visualizations")
    parser.add_argument("--model", "-m", type=str, default="outputs/model.nc", help="Model trace file")
    parser.add_argument("--output", "-o", type=str, default="outputs/figures/diagnostics", help="Output directory")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("MODEL DIAGNOSTICS VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Load trace
    print(f"Loading model from {args.model}...")
    trace = az.from_netcdf(args.model)

    # Print summary
    print("\nModel Summary:")
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])
    print(summary.to_string())
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # Generate all plots
    plot_trace_plots(trace, output_dir)
    plot_posterior_distributions(trace, output_dir)
    plot_entity_rates(trace, output_dir)
    plot_convergence_diagnostics(trace, output_dir)
    plot_posterior_predictive_check(trace, output_dir)
    plot_autocorrelation(trace, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
