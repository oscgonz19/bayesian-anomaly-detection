"""
Re-train UNSW model with 4 chains and generate comprehensive diagnostics.
"""

import os
import json
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pathlib import Path
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsad.unsw_adapter import UNSWSettings, load_unsw_data, build_modeling_table, get_model_arrays

# Output directory
OUTPUT_DIR = Path("outputs/unsw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("UNSW-NB15 Model Training with 4 Chains")
print("="*70)

# Settings with 4 chains
settings = UNSWSettings()
settings.n_chains = 4  # <-- 4 cadenas
settings.n_samples = 2000
settings.n_tune = 1000
settings.ensure_dirs()

# Load and prepare data
print("\n1. Loading UNSW-NB15 data...")
train_df, test_df = load_unsw_data(settings)
print(f"   Training: {train_df.shape}, Testing: {test_df.shape}")

print("\n2. Building modeling table...")
modeling_df, metadata = build_modeling_table(train_df, settings)
print(f"   Windows: {metadata['n_windows']}, Entities: {metadata['n_entities']}")
print(f"   Attack rate: {metadata['attack_rate']:.2%}")

# Get arrays
arrays = get_model_arrays(modeling_df)
y = arrays["y"]
entity_idx = arrays["entity_idx"]
n_entities = arrays["n_entities"]

print(f"\n3. Training Bayesian model with 4 chains...")
print(f"   Samples: {settings.n_samples}, Tune: {settings.n_tune}, Chains: {settings.n_chains}")

# Build and train model
coords = {"entity": np.arange(n_entities), "obs": np.arange(len(y))}

with pm.Model(coords=coords) as model:
    entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
    y_data = pm.Data("y_obs", y, dims="obs")

    # Priors
    mu = pm.Exponential("mu", lam=settings.mu_prior_rate)
    alpha = pm.HalfNormal("alpha", sigma=settings.alpha_prior_sd)
    theta = pm.Gamma("theta", alpha=mu * alpha, beta=alpha, dims="entity")
    phi = pm.HalfNormal("phi", sigma=settings.overdispersion_prior_sd)

    # Likelihood
    pm.NegativeBinomial("y", mu=theta[entity_idx_data], alpha=phi, observed=y_data, dims="obs")

# Fit
rng = np.random.default_rng(settings.random_seed)
seed = int(rng.integers(0, 2**31))

with model:
    trace = pm.sample(
        draws=settings.n_samples,
        tune=settings.n_tune,
        chains=settings.n_chains,
        target_accept=settings.target_accept,
        random_seed=seed,
        cores=min(settings.n_chains, os.cpu_count() or 1),
        return_inferencedata=True,
        progressbar=True,
    )
    print("\n   Generating posterior predictive samples...")
    trace.extend(pm.sample_posterior_predictive(trace, random_seed=seed + 1))

# Save model
print("\n4. Saving model...")
trace.to_netcdf(OUTPUT_DIR / "model_4chains.nc")
print(f"   Saved: {OUTPUT_DIR / 'model_4chains.nc'}")

# Compute scores
print("\n5. Computing anomaly scores...")

theta_samples = trace.posterior["theta"].values
phi_samples = trace.posterior["phi"].values

n_chains, n_draws, n_ents = theta_samples.shape
theta_flat = theta_samples.reshape(-1, n_ents)
phi_flat = phi_samples.reshape(-1)

n_samples_total = theta_flat.shape[0]
n_obs = len(y)

log_likelihoods = np.zeros((n_samples_total, n_obs))

for s in range(n_samples_total):
    mu_s = theta_flat[s, entity_idx]
    phi_s = phi_flat[s]
    n_param = phi_s
    p_param = phi_s / (phi_s + mu_s)
    log_likelihoods[s, :] = stats.nbinom.logpmf(y, n=n_param, p=p_param)

avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples_total)
anomaly_scores = -avg_log_lik

individual_scores = -log_likelihoods
score_std = np.std(individual_scores, axis=0)
score_lower = np.percentile(individual_scores, 5, axis=0)
score_upper = np.percentile(individual_scores, 95, axis=0)

# Compute intervals
if hasattr(trace, 'posterior_predictive') and 'y' in trace.posterior_predictive:
    ppc = trace.posterior_predictive['y'].values
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])
    predicted_lower = np.percentile(ppc_flat, 5, axis=0)
    predicted_upper = np.percentile(ppc_flat, 95, axis=0)
    predicted_mean = np.mean(ppc_flat, axis=0)
else:
    predicted_mean = theta_flat[:, entity_idx].mean(axis=0)
    predicted_lower = np.percentile(theta_flat[:, entity_idx], 5, axis=0)
    predicted_upper = np.percentile(theta_flat[:, entity_idx], 95, axis=0)

# Create scored dataframe
scored_df = modeling_df.copy()
scored_df["anomaly_score"] = anomaly_scores
scored_df["score_std"] = score_std
scored_df["score_lower"] = score_lower
scored_df["score_upper"] = score_upper
scored_df["predicted_mean"] = predicted_mean
scored_df["predicted_lower"] = predicted_lower
scored_df["predicted_upper"] = predicted_upper
scored_df["anomaly_rank"] = scored_df["anomaly_score"].rank(ascending=False, method="first").astype(int)
scored_df = scored_df.sort_values("anomaly_score", ascending=False)

# Save
scored_df.to_parquet(OUTPUT_DIR / "scored_df_4chains.parquet")
print(f"   Saved: {OUTPUT_DIR / 'scored_df_4chains.parquet'}")

# Evaluate
print("\n6. Evaluating model...")
y_true = scored_df["has_attack"].astype(int).values
scores = scored_df["anomaly_score"].values

metrics = {
    "pr_auc": float(average_precision_score(y_true, scores)),
    "roc_auc": float(roc_auc_score(y_true, scores)),
    "n_observations": len(y_true),
    "n_positives": int(y_true.sum()),
    "attack_rate": float(y_true.mean()),
    "n_chains": settings.n_chains,
    "n_samples": settings.n_samples,
}

for k in [10, 25, 50, 100]:
    if k <= len(y_true):
        n_positives = y_true.sum()
        top_k_idx = np.argsort(scores)[-k:]
        tp_at_k = y_true[top_k_idx].sum()
        metrics[f"recall_at_{k}"] = float(tp_at_k / n_positives) if n_positives > 0 else 0.0
        metrics[f"precision_at_{k}"] = float(tp_at_k / k)

with open(OUTPUT_DIR / "metrics_4chains.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n   PR-AUC: {metrics['pr_auc']:.4f}")
print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"   Recall@10: {metrics.get('recall_at_10', 0):.2%}")

# ============================================================================
# GENERATE COMPREHENSIVE DIAGNOSTICS
# ============================================================================

print("\n7. Generating comprehensive diagnostics...")

DIAG_DIR = OUTPUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def create_health_dashboard():
    """Create comprehensive model health dashboard."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('UNSW-NB15 Model Health Dashboard (4 Chains)', fontsize=18, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # Get summary
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])

    # Row 1: MCMC Diagnostics
    # R-hat
    ax1 = fig.add_subplot(gs[0, 0])
    rhats = summary["r_hat"].values
    params = summary.index.tolist()
    colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhats]
    ax1.barh(params, rhats, color=colors, alpha=0.7)
    ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=2)
    ax1.axvline(x=1.01, color='green', linestyle='--', linewidth=1, label='Good (<1.01)')
    ax1.axvline(x=1.05, color='orange', linestyle='--', linewidth=1, label='Warning (<1.05)')
    ax1.set_xlabel('R-hat')
    ax1.set_title('Convergence: R-hat', fontsize=10, fontweight='bold')
    ax1.set_xlim(0.99, max(1.06, max(rhats) + 0.01))
    ax1.legend(fontsize=7)

    # ESS
    ax2 = fig.add_subplot(gs[0, 1])
    ess_bulk = summary["ess_bulk"].values
    ess_tail = summary["ess_tail"].values
    x = np.arange(len(params))
    width = 0.35
    ax2.barh(x - width/2, ess_bulk, width, label='ESS Bulk', color='steelblue', alpha=0.7)
    ax2.barh(x + width/2, ess_tail, width, label='ESS Tail', color='coral', alpha=0.7)
    ax2.axvline(x=400, color='green', linestyle='--', linewidth=1, label='Good (>400)')
    ax2.set_yticks(x)
    ax2.set_yticklabels(params)
    ax2.set_xlabel('Effective Sample Size')
    ax2.set_title('Sampling Efficiency: ESS', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7)

    # Divergences
    ax3 = fig.add_subplot(gs[0, 2])
    divergences = trace.sample_stats["diverging"].values.flatten()
    n_divergences = divergences.sum()
    n_total = len(divergences)
    pct_divergent = 100 * n_divergences / n_total
    colors_div = ['green' if pct_divergent < 0.1 else 'orange' if pct_divergent < 1 else 'red']
    ax3.bar(['Divergences'], [pct_divergent], color=colors_div, alpha=0.7, width=0.5)
    ax3.axhline(y=0.1, color='green', linestyle='--', label='Good (<0.1%)')
    ax3.axhline(y=1.0, color='orange', linestyle='--', label='Warning (<1%)')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title(f'Divergences: {n_divergences}/{n_total}', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, max(2, pct_divergent * 1.5))

    # Summary text
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    r_hat_max = summary["r_hat"].max()
    ess_min = min(summary["ess_bulk"].min(), summary["ess_tail"].min())
    status_rhat = "PASS" if r_hat_max < 1.01 else "WARNING" if r_hat_max < 1.05 else "FAIL"
    status_ess = "PASS" if ess_min > 400 else "WARNING" if ess_min > 100 else "FAIL"
    status_div = "PASS" if pct_divergent < 0.1 else "WARNING" if pct_divergent < 1 else "FAIL"
    overall = "HEALTHY" if all(s == "PASS" for s in [status_rhat, status_ess, status_div]) else "CHECK"

    summary_text = f"""
MODEL HEALTH SUMMARY
════════════════════

R-hat (max):     {r_hat_max:.4f}  [{status_rhat}]
ESS (min):       {ess_min:.0f}  [{status_ess}]
Divergences:     {pct_divergent:.2f}%  [{status_div}]

════════════════════
OVERALL: {overall}

Chains: 4
Draws:  {settings.n_samples}
Entities: {n_entities}
"""
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Row 2: Posteriors
    mu_samples = trace.posterior["mu"].values.flatten()
    alpha_samples_flat = trace.posterior["alpha"].values.flatten()
    phi_samples_flat = trace.posterior["phi"].values.flatten()

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax5.axvline(mu_samples.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {mu_samples.mean():.2f}')
    ax5.set_xlabel('μ (Population Mean)')
    ax5.set_title('Posterior: μ', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=7)

    ax6 = fig.add_subplot(gs[1, 1])
    ax6.hist(alpha_samples_flat, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax6.axvline(alpha_samples_flat.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {alpha_samples_flat.mean():.2f}')
    ax6.set_xlabel('α (Concentration)')
    ax6.set_title('Posterior: α', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=7)

    ax7 = fig.add_subplot(gs[1, 2])
    ax7.hist(phi_samples_flat, bins=50, density=True, alpha=0.7, color='mediumseagreen', edgecolor='white')
    ax7.axvline(phi_samples_flat.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean: {phi_samples_flat.mean():.2f}')
    ax7.set_xlabel('φ (Overdispersion)')
    ax7.set_title('Posterior: φ', fontsize=10, fontweight='bold')
    ax7.legend(fontsize=7)

    ax8 = fig.add_subplot(gs[1, 3])
    theta_means = trace.posterior["theta"].values.mean(axis=(0, 1))
    ax8.hist(theta_means, bins=30, density=True, alpha=0.7, color='purple', edgecolor='white')
    ax8.axvline(theta_means.mean(), color='red', linestyle='-', linewidth=2)
    ax8.set_xlabel('θ (Entity Rates)')
    ax8.set_title(f'Entity Rates ({len(theta_means)} entities)', fontsize=10, fontweight='bold')

    # Row 3: Traces
    n_chains_plot = trace.posterior.sizes['chain']

    ax9 = fig.add_subplot(gs[2, 0])
    for chain in range(n_chains_plot):
        ax9.plot(trace.posterior["mu"].values[chain, :], alpha=0.7, linewidth=0.5)
    ax9.set_xlabel('Draw')
    ax9.set_ylabel('μ')
    ax9.set_title('Trace: μ', fontsize=10, fontweight='bold')

    ax10 = fig.add_subplot(gs[2, 1])
    for chain in range(n_chains_plot):
        ax10.plot(trace.posterior["alpha"].values[chain, :], alpha=0.7, linewidth=0.5)
    ax10.set_xlabel('Draw')
    ax10.set_ylabel('α')
    ax10.set_title('Trace: α', fontsize=10, fontweight='bold')

    ax11 = fig.add_subplot(gs[2, 2])
    for chain in range(n_chains_plot):
        ax11.plot(trace.posterior["phi"].values[chain, :], alpha=0.7, linewidth=0.5)
    ax11.set_xlabel('Draw')
    ax11.set_ylabel('φ')
    ax11.set_title('Trace: φ', fontsize=10, fontweight='bold')

    ax12 = fig.add_subplot(gs[2, 3])
    # Sample entity posteriors
    entity_means_all = trace.posterior["theta"].values.mean(axis=(0, 1))
    sorted_idx = np.argsort(entity_means_all)
    for i in range(min(5, len(sorted_idx))):
        ent_idx = sorted_idx[i * (len(sorted_idx)//5) if i < 4 else -1]
        theta_ent = trace.posterior["theta"].values[:, :, ent_idx].flatten()
        ax12.hist(theta_ent, bins=30, alpha=0.5, label=f'E{ent_idx}')
    ax12.set_xlabel('θ')
    ax12.set_title('Sample Entity Posteriors', fontsize=10, fontweight='bold')
    ax12.legend(fontsize=6)

    # Row 4: Model Fit
    ax13 = fig.add_subplot(gs[3, 0])
    ax13.scatter(scored_df['predicted_mean'], scored_df['event_count'], alpha=0.3, s=10, c='steelblue')
    max_val = max(scored_df['predicted_mean'].max(), scored_df['event_count'].max())
    ax13.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
    ax13.set_xlabel('Predicted')
    ax13.set_ylabel('Actual')
    ax13.set_title('Predicted vs Actual', fontsize=10, fontweight='bold')
    ax13.legend(fontsize=7)

    ax14 = fig.add_subplot(gs[3, 1])
    residuals = scored_df['event_count'] - scored_df['predicted_mean']
    ax14.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue')
    ax14.axvline(0, color='red', linestyle='--', linewidth=2)
    ax14.set_xlabel('Residual')
    ax14.set_title('Residuals', fontsize=10, fontweight='bold')

    ax15 = fig.add_subplot(gs[3, 2])
    benign = scored_df[~scored_df['has_attack']]['anomaly_score']
    attack = scored_df[scored_df['has_attack']]['anomaly_score']
    ax15.hist(benign, bins=30, alpha=0.6, label=f'Normal (n={len(benign)})', color='steelblue', density=True)
    ax15.hist(attack, bins=30, alpha=0.6, label=f'Attack (n={len(attack)})', color='crimson', density=True)
    ax15.set_xlabel('Anomaly Score')
    ax15.set_title('Score Separation', fontsize=10, fontweight='bold')
    ax15.legend(fontsize=7)

    ax16 = fig.add_subplot(gs[3, 3])
    ax16.axis('off')
    perf_text = f"""
PERFORMANCE (UNSW-NB15)
══════════════════════

PR-AUC:       {metrics['pr_auc']:.4f}
ROC-AUC:      {metrics['roc_auc']:.4f}

Attack Rate:  {metrics['attack_rate']*100:.2f}%
Observations: {metrics['n_observations']:,}

Recall@10:    {metrics.get('recall_at_10', 0):.2%}
Recall@50:    {metrics.get('recall_at_50', 0):.2%}
"""
    ax16.text(0.1, 0.5, perf_text, transform=ax16.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(DIAG_DIR / 'model_health_dashboard_4chains.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: {DIAG_DIR / 'model_health_dashboard_4chains.png'}")


create_health_dashboard()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nOutputs:")
print(f"  Model: {OUTPUT_DIR / 'model_4chains.nc'}")
print(f"  Scores: {OUTPUT_DIR / 'scored_df_4chains.parquet'}")
print(f"  Metrics: {OUTPUT_DIR / 'metrics_4chains.json'}")
print(f"  Dashboard: {DIAG_DIR / 'model_health_dashboard_4chains.png'}")
