"""
Generate comprehensive model diagnostics for UNSW-NB15 model.
Similar to the synthetic data diagnostics but for real data.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Paths
UNSW_MODEL_PATH = Path("outputs/unsw/model.nc")
UNSW_SCORED_PATH = Path("outputs/unsw/scored_df.parquet")
UNSW_MODELING_PATH = Path("outputs/unsw/modeling_table.parquet")
OUTPUT_DIR = Path("outputs/unsw/diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading UNSW model and data...")
trace = az.from_netcdf(UNSW_MODEL_PATH)
scored_df = pd.read_parquet(UNSW_SCORED_PATH)
modeling_df = pd.read_parquet(UNSW_MODELING_PATH)

print(f"Model loaded: {trace.posterior.dims}")
print(f"Scored observations: {len(scored_df)}")


def create_comprehensive_diagnostics():
    """Create a comprehensive diagnostics dashboard."""

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('UNSW-NB15 Model Health Dashboard', fontsize=18, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Row 1: MCMC Diagnostics
    # =========================================================================

    # 1.1 R-hat summary
    ax1 = fig.add_subplot(gs[0, 0])
    summary = az.summary(trace, var_names=["mu", "alpha", "phi"])
    rhats = summary["r_hat"].values
    params = summary.index.tolist()
    colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhats]
    bars = ax1.barh(params, rhats, color=colors, alpha=0.7)
    ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='Ideal (1.0)')
    ax1.axvline(x=1.01, color='green', linestyle='--', linewidth=1, label='Good (<1.01)')
    ax1.axvline(x=1.05, color='orange', linestyle='--', linewidth=1, label='Warning (<1.05)')
    ax1.set_xlabel('R-hat')
    ax1.set_title('Convergence: R-hat\n(All should be < 1.01)', fontsize=10, fontweight='bold')
    ax1.set_xlim(0.99, max(1.06, max(rhats) + 0.01))
    ax1.legend(fontsize=7, loc='lower right')

    # 1.2 ESS summary
    ax2 = fig.add_subplot(gs[0, 1])
    ess_bulk = summary["ess_bulk"].values
    ess_tail = summary["ess_tail"].values
    x = np.arange(len(params))
    width = 0.35
    ax2.barh(x - width/2, ess_bulk, width, label='ESS Bulk', color='steelblue', alpha=0.7)
    ax2.barh(x + width/2, ess_tail, width, label='ESS Tail', color='coral', alpha=0.7)
    ax2.axvline(x=400, color='green', linestyle='--', linewidth=1, label='Good (>400)')
    ax2.axvline(x=100, color='orange', linestyle='--', linewidth=1, label='Min (>100)')
    ax2.set_yticks(x)
    ax2.set_yticklabels(params)
    ax2.set_xlabel('Effective Sample Size')
    ax2.set_title('Sampling Efficiency: ESS\n(Higher is better)', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right')

    # 1.3 Divergences
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
    ax3.set_title(f'MCMC Health: Divergences\n({n_divergences}/{n_total} = {pct_divergent:.2f}%)',
                  fontsize=10, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.set_ylim(0, max(2, pct_divergent * 1.5))

    # 1.4 Summary stats text
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')

    # Get diagnostics
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

    Chains: {trace.posterior.dims['chain']}
    Draws:  {trace.posterior.dims['draw']}
    Entities: {trace.posterior.dims.get('entity', 'N/A')}
    """
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Diagnostic Summary', fontsize=10, fontweight='bold')

    # =========================================================================
    # Row 2: Posterior Distributions
    # =========================================================================

    # 2.1 mu posterior
    ax5 = fig.add_subplot(gs[1, 0])
    mu_samples = trace.posterior["mu"].values.flatten()
    ax5.hist(mu_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax5.axvline(mu_samples.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {mu_samples.mean():.2f}')
    ax5.axvline(np.percentile(mu_samples, 2.5), color='red', linestyle='--', linewidth=1)
    ax5.axvline(np.percentile(mu_samples, 97.5), color='red', linestyle='--', linewidth=1,
                label=f'95% CI: [{np.percentile(mu_samples, 2.5):.2f}, {np.percentile(mu_samples, 97.5):.2f}]')
    ax5.set_xlabel('μ (Population Mean Rate)')
    ax5.set_ylabel('Density')
    ax5.set_title('Posterior: μ\n(Global average event rate)', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=7)

    # 2.2 alpha posterior
    ax6 = fig.add_subplot(gs[1, 1])
    alpha_samples = trace.posterior["alpha"].values.flatten()
    ax6.hist(alpha_samples, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax6.axvline(alpha_samples.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {alpha_samples.mean():.2f}')
    ax6.axvline(np.percentile(alpha_samples, 2.5), color='red', linestyle='--', linewidth=1)
    ax6.axvline(np.percentile(alpha_samples, 97.5), color='red', linestyle='--', linewidth=1,
                label=f'95% CI: [{np.percentile(alpha_samples, 2.5):.2f}, {np.percentile(alpha_samples, 97.5):.2f}]')
    ax6.set_xlabel('α (Concentration/Pooling)')
    ax6.set_ylabel('Density')
    ax6.set_title('Posterior: α\n(Entity heterogeneity)', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=7)

    # 2.3 phi posterior
    ax7 = fig.add_subplot(gs[1, 2])
    phi_samples = trace.posterior["phi"].values.flatten()
    ax7.hist(phi_samples, bins=50, density=True, alpha=0.7, color='mediumseagreen', edgecolor='white')
    ax7.axvline(phi_samples.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {phi_samples.mean():.2f}')
    ax7.axvline(np.percentile(phi_samples, 2.5), color='red', linestyle='--', linewidth=1)
    ax7.axvline(np.percentile(phi_samples, 97.5), color='red', linestyle='--', linewidth=1,
                label=f'95% CI: [{np.percentile(phi_samples, 2.5):.2f}, {np.percentile(phi_samples, 97.5):.2f}]')
    ax7.set_xlabel('φ (Overdispersion)')
    ax7.set_ylabel('Density')
    ax7.set_title('Posterior: φ\n(Variance control)', fontsize=10, fontweight='bold')
    ax7.legend(fontsize=7)

    # 2.4 theta (entity rates) distribution
    ax8 = fig.add_subplot(gs[1, 3])
    theta_samples = trace.posterior["theta"].values
    theta_means = theta_samples.mean(axis=(0, 1))  # Mean across chains and draws
    ax8.hist(theta_means, bins=30, density=True, alpha=0.7, color='purple', edgecolor='white')
    ax8.axvline(theta_means.mean(), color='red', linestyle='-', linewidth=2,
                label=f'Mean of means: {theta_means.mean():.2f}')
    ax8.set_xlabel('θ (Entity-specific rates)')
    ax8.set_ylabel('Density')
    ax8.set_title(f'Entity Rates Distribution\n({len(theta_means)} entities)', fontsize=10, fontweight='bold')
    ax8.legend(fontsize=7)

    # =========================================================================
    # Row 3: Trace Plots (Chain mixing)
    # =========================================================================

    # 3.1 mu trace
    ax9 = fig.add_subplot(gs[2, 0])
    for chain in range(trace.posterior.dims['chain']):
        ax9.plot(trace.posterior["mu"].values[chain, :], alpha=0.7, linewidth=0.5, label=f'Chain {chain}')
    ax9.set_xlabel('Draw')
    ax9.set_ylabel('μ')
    ax9.set_title('Trace: μ (should look like fuzzy caterpillar)', fontsize=10, fontweight='bold')
    ax9.legend(fontsize=7, loc='upper right')

    # 3.2 alpha trace
    ax10 = fig.add_subplot(gs[2, 1])
    for chain in range(trace.posterior.dims['chain']):
        ax10.plot(trace.posterior["alpha"].values[chain, :], alpha=0.7, linewidth=0.5, label=f'Chain {chain}')
    ax10.set_xlabel('Draw')
    ax10.set_ylabel('α')
    ax10.set_title('Trace: α (chains should overlap)', fontsize=10, fontweight='bold')
    ax10.legend(fontsize=7, loc='upper right')

    # 3.3 phi trace
    ax11 = fig.add_subplot(gs[2, 2])
    for chain in range(trace.posterior.dims['chain']):
        ax11.plot(trace.posterior["phi"].values[chain, :], alpha=0.7, linewidth=0.5, label=f'Chain {chain}')
    ax11.set_xlabel('Draw')
    ax11.set_ylabel('φ')
    ax11.set_title('Trace: φ (no trends = good)', fontsize=10, fontweight='bold')
    ax11.legend(fontsize=7, loc='upper right')

    # 3.4 Selected theta traces
    ax12 = fig.add_subplot(gs[2, 3])
    n_entities_to_show = min(5, trace.posterior.dims.get('entity', 0))
    if n_entities_to_show > 0:
        # Select entities with different rates
        entity_means = trace.posterior["theta"].values.mean(axis=(0, 1))
        sorted_idx = np.argsort(entity_means)
        selected = sorted_idx[::max(1, len(sorted_idx)//n_entities_to_show)][:n_entities_to_show]

        for i, ent_idx in enumerate(selected):
            theta_ent = trace.posterior["theta"].values[:, :, ent_idx].flatten()
            ax12.hist(theta_ent, bins=30, alpha=0.5, label=f'Entity {ent_idx} (μ={entity_means[ent_idx]:.1f})')
    ax12.set_xlabel('θ')
    ax12.set_ylabel('Density')
    ax12.set_title('Sample Entity Posteriors\n(Different baselines)', fontsize=10, fontweight='bold')
    ax12.legend(fontsize=7)

    # =========================================================================
    # Row 4: Model Fit & Predictions
    # =========================================================================

    # 4.1 Predicted vs Actual
    ax13 = fig.add_subplot(gs[3, 0])
    if 'predicted_mean' in scored_df.columns and 'event_count' in scored_df.columns:
        ax13.scatter(scored_df['predicted_mean'], scored_df['event_count'],
                    alpha=0.3, s=10, c='steelblue')
        max_val = max(scored_df['predicted_mean'].max(), scored_df['event_count'].max())
        ax13.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit (y=x)')
        ax13.set_xlabel('Predicted Mean')
        ax13.set_ylabel('Actual Count')
        ax13.set_title('Predicted vs Actual\n(Points near line = good fit)', fontsize=10, fontweight='bold')
        ax13.legend(fontsize=7)

    # 4.2 Residuals
    ax14 = fig.add_subplot(gs[3, 1])
    if 'predicted_mean' in scored_df.columns and 'event_count' in scored_df.columns:
        residuals = scored_df['event_count'] - scored_df['predicted_mean']
        ax14.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax14.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero (ideal)')
        ax14.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=2,
                    label=f'Mean: {residuals.mean():.2f}')
        ax14.set_xlabel('Residual (Actual - Predicted)')
        ax14.set_ylabel('Density')
        ax14.set_title('Residual Distribution\n(Centered at 0 = unbiased)', fontsize=10, fontweight='bold')
        ax14.legend(fontsize=7)

    # 4.3 Score distribution by class
    ax15 = fig.add_subplot(gs[3, 2])
    if 'anomaly_score' in scored_df.columns and 'has_attack' in scored_df.columns:
        benign = scored_df[~scored_df['has_attack']]['anomaly_score']
        attack = scored_df[scored_df['has_attack']]['anomaly_score']
        ax15.hist(benign, bins=30, alpha=0.6, label=f'Normal (n={len(benign)})',
                 color='steelblue', density=True)
        ax15.hist(attack, bins=30, alpha=0.6, label=f'Attack (n={len(attack)})',
                 color='crimson', density=True)
        ax15.set_xlabel('Anomaly Score')
        ax15.set_ylabel('Density')
        ax15.set_title('Score Separation\n(Less overlap = better)', fontsize=10, fontweight='bold')
        ax15.legend(fontsize=7)

    # 4.4 Performance metrics
    ax16 = fig.add_subplot(gs[3, 3])
    ax16.axis('off')

    # Load metrics if available
    import json
    metrics_path = Path("outputs/unsw/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        perf_text = f"""
    MODEL PERFORMANCE (UNSW-NB15)
    ════════════════════════════

    PR-AUC:          {metrics.get('pr_auc', 'N/A'):.4f}
    ROC-AUC:         {metrics.get('roc_auc', 'N/A'):.4f}

    Attack Rate:     {metrics.get('attack_rate', 0)*100:.2f}%
    Observations:    {metrics.get('n_observations', 'N/A'):,}
    Attacks:         {metrics.get('n_positives', 'N/A'):,}

    Recall@10:       {metrics.get('recall_at_10', 'N/A'):.2%}
    Recall@50:       {metrics.get('recall_at_50', 'N/A'):.2%}
    Precision@10:    {metrics.get('precision_at_10', 'N/A'):.2%}
    """
    else:
        perf_text = "Metrics file not found"

    ax16.text(0.1, 0.5, perf_text, transform=ax16.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax16.set_title('Performance Summary', fontsize=10, fontweight='bold')

    plt.savefig(OUTPUT_DIR / 'model_health_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'model_health_dashboard.png'}")


def create_entity_analysis():
    """Create entity-specific analysis plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('UNSW-NB15: Entity-Level Analysis', fontsize=14, fontweight='bold')

    # Get theta posteriors
    theta_samples = trace.posterior["theta"].values
    theta_means = theta_samples.mean(axis=(0, 1))
    theta_stds = theta_samples.std(axis=(0, 1))

    # 1. Entity rates with uncertainty
    ax1 = axes[0, 0]
    sorted_idx = np.argsort(theta_means)
    y_pos = np.arange(len(theta_means))

    # Show subset if too many
    if len(theta_means) > 30:
        step = len(theta_means) // 30
        show_idx = sorted_idx[::step]
    else:
        show_idx = sorted_idx

    ax1.errorbar(theta_means[show_idx], range(len(show_idx)),
                xerr=theta_stds[show_idx]*1.96, fmt='o', capsize=3,
                color='steelblue', alpha=0.7, markersize=4)
    ax1.set_xlabel('θ (Entity Rate) with 95% CI')
    ax1.set_ylabel('Entity (sorted by rate)')
    ax1.set_title('Entity-Specific Baselines\n(Each entity has its own "normal")')

    # 2. Coefficient of variation
    ax2 = axes[0, 1]
    cv = theta_stds / theta_means  # Coefficient of variation
    ax2.hist(cv, bins=30, color='coral', alpha=0.7, edgecolor='white')
    ax2.axvline(cv.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean CV: {cv.mean():.2f}')
    ax2.set_xlabel('Coefficient of Variation (σ/μ)')
    ax2.set_ylabel('Count')
    ax2.set_title('Uncertainty per Entity\n(Higher = less certain)')
    ax2.legend()

    # 3. Shrinkage analysis
    ax3 = axes[1, 0]
    mu_mean = trace.posterior["mu"].values.mean()
    shrinkage = (theta_means - mu_mean) / mu_mean  # Relative deviation from population
    ax3.scatter(theta_means, np.abs(shrinkage), alpha=0.5, s=20, c='purple')
    ax3.axhline(0, color='red', linestyle='--', label=f'Population mean (μ={mu_mean:.1f})')
    ax3.set_xlabel('Entity Rate (θ)')
    ax3.set_ylabel('|Relative Deviation from μ|')
    ax3.set_title('Partial Pooling Effect\n(Extreme values shrink toward μ)')
    ax3.legend()

    # 4. Rate vs uncertainty
    ax4 = axes[1, 1]
    scatter = ax4.scatter(theta_means, theta_stds, c=cv, cmap='viridis',
                         alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax4, label='CV')
    ax4.set_xlabel('Mean Rate (θ)')
    ax4.set_ylabel('Uncertainty (σ)')
    ax4.set_title('Rate vs Uncertainty\n(Larger rates may have more uncertainty)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'entity_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'entity_analysis.png'}")


def create_posterior_predictive_check():
    """Create posterior predictive check plots."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('UNSW-NB15: Posterior Predictive Checks', fontsize=14, fontweight='bold')

    # Get actual data
    y_obs = modeling_df['event_count'].values

    # Check if we have posterior predictive samples
    if hasattr(trace, 'posterior_predictive') and 'y' in trace.posterior_predictive:
        y_pred = trace.posterior_predictive['y'].values
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

        # 1. Distribution comparison
        ax1 = axes[0]
        ax1.hist(y_obs, bins=50, density=True, alpha=0.7, label='Observed', color='steelblue')

        # Sample some predictions for visualization
        for i in range(min(100, y_pred_flat.shape[0])):
            ax1.hist(y_pred_flat[i], bins=50, density=True, alpha=0.02, color='coral')
        ax1.hist(y_pred_flat.mean(axis=0), bins=50, density=True, alpha=0.5,
                label='Predicted (mean)', color='coral')

        ax1.set_xlabel('Event Count')
        ax1.set_ylabel('Density')
        ax1.set_title('Observed vs Predicted Distribution')
        ax1.legend()
        ax1.set_xlim(0, np.percentile(y_obs, 99))

        # 2. Mean comparison per observation
        ax2 = axes[1]
        pred_means = y_pred_flat.mean(axis=0)
        ax2.scatter(y_obs, pred_means, alpha=0.3, s=10)
        max_val = max(y_obs.max(), pred_means.max())
        ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
        ax2.set_xlabel('Observed')
        ax2.set_ylabel('Predicted Mean')
        ax2.set_title('Observed vs Predicted')
        ax2.legend()

        # 3. Coverage check
        ax3 = axes[2]
        pred_lower = np.percentile(y_pred_flat, 5, axis=0)
        pred_upper = np.percentile(y_pred_flat, 95, axis=0)
        in_interval = (y_obs >= pred_lower) & (y_obs <= pred_upper)
        coverage = in_interval.mean() * 100

        ax3.bar(['Inside 90% CI', 'Outside 90% CI'],
               [coverage, 100-coverage],
               color=['green', 'red'], alpha=0.7)
        ax3.axhline(90, color='black', linestyle='--', label='Expected (90%)')
        ax3.set_ylabel('Percentage')
        ax3.set_title(f'Prediction Interval Coverage\n(Actual: {coverage:.1f}%)')
        ax3.legend()
        ax3.set_ylim(0, 100)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'Posterior predictive\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('N/A')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_predictive_check.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'posterior_predictive_check.png'}")


if __name__ == "__main__":
    print("="*60)
    print("Creating UNSW-NB15 Model Diagnostics")
    print("="*60)

    create_comprehensive_diagnostics()
    create_entity_analysis()
    create_posterior_predictive_check()

    print("="*60)
    print(f"All diagnostics saved to: {OUTPUT_DIR}")
    print("="*60)
