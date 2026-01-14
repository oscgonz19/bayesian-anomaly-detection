"""
Train BSAD model on UNSW-NB15 rare-attack datasets (1%, 2%, 5%).
These datasets are properly designed for anomaly detection.
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

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/rare_attack")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 2% attack rate (good balance)
ATTACK_RATE = "2pct"
DATA_PATH = DATA_DIR / f"unsw_nb15_rare_attack_{ATTACK_RATE}.parquet"

print("="*70)
print(f"BSAD Training on Rare-Attack Dataset ({ATTACK_RATE})")
print("="*70)

# Load data
print(f"\n1. Loading {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Check attack rate
attack_col = 'label' if 'label' in df.columns else 'has_attack'
attack_rate = df[attack_col].mean()
print(f"   Attack rate: {attack_rate:.2%}")

# Create entity from proto + service
print("\n2. Creating entities and windows...")
df['entity'] = df['proto'].astype(str) + '_' + df['service'].astype(str)

# Create windows by grouping consecutive rows per entity
WINDOW_SIZE = 500  # Smaller windows for more observations
df['entity_row'] = df.groupby('entity').cumcount()
df['window'] = df['entity_row'] // WINDOW_SIZE

# Aggregate by entity and window
print("\n3. Building modeling table...")
count_col = 'spkts'  # Source packets - good count variable

agg_funcs = {
    count_col: 'sum',
    'dpkts': 'sum',
    'sbytes': 'sum',
    'dbytes': 'sum',
    attack_col: 'any',
}

if 'attack_cat' in df.columns:
    agg_funcs['attack_cat'] = lambda x: x[x != 'Normal'].iloc[0] if (x != 'Normal').any() else 'Normal'

modeling_df = df.groupby(['entity', 'window']).agg(agg_funcs).reset_index()
modeling_df = modeling_df.rename(columns={count_col: 'event_count', attack_col: 'has_attack'})

# Filter entities with minimum windows
MIN_WINDOWS = 3
entity_counts = modeling_df.groupby('entity').size()
valid_entities = entity_counts[entity_counts >= MIN_WINDOWS].index
modeling_df = modeling_df[modeling_df['entity'].isin(valid_entities)].reset_index(drop=True)

# Entity encoding
unique_entities = modeling_df['entity'].unique()
entity_mapping = {e: i for i, e in enumerate(unique_entities)}
modeling_df['entity_idx'] = modeling_df['entity'].map(entity_mapping)

n_entities = len(unique_entities)
n_windows = len(modeling_df)
final_attack_rate = modeling_df['has_attack'].mean()

print(f"   Entities: {n_entities}")
print(f"   Windows: {n_windows}")
print(f"   Final attack rate: {final_attack_rate:.2%}")

# Get arrays
y = modeling_df['event_count'].values.astype(np.int64)
entity_idx = modeling_df['entity_idx'].values.astype(np.int64)

# Model settings
N_CHAINS = 4
N_SAMPLES = 2000
N_TUNE = 1000
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

print(f"\n4. Training Bayesian model...")
print(f"   Chains: {N_CHAINS}, Samples: {N_SAMPLES}, Tune: {N_TUNE}")

coords = {"entity": np.arange(n_entities), "obs": np.arange(len(y))}

with pm.Model(coords=coords) as model:
    entity_idx_data = pm.Data("entity_idx", entity_idx, dims="obs")
    y_data = pm.Data("y_obs", y, dims="obs")

    # Priors
    mu = pm.Exponential("mu", lam=0.01)  # Higher prior mean for packet counts
    alpha = pm.HalfNormal("alpha", sigma=2.0)
    theta = pm.Gamma("theta", alpha=mu * alpha, beta=alpha, dims="entity")
    phi = pm.HalfNormal("phi", sigma=5.0)

    # Likelihood
    pm.NegativeBinomial("y", mu=theta[entity_idx_data], alpha=phi, observed=y_data, dims="obs")

# Fit
rng = np.random.default_rng(RANDOM_SEED)
seed = int(rng.integers(0, 2**31))

with model:
    trace = pm.sample(
        draws=N_SAMPLES,
        tune=N_TUNE,
        chains=N_CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=seed,
        cores=min(N_CHAINS, os.cpu_count() or 1),
        return_inferencedata=True,
        progressbar=True,
    )
    print("\n   Generating posterior predictive...")
    trace.extend(pm.sample_posterior_predictive(trace, random_seed=seed + 1))

# Save model
trace.to_netcdf(OUTPUT_DIR / f"model_{ATTACK_RATE}.nc")
print(f"\n5. Saved model: {OUTPUT_DIR / f'model_{ATTACK_RATE}.nc'}")

# Compute scores
print("\n6. Computing anomaly scores...")
theta_samples = trace.posterior["theta"].values
phi_samples = trace.posterior["phi"].values

n_chains, n_draws, n_ents = theta_samples.shape
theta_flat = theta_samples.reshape(-1, n_ents)
phi_flat = phi_samples.reshape(-1)
n_samples_total = theta_flat.shape[0]

log_likelihoods = np.zeros((n_samples_total, len(y)))
for s in range(n_samples_total):
    mu_s = theta_flat[s, entity_idx]
    phi_s = phi_flat[s]
    log_likelihoods[s, :] = stats.nbinom.logpmf(y, n=phi_s, p=phi_s/(phi_s + mu_s))

avg_log_lik = logsumexp(log_likelihoods, axis=0) - np.log(n_samples_total)
anomaly_scores = -avg_log_lik
score_std = np.std(-log_likelihoods, axis=0)
score_lower = np.percentile(-log_likelihoods, 5, axis=0)
score_upper = np.percentile(-log_likelihoods, 95, axis=0)

# Predictions
ppc = trace.posterior_predictive['y'].values.reshape(-1, len(y))
predicted_mean = ppc.mean(axis=0)
predicted_lower = np.percentile(ppc, 5, axis=0)
predicted_upper = np.percentile(ppc, 95, axis=0)

# Create scored dataframe
scored_df = modeling_df.copy()
scored_df['anomaly_score'] = anomaly_scores
scored_df['score_std'] = score_std
scored_df['score_lower'] = score_lower
scored_df['score_upper'] = score_upper
scored_df['predicted_mean'] = predicted_mean
scored_df['predicted_lower'] = predicted_lower
scored_df['predicted_upper'] = predicted_upper
scored_df['anomaly_rank'] = scored_df['anomaly_score'].rank(ascending=False, method='first').astype(int)
scored_df = scored_df.sort_values('anomaly_score', ascending=False)

scored_df.to_parquet(OUTPUT_DIR / f"scored_df_{ATTACK_RATE}.parquet")

# Evaluate
print("\n7. Evaluating...")
y_true = scored_df['has_attack'].astype(int).values
scores = scored_df['anomaly_score'].values

metrics = {
    "attack_rate_config": ATTACK_RATE,
    "pr_auc": float(average_precision_score(y_true, scores)),
    "roc_auc": float(roc_auc_score(y_true, scores)),
    "n_observations": len(y_true),
    "n_positives": int(y_true.sum()),
    "attack_rate": float(y_true.mean()),
    "n_entities": n_entities,
    "n_chains": N_CHAINS,
}

for k in [10, 25, 50, 100]:
    if k <= len(y_true):
        n_pos = y_true.sum()
        top_k = np.argsort(scores)[-k:]
        tp = y_true[top_k].sum()
        metrics[f"recall_at_{k}"] = float(tp / n_pos) if n_pos > 0 else 0.0
        metrics[f"precision_at_{k}"] = float(tp / k)

with open(OUTPUT_DIR / f"metrics_{ATTACK_RATE}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n   PR-AUC: {metrics['pr_auc']:.4f}")
print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"   Recall@10: {metrics.get('recall_at_10', 0):.2%}")
print(f"   Recall@50: {metrics.get('recall_at_50', 0):.2%}")

# ============================================================================
# COMPREHENSIVE DIAGNOSTICS DASHBOARD
# ============================================================================
print("\n8. Generating diagnostics dashboard...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'BSAD Model Health Dashboard - Rare Attack {ATTACK_RATE.upper()}\n(Proper Anomaly Detection Regime)',
             fontsize=16, fontweight='bold', y=0.98)

gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# Get summary
summary = az.summary(trace, var_names=["mu", "alpha", "phi"])

# === ROW 1: MCMC DIAGNOSTICS ===
ax1 = fig.add_subplot(gs[0, 0])
rhats = summary["r_hat"].values
params = summary.index.tolist()
colors = ['green' if r < 1.01 else 'orange' if r < 1.05 else 'red' for r in rhats]
ax1.barh(params, rhats, color=colors, alpha=0.7)
ax1.axvline(x=1.0, color='black', linestyle='-', linewidth=2)
ax1.axvline(x=1.01, color='green', linestyle='--', label='Good')
ax1.axvline(x=1.05, color='orange', linestyle='--', label='Warning')
ax1.set_xlabel('R-hat')
ax1.set_title('Convergence: R-hat\n(All < 1.01 = PASS)', fontweight='bold')
ax1.set_xlim(0.99, max(1.06, max(rhats)+0.01))
ax1.legend(fontsize=7)

ax2 = fig.add_subplot(gs[0, 1])
ess_bulk = summary["ess_bulk"].values
ess_tail = summary["ess_tail"].values
x = np.arange(len(params))
ax2.barh(x - 0.2, ess_bulk, 0.35, label='Bulk', color='steelblue', alpha=0.7)
ax2.barh(x + 0.2, ess_tail, 0.35, label='Tail', color='coral', alpha=0.7)
ax2.axvline(x=400, color='green', linestyle='--', label='>400 Good')
ax2.set_yticks(x)
ax2.set_yticklabels(params)
ax2.set_xlabel('ESS')
ax2.set_title('Sampling Efficiency\n(Higher = Better)', fontweight='bold')
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[0, 2])
divs = trace.sample_stats["diverging"].values.flatten()
n_div = divs.sum()
n_tot = len(divs)
pct = 100 * n_div / n_tot
col = 'green' if pct < 0.1 else 'orange' if pct < 1 else 'red'
ax3.bar(['Divergences'], [pct], color=col, alpha=0.7, width=0.5)
ax3.axhline(y=0.1, color='green', linestyle='--')
ax3.axhline(y=1.0, color='orange', linestyle='--')
ax3.set_ylabel('%')
ax3.set_title(f'Divergences: {n_div}/{n_tot}\n({pct:.2f}%)', fontweight='bold')
ax3.set_ylim(0, max(2, pct*1.5))

ax4 = fig.add_subplot(gs[0, 3])
ax4.axis('off')
r_max = summary["r_hat"].max()
ess_min = min(summary["ess_bulk"].min(), summary["ess_tail"].min())
st_r = "PASS" if r_max < 1.01 else "WARN" if r_max < 1.05 else "FAIL"
st_e = "PASS" if ess_min > 400 else "WARN" if ess_min > 100 else "FAIL"
st_d = "PASS" if pct < 0.1 else "WARN" if pct < 1 else "FAIL"
overall = "HEALTHY" if all(s=="PASS" for s in [st_r, st_e, st_d]) else "CHECK"

txt = f"""
MODEL HEALTH SUMMARY
════════════════════

R-hat (max):   {r_max:.4f} [{st_r}]
ESS (min):     {ess_min:.0f} [{st_e}]
Divergences:   {pct:.2f}% [{st_d}]

════════════════════
OVERALL: {overall}

Config: {ATTACK_RATE} attack rate
Chains: {N_CHAINS}
Draws: {N_SAMPLES}
Entities: {n_entities}
"""
ax4.text(0.1, 0.5, txt, transform=ax4.transAxes, fontsize=10, va='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# === ROW 2: POSTERIORS ===
mu_s = trace.posterior["mu"].values.flatten()
alpha_s = trace.posterior["alpha"].values.flatten()
phi_s = trace.posterior["phi"].values.flatten()

ax5 = fig.add_subplot(gs[1, 0])
ax5.hist(mu_s, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax5.axvline(mu_s.mean(), color='red', lw=2, label=f'Mean: {mu_s.mean():.1f}')
ax5.set_xlabel('μ (Population Mean)')
ax5.set_title('Posterior: μ', fontweight='bold')
ax5.legend(fontsize=8)

ax6 = fig.add_subplot(gs[1, 1])
ax6.hist(alpha_s, bins=50, density=True, alpha=0.7, color='coral', edgecolor='white')
ax6.axvline(alpha_s.mean(), color='red', lw=2, label=f'Mean: {alpha_s.mean():.3f}')
ax6.set_xlabel('α (Concentration)')
ax6.set_title('Posterior: α', fontweight='bold')
ax6.legend(fontsize=8)

ax7 = fig.add_subplot(gs[1, 2])
ax7.hist(phi_s, bins=50, density=True, alpha=0.7, color='mediumseagreen', edgecolor='white')
ax7.axvline(phi_s.mean(), color='red', lw=2, label=f'Mean: {phi_s.mean():.2f}')
ax7.set_xlabel('φ (Overdispersion)')
ax7.set_title('Posterior: φ', fontweight='bold')
ax7.legend(fontsize=8)

ax8 = fig.add_subplot(gs[1, 3])
theta_means = trace.posterior["theta"].values.mean(axis=(0,1))
ax8.hist(theta_means, bins=20, density=True, alpha=0.7, color='purple', edgecolor='white')
ax8.set_xlabel('θ (Entity Rates)')
ax8.set_title(f'Entity Rates ({n_entities} entities)', fontweight='bold')

# === ROW 3: TRACES ===
n_ch = trace.posterior.sizes['chain']

ax9 = fig.add_subplot(gs[2, 0])
for c in range(n_ch):
    ax9.plot(trace.posterior["mu"].values[c], alpha=0.7, lw=0.5)
ax9.set_xlabel('Draw')
ax9.set_ylabel('μ')
ax9.set_title('Trace: μ (fuzzy caterpillar = good)', fontweight='bold')

ax10 = fig.add_subplot(gs[2, 1])
for c in range(n_ch):
    ax10.plot(trace.posterior["alpha"].values[c], alpha=0.7, lw=0.5)
ax10.set_xlabel('Draw')
ax10.set_ylabel('α')
ax10.set_title('Trace: α (chains overlap = good)', fontweight='bold')

ax11 = fig.add_subplot(gs[2, 2])
for c in range(n_ch):
    ax11.plot(trace.posterior["phi"].values[c], alpha=0.7, lw=0.5)
ax11.set_xlabel('Draw')
ax11.set_ylabel('φ')
ax11.set_title('Trace: φ (no trends = good)', fontweight='bold')

ax12 = fig.add_subplot(gs[2, 3])
sorted_idx = np.argsort(theta_means)
for i in range(min(5, len(sorted_idx))):
    idx = sorted_idx[i * max(1, len(sorted_idx)//5)]
    th = trace.posterior["theta"].values[:,:,idx].flatten()
    ax12.hist(th, bins=30, alpha=0.5, label=f'E{idx}')
ax12.set_xlabel('θ')
ax12.set_title('Sample Entity Posteriors', fontweight='bold')
ax12.legend(fontsize=6)

# === ROW 4: MODEL FIT ===
ax13 = fig.add_subplot(gs[3, 0])
ax13.scatter(scored_df['predicted_mean'], scored_df['event_count'], alpha=0.4, s=15, c='steelblue')
mx = max(scored_df['predicted_mean'].max(), scored_df['event_count'].max())
ax13.plot([0, mx], [0, mx], 'r--', lw=2, label='y=x')
ax13.set_xlabel('Predicted')
ax13.set_ylabel('Actual')
ax13.set_title('Predicted vs Actual', fontweight='bold')
ax13.legend()

ax14 = fig.add_subplot(gs[3, 1])
resid = scored_df['event_count'] - scored_df['predicted_mean']
ax14.hist(resid, bins=40, density=True, alpha=0.7, color='steelblue')
ax14.axvline(0, color='red', linestyle='--', lw=2)
ax14.axvline(resid.mean(), color='orange', lw=2, label=f'Mean: {resid.mean():.1f}')
ax14.set_xlabel('Residual')
ax14.set_title('Residuals (centered = unbiased)', fontweight='bold')
ax14.legend(fontsize=8)

ax15 = fig.add_subplot(gs[3, 2])
benign = scored_df[~scored_df['has_attack']]['anomaly_score']
attack = scored_df[scored_df['has_attack']]['anomaly_score']
ax15.hist(benign, bins=25, alpha=0.6, label=f'Normal (n={len(benign)})', color='steelblue', density=True)
ax15.hist(attack, bins=25, alpha=0.6, label=f'Attack (n={len(attack)})', color='crimson', density=True)
ax15.set_xlabel('Anomaly Score')
ax15.set_title('Score Separation\n(Less overlap = better)', fontweight='bold')
ax15.legend(fontsize=8)

ax16 = fig.add_subplot(gs[3, 3])
ax16.axis('off')
perf = f"""
PERFORMANCE METRICS
═══════════════════

PR-AUC:       {metrics['pr_auc']:.4f}
ROC-AUC:      {metrics['roc_auc']:.4f}

Attack Rate:  {metrics['attack_rate']*100:.2f}%
Observations: {metrics['n_observations']}
Attacks:      {metrics['n_positives']}

Recall@10:    {metrics.get('recall_at_10',0):.1%}
Recall@25:    {metrics.get('recall_at_25',0):.1%}
Recall@50:    {metrics.get('recall_at_50',0):.1%}
Prec@10:      {metrics.get('precision_at_10',0):.1%}
"""
ax16.text(0.1, 0.5, perf, transform=ax16.transAxes, fontsize=10, va='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(OUTPUT_DIR / f'model_health_dashboard_{ATTACK_RATE}.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n   Saved: {OUTPUT_DIR / f'model_health_dashboard_{ATTACK_RATE}.png'}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nThis is the CORRECT regime for BSAD:")
print(f"  - Attack rate: {final_attack_rate:.2%} (rare events)")
print(f"  - PR-AUC: {metrics['pr_auc']:.4f}")
