"""
Generate explanatory diagrams for BSAD documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_hierarchical_model_diagram():
    """Create a visual diagram of the hierarchical Bayesian model."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Hierarchical Negative Binomial Model',
            fontsize=18, fontweight='bold', ha='center', va='center')

    # Colors
    pop_color = '#E8F4FD'  # Light blue for population
    entity_color = '#FFF3E0'  # Light orange for entity
    obs_color = '#E8F5E9'  # Light green for observations

    # Population Level Box
    pop_box = FancyBboxPatch((1, 7), 12, 1.8, boxstyle="round,pad=0.1",
                              facecolor=pop_color, edgecolor='#1976D2', linewidth=2)
    ax.add_patch(pop_box)
    ax.text(7, 8.2, 'POPULATION LEVEL', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#1976D2')
    ax.text(4, 7.5, r'$\mu \sim \mathrm{Exponential}(0.1)$', fontsize=14, ha='center')
    ax.text(10, 7.5, r'$\alpha \sim \mathrm{HalfNormal}(2.0)$', fontsize=14, ha='center')
    ax.text(4, 7.1, 'Global mean rate', fontsize=10, ha='center', style='italic', color='gray')
    ax.text(10, 7.1, 'Pooling strength', fontsize=10, ha='center', style='italic', color='gray')

    # Entity Level Box
    entity_box = FancyBboxPatch((1, 4), 12, 2.2, boxstyle="round,pad=0.1",
                                 facecolor=entity_color, edgecolor='#F57C00', linewidth=2)
    ax.add_patch(entity_box)
    ax.text(7, 5.8, 'ENTITY LEVEL (Partial Pooling)', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#F57C00')
    ax.text(7, 5.1, r'$\theta_e \sim \mathrm{Gamma}(\mu \cdot \alpha, \alpha)$', fontsize=14, ha='center')
    ax.text(7, 4.6, 'Entity-specific rate', fontsize=10, ha='center', style='italic', color='gray')

    # Entity circles
    for i, (x, label) in enumerate([(3, r'$\theta_1$'), (5.5, r'$\theta_2$'),
                                      (8.5, r'$\theta_3$'), (11, r'$\theta_n$')]):
        circle = Circle((x, 4.3), 0.25, facecolor='white', edgecolor='#F57C00', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, 4.3, label, fontsize=10, ha='center', va='center')
    ax.text(7, 4.3, '...', fontsize=14, ha='center', va='center')

    # Observation Level Box
    obs_box = FancyBboxPatch((1, 1), 12, 2.2, boxstyle="round,pad=0.1",
                              facecolor=obs_color, edgecolor='#388E3C', linewidth=2)
    ax.add_patch(obs_box)
    ax.text(7, 2.8, 'OBSERVATION LEVEL', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#388E3C')
    ax.text(4, 2.1, r'$\phi \sim \mathrm{HalfNormal}(5.0)$', fontsize=14, ha='center')
    ax.text(10, 2.1, r'$y_{e,t} \sim \mathrm{NegBin}(\theta_e, \phi)$', fontsize=14, ha='center')
    ax.text(4, 1.7, 'Overdispersion', fontsize=10, ha='center', style='italic', color='gray')
    ax.text(10, 1.7, 'Observed counts', fontsize=10, ha='center', style='italic', color='gray')

    # Arrows
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=arrow_style, color='#666666')

    # Pop to Entity
    ax.annotate("", xy=(7, 6.3), xytext=(7, 6.9),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

    # Entity to Obs
    ax.annotate("", xy=(7, 3.3), xytext=(7, 3.9),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

    # Side annotations
    ax.text(0.3, 7.9, 'Hyperpriors\nlearned from\nall data', fontsize=9, ha='left',
            va='center', color='#1976D2', style='italic')
    ax.text(0.3, 5.1, 'Each entity\ngets own rate\n(shrunk to pop)', fontsize=9, ha='left',
            va='center', color='#F57C00', style='italic')
    ax.text(0.3, 2.1, 'Counts per\nentity-time\nwindow', fontsize=9, ha='left',
            va='center', color='#388E3C', style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hierarchical_model_diagram.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'hierarchical_model_diagram.png'}")


def create_pipeline_architecture():
    """Create a diagram of the BSAD pipeline architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(8, 7.5, 'BSAD Pipeline Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # Colors for each stage
    colors = {
        'data': '#E3F2FD',
        'feature': '#FFF8E1',
        'model': '#FCE4EC',
        'score': '#E8F5E9',
        'output': '#F3E5F5'
    }

    # Stage 1: Data Ingestion
    box1 = FancyBboxPatch((0.5, 4.5), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['data'], edgecolor='#1565C0', linewidth=2)
    ax.add_patch(box1)
    ax.text(1.75, 6.1, '1. DATA', fontsize=11, fontweight='bold', ha='center', color='#1565C0')
    ax.text(1.75, 5.5, 'Load raw data', fontsize=10, ha='center')
    ax.text(1.75, 5.1, 'UNSW-NB15 or', fontsize=9, ha='center', color='gray')
    ax.text(1.75, 4.8, 'synthetic counts', fontsize=9, ha='center', color='gray')

    # Stage 2: Feature Engineering
    box2 = FancyBboxPatch((3.5, 4.5), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['feature'], edgecolor='#F9A825', linewidth=2)
    ax.add_patch(box2)
    ax.text(4.75, 6.1, '2. FEATURES', fontsize=11, fontweight='bold', ha='center', color='#F9A825')
    ax.text(4.75, 5.5, 'Entity extraction', fontsize=10, ha='center')
    ax.text(4.75, 5.1, 'Count aggregation', fontsize=9, ha='center', color='gray')
    ax.text(4.75, 4.8, 'Time windowing', fontsize=9, ha='center', color='gray')

    # Stage 3: Model Training
    box3 = FancyBboxPatch((6.5, 4.5), 3, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['model'], edgecolor='#C62828', linewidth=2)
    ax.add_patch(box3)
    ax.text(8, 6.1, '3. BAYESIAN MODEL', fontsize=11, fontweight='bold', ha='center', color='#C62828')
    ax.text(8, 5.5, 'MCMC Sampling', fontsize=10, ha='center')
    ax.text(8, 5.1, 'Posterior inference', fontsize=9, ha='center', color='gray')
    ax.text(8, 4.8, 'PyMC + NUTS', fontsize=9, ha='center', color='gray')

    # Stage 4: Scoring
    box4 = FancyBboxPatch((10, 4.5), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['score'], edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(box4)
    ax.text(11.25, 6.1, '4. SCORING', fontsize=11, fontweight='bold', ha='center', color='#2E7D32')
    ax.text(11.25, 5.5, 'Posterior predictive', fontsize=10, ha='center')
    ax.text(11.25, 5.1, r'$-\log P(y|\theta)$', fontsize=10, ha='center', color='gray')
    ax.text(11.25, 4.8, 'Uncertainty bounds', fontsize=9, ha='center', color='gray')

    # Stage 5: Output
    box5 = FancyBboxPatch((13, 4.5), 2.5, 2, boxstyle="round,pad=0.1",
                           facecolor=colors['output'], edgecolor='#6A1B9A', linewidth=2)
    ax.add_patch(box5)
    ax.text(14.25, 6.1, '5. OUTPUT', fontsize=11, fontweight='bold', ha='center', color='#6A1B9A')
    ax.text(14.25, 5.5, 'Ranked anomalies', fontsize=10, ha='center')
    ax.text(14.25, 5.1, 'Visualizations', fontsize=9, ha='center', color='gray')
    ax.text(14.25, 4.8, 'Metrics & reports', fontsize=9, ha='center', color='gray')

    # Arrows between stages
    for x in [3, 6, 9.5, 12.5]:
        ax.annotate("", xy=(x+0.4, 5.5), xytext=(x, 5.5),
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=2))

    # Bottom: Key outputs at each stage
    ax.text(1.75, 3.8, 'raw_df', fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(4.75, 3.8, 'modeling_table', fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(8, 3.8, 'model.nc', fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(11.25, 3.8, 'scores.parquet', fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    ax.text(14.25, 3.8, 'plots/*.png', fontsize=9, ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    # Bottom section: Key insight
    insight_box = FancyBboxPatch((2, 1), 12, 2, boxstyle="round,pad=0.1",
                                  facecolor='#FAFAFA', edgecolor='#9E9E9E', linewidth=1)
    ax.add_patch(insight_box)
    ax.text(8, 2.5, 'Key Innovation: Entity-Specific Baselines',
            fontsize=12, fontweight='bold', ha='center')
    ax.text(8, 1.9, 'Same count value can be normal for one entity but anomalous for another',
            fontsize=10, ha='center', color='#666666')
    ax.text(8, 1.4, 'Example: 50 packets is NORMAL for HTTP but ANOMALOUS for DNS',
            fontsize=10, ha='center', style='italic', color='#666666')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pipeline_architecture.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'pipeline_architecture.png'}")


def create_scoring_explanation():
    """Create a diagram explaining the anomaly scoring mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Normal distribution for entity
    ax1 = axes[0]
    x = np.linspace(0, 50, 100)
    # Simulate a negative binomial-like distribution
    from scipy import stats
    r, p = 5, 0.3
    pmf = stats.nbinom.pmf(np.arange(50), r, p)
    ax1.bar(np.arange(50), pmf, color='#64B5F6', alpha=0.7, label='Expected distribution')
    ax1.axvline(x=10, color='green', linestyle='--', linewidth=2, label='Normal observation (y=10)')
    ax1.axvline(x=40, color='red', linestyle='--', linewidth=2, label='Anomalous observation (y=40)')
    ax1.set_xlabel('Count value', fontsize=11)
    ax1.set_ylabel('Probability', fontsize=11)
    ax1.set_title('Step 1: Entity Distribution\n(Posterior Predictive)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(0, 50)

    # Plot 2: Log probability transformation
    ax2 = axes[1]
    counts = np.arange(1, 50)
    log_probs = -np.log(stats.nbinom.pmf(counts, r, p) + 1e-10)
    ax2.plot(counts, log_probs, 'b-', linewidth=2)
    ax2.scatter([10], [-np.log(stats.nbinom.pmf(10, r, p) + 1e-10)],
                color='green', s=100, zorder=5, label='y=10: Low score')
    ax2.scatter([40], [-np.log(stats.nbinom.pmf(40, r, p) + 1e-10)],
                color='red', s=100, zorder=5, label='y=40: High score')
    ax2.set_xlabel('Count value', fontsize=11)
    ax2.set_ylabel('Anomaly Score = -log P(y|posterior)', fontsize=11)
    ax2.set_title('Step 2: Score Transformation\n(Higher = More Anomalous)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 50)

    # Plot 3: Score with uncertainty
    ax3 = axes[2]
    ranks = np.arange(1, 21)
    scores = 15 - np.log(ranks) * 3 + np.random.randn(20) * 0.5
    uncertainties = np.random.uniform(0.5, 2, 20)
    colors = ['red' if i < 5 else 'orange' if i < 10 else 'green' for i in range(20)]

    ax3.errorbar(ranks, scores, yerr=uncertainties, fmt='o', capsize=4,
                 color='#1976D2', ecolor='#90CAF9', markersize=8)
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Threshold')
    ax3.fill_between([0, 21], [10, 10], [15, 15], alpha=0.2, color='red', label='High risk zone')
    ax3.set_xlabel('Anomaly Rank', fontsize=11)
    ax3.set_ylabel('Score with 90% CI', fontsize=11)
    ax3.set_title('Step 3: Prioritized Output\n(With Uncertainty Bounds)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 21)

    plt.suptitle('How BSAD Anomaly Scoring Works', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'scoring_explanation.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'scoring_explanation.png'}")


def create_partial_pooling_diagram():
    """Create a diagram explaining partial pooling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    np.random.seed(42)

    # Generate data
    n_entities = 8
    true_rates = [5, 8, 12, 15, 20, 25, 35, 50]
    observations = [np.random.poisson(r, size=np.random.randint(5, 50)) for r in true_rates]
    observed_means = [obs.mean() for obs in observations]
    n_obs = [len(obs) for obs in observations]

    global_mean = np.mean([np.mean(obs) for obs in observations])

    # Plot 1: No Pooling
    ax1 = axes[0]
    ax1.barh(range(n_entities), observed_means, color='#EF5350', alpha=0.7)
    ax1.axvline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Global mean: {global_mean:.1f}')
    ax1.set_yticks(range(n_entities))
    ax1.set_yticklabels([f'Entity {i+1}\n(n={n_obs[i]})' for i in range(n_entities)], fontsize=9)
    ax1.set_xlabel('Estimated Rate', fontsize=11)
    ax1.set_title('NO POOLING\n(Each entity independent)', fontsize=12, fontweight='bold', color='#C62828')
    ax1.legend(fontsize=9)
    ax1.text(0.5, -0.15, 'Problem: High variance\nfor sparse entities',
             transform=ax1.transAxes, ha='center', fontsize=10, color='#C62828')

    # Plot 2: Complete Pooling
    ax2 = axes[1]
    ax2.barh(range(n_entities), [global_mean] * n_entities, color='#42A5F5', alpha=0.7)
    ax2.axvline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Global mean: {global_mean:.1f}')
    ax2.set_yticks(range(n_entities))
    ax2.set_yticklabels([f'Entity {i+1}\n(n={n_obs[i]})' for i in range(n_entities)], fontsize=9)
    ax2.set_xlabel('Estimated Rate', fontsize=11)
    ax2.set_title('COMPLETE POOLING\n(All entities same)', fontsize=12, fontweight='bold', color='#1565C0')
    ax2.legend(fontsize=9)
    ax2.text(0.5, -0.15, 'Problem: Ignores\nindividual differences',
             transform=ax2.transAxes, ha='center', fontsize=10, color='#1565C0')

    # Plot 3: Partial Pooling (Bayesian)
    ax3 = axes[2]
    # Shrinkage towards mean, more for sparse entities
    partial_estimates = []
    for i in range(n_entities):
        shrinkage = 1 / (1 + n_obs[i] / 10)  # More shrinkage for fewer observations
        partial = shrinkage * global_mean + (1 - shrinkage) * observed_means[i]
        partial_estimates.append(partial)

    colors = ['#66BB6A' if n > 20 else '#FFA726' for n in n_obs]
    ax3.barh(range(n_entities), partial_estimates, color=colors, alpha=0.7)
    ax3.axvline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Global mean: {global_mean:.1f}')
    ax3.set_yticks(range(n_entities))
    ax3.set_yticklabels([f'Entity {i+1}\n(n={n_obs[i]})' for i in range(n_entities)], fontsize=9)
    ax3.set_xlabel('Estimated Rate', fontsize=11)
    ax3.set_title('PARTIAL POOLING (BSAD)\n(Adaptive shrinkage)', fontsize=12, fontweight='bold', color='#2E7D32')
    ax3.legend(fontsize=9)
    ax3.text(0.5, -0.15, 'Sparse entities shrink to global\nDense entities keep their own rate',
             transform=ax3.transAxes, ha='center', fontsize=10, color='#2E7D32')

    plt.suptitle('Partial Pooling: The Key Innovation in Hierarchical Models',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'partial_pooling_explained.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Created: {OUTPUT_DIR / 'partial_pooling_explained.png'}")


def main():
    """Generate all diagrams."""
    print("Generating BSAD explanatory diagrams...")
    print("=" * 50)

    create_hierarchical_model_diagram()
    create_pipeline_architecture()
    create_scoring_explanation()
    create_partial_pooling_diagram()

    print("=" * 50)
    print(f"All diagrams saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
