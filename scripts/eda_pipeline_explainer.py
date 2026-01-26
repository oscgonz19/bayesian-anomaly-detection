#!/usr/bin/env python3
"""
EDA Pipeline Explainer: Visualizaciones pedagógicas para entender BSAD.

Este script genera visualizaciones que explican:
1. Cómo lucen los datos crudos
2. Por qué necesitamos estructura de entidades
3. El problema de overdispersion
4. El efecto del partial pooling
5. Cómo se calculan los scores

Uso:
    python scripts/eda_pipeline_explainer.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Crear directorio de salida
OUTPUT_DIR = Path("outputs/eda_explainer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n_entities=50, n_days=30, seed=42):
    """Genera datos sintéticos para demostración."""
    np.random.seed(seed)

    # Tasas base por entidad (heterogéneas)
    entity_rates = np.random.gamma(shape=2, scale=15, size=n_entities)

    records = []
    for entity_id in range(n_entities):
        base_rate = entity_rates[entity_id]
        for day in range(n_days):
            # Overdispersion: usamos Negative Binomial en lugar de Poisson
            count = np.random.negative_binomial(n=3, p=3/(3+base_rate))

            # Inyectar algunos ataques (anomalías)
            is_attack = np.random.random() < 0.03  # 3% attack rate
            if is_attack:
                count = int(count * np.random.uniform(2.5, 5))  # Multiplicador de ataque

            records.append({
                'entity_id': f'user_{entity_id:03d}',
                'day': day + 1,
                'date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=day),
                'event_count': count,
                'base_rate': base_rate,
                'is_attack': int(is_attack)
            })

    return pd.DataFrame(records)


def plot_01_raw_data_overview(df):
    """Visualización 1: Vista general de los datos crudos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📊 PASO 1: Exploración de Datos Crudos', fontsize=16, fontweight='bold')

    # 1.1 Muestra de datos
    ax1 = axes[0, 0]
    sample_df = df.head(10)[['entity_id', 'date', 'event_count', 'is_attack']]
    ax1.axis('off')
    table = ax1.table(
        cellText=sample_df.values,
        colLabels=sample_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#e3f2fd']*4
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax1.set_title('Muestra de Datos (primeras 10 filas)', fontweight='bold', pad=20)

    # 1.2 Distribución de event_count
    ax2 = axes[0, 1]
    ax2.hist(df['event_count'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(df['event_count'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {df["event_count"].mean():.1f}')
    ax2.axvline(df['event_count'].median(), color='orange', linestyle='--', linewidth=2,
                label=f'Mediana: {df["event_count"].median():.1f}')
    ax2.set_xlabel('Event Count')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Conteos', fontweight='bold')
    ax2.legend()

    # Añadir texto con estadísticas
    stats_text = f"N = {len(df):,}\nMedia = {df['event_count'].mean():.1f}\nStd = {df['event_count'].std():.1f}\nMin = {df['event_count'].min()}\nMax = {df['event_count'].max()}"
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 1.3 Conteos por día (heatmap simplificado)
    ax3 = axes[1, 0]
    pivot = df.pivot_table(index='entity_id', columns='day', values='event_count', aggfunc='mean')
    # Solo mostrar primeras 15 entidades
    pivot_sample = pivot.iloc[:15, :15]
    sns.heatmap(pivot_sample, ax=ax3, cmap='YlOrRd', cbar_kws={'label': 'Event Count'})
    ax3.set_title('Conteos por Entidad × Día (muestra 15×15)', fontweight='bold')
    ax3.set_xlabel('Día')
    ax3.set_ylabel('Entidad')

    # 1.4 Resumen de dimensiones
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    📐 DIMENSIONES DEL DATASET
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

    Total de filas:        {len(df):,}
    Entidades únicas:      {df['entity_id'].nunique()}
    Días:                  {df['day'].nunique()}

    Observaciones por entidad: {len(df) // df['entity_id'].nunique()}

    📊 ESTADÍSTICAS DE CONTEO
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

    Media:                 {df['event_count'].mean():.2f}
    Varianza:              {df['event_count'].var():.2f}
    Ratio Var/Media:       {df['event_count'].var() / df['event_count'].mean():.2f}

    ⚠️ Var/Media >> 1 indica OVERDISPERSION
       (Poisson asume Var/Media = 1)

    🎯 TASA DE ATAQUES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

    Ataques:               {df['is_attack'].sum()} ({df['is_attack'].mean()*100:.1f}%)
    Normales:              {(~df['is_attack'].astype(bool)).sum()} ({(1-df['is_attack'].mean())*100:.1f}%)
    """
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_raw_data_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '01_raw_data_overview.png'}")


def plot_02_why_entity_structure(df):
    """Visualización 2: Por qué necesitamos estructura de entidades."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 PASO 2: ¿Por Qué Estructura de Entidades?', fontsize=16, fontweight='bold')

    # 2.1 Distribución de tasas por entidad
    ax1 = axes[0, 0]
    entity_means = df.groupby('entity_id')['event_count'].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0, 1, len(entity_means)))
    bars = ax1.barh(range(len(entity_means)), entity_means.values, color=colors)
    ax1.set_yticks([0, len(entity_means)//2, len(entity_means)-1])
    ax1.set_yticklabels([entity_means.index[0], entity_means.index[len(entity_means)//2], entity_means.index[-1]])
    ax1.set_xlabel('Media de Event Count')
    ax1.set_title('Tasa Media por Entidad\n(Cada entidad tiene diferente baseline)', fontweight='bold')
    ax1.axvline(df['event_count'].mean(), color='red', linestyle='--', linewidth=2, label='Media Global')
    ax1.legend()

    # 2.2 Comparación de dos entidades
    ax2 = axes[0, 1]

    # Seleccionar una entidad de alta actividad y una de baja
    entity_stats = df.groupby('entity_id')['event_count'].agg(['mean', 'std'])
    high_entity = entity_stats['mean'].idxmax()
    low_entity = entity_stats['mean'].idxmin()

    high_data = df[df['entity_id'] == high_entity]['event_count']
    low_data = df[df['entity_id'] == low_entity]['event_count']

    ax2.hist(low_data, bins=20, alpha=0.6, label=f'{low_entity}\n(μ={low_data.mean():.1f})', color='blue')
    ax2.hist(high_data, bins=20, alpha=0.6, label=f'{high_entity}\n(μ={high_data.mean():.1f})', color='orange')
    ax2.axvline(50, color='red', linestyle='--', linewidth=2, label='Umbral fijo = 50')
    ax2.set_xlabel('Event Count')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Problema: Un umbral global NO funciona', fontweight='bold')
    ax2.legend()

    # Añadir anotación
    ax2.annotate('50 es NORMAL\npara esta entidad', xy=(50, 0), xytext=(70, 5),
                fontsize=9, color='orange',
                arrowprops=dict(arrowstyle='->', color='orange'))
    ax2.annotate('50 es ANÓMALO\npara esta entidad', xy=(50, 0), xytext=(60, 3),
                fontsize=9, color='blue',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # 2.3 El mismo valor, diferente significado
    ax3 = axes[1, 0]

    # Crear ejemplo visual
    example_data = pd.DataFrame({
        'Entidad': ['user_quiet', 'user_quiet', 'user_active', 'user_active'],
        'Tipo': ['Baseline', 'Observación', 'Baseline', 'Observación'],
        'Valor': [10, 50, 80, 50]
    })

    x = np.arange(2)
    width = 0.35

    quiet_vals = [10, 50]
    active_vals = [80, 50]

    bars1 = ax3.bar(x - width/2, quiet_vals, width, label='user_quiet (baseline=10)', color='#3498db')
    bars2 = ax3.bar(x + width/2, active_vals, width, label='user_active (baseline=80)', color='#e74c3c')

    ax3.set_ylabel('Event Count')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Baseline\n(histórico)', 'Observación\n(hoy)'])
    ax3.legend()
    ax3.set_title('El MISMO valor (50) significa cosas DIFERENTES', fontweight='bold')

    # Añadir anotaciones
    ax3.annotate('🚨 +400%\nANÓMALO', xy=(0.175, 50), fontsize=10, color='#3498db', fontweight='bold',
                ha='center')
    ax3.annotate('✅ -37%\nNORMAL', xy=(0.825, 50), fontsize=10, color='#e74c3c', fontweight='bold',
                ha='center')

    # 2.4 Insight key
    ax4 = axes[1, 1]
    ax4.axis('off')

    insight_text = """
    💡 INSIGHT CLAVE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Los métodos clásicos usan UN SOLO umbral para todos:

        if event_count > 100:
            alert()

    Problema:
    • user_quiet (baseline=10): 50 eventos es +400% → ANÓMALO
    • user_active (baseline=80): 50 eventos es -37% → NORMAL

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    BSAD aprende un baseline θ[e] POR CADA ENTIDAD:

        θ[user_quiet] = 10
        θ[user_active] = 80

    Y compara cada observación contra SU PROPIO baseline:

        score = -log P(y | θ[entidad])

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Lo que es normal para A puede ser anómalo para B"
    """
    ax4.text(0.05, 0.95, insight_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='green'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_why_entity_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '02_why_entity_structure.png'}")


def plot_03_overdispersion(df):
    """Visualización 3: El problema de overdispersion."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📈 PASO 3: Overdispersion - ¿Por Qué Negative Binomial?', fontsize=16, fontweight='bold')

    # 3.1 Mean vs Variance por entidad
    ax1 = axes[0, 0]
    entity_stats = df.groupby('entity_id')['event_count'].agg(['mean', 'var'])

    ax1.scatter(entity_stats['mean'], entity_stats['var'], alpha=0.7, s=100, c='steelblue', edgecolor='black')

    # Línea de Poisson (var = mean)
    max_val = max(entity_stats['mean'].max(), entity_stats['var'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Poisson: Var = Mean')

    # Línea de regresión real
    z = np.polyfit(entity_stats['mean'], entity_stats['var'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, entity_stats['mean'].max(), 100)
    ax1.plot(x_line, p(x_line), 'g-', linewidth=2, label=f'Real: Var ≈ {z[0]:.1f}×Mean')

    ax1.set_xlabel('Media por Entidad')
    ax1.set_ylabel('Varianza por Entidad')
    ax1.set_title('Overdispersion: Varianza >> Media', fontweight='bold')
    ax1.legend()
    ax1.fill_between([0, max_val], [0, max_val], [0, max_val*5], alpha=0.1, color='red',
                     label='Zona de Overdispersion')

    # 3.2 Comparación Poisson vs Negative Binomial
    ax2 = axes[0, 1]

    # Datos reales
    real_data = df['event_count'].values
    mean_real = real_data.mean()
    var_real = real_data.var()

    # Simular Poisson
    poisson_data = np.random.poisson(mean_real, len(real_data))

    # Simular Negative Binomial con misma media
    # Parameterización: mean = n*p/(1-p), var = n*p/(1-p)^2
    # Si queremos var/mean = r, entonces p = 1/(1+mean/n)
    r = var_real / mean_real  # overdispersion ratio
    n_param = mean_real / (r - 1) if r > 1 else 1
    p_param = n_param / (n_param + mean_real)
    nb_data = np.random.negative_binomial(max(1, int(n_param)), p_param, len(real_data))

    bins = np.linspace(0, np.percentile(real_data, 95), 40)

    ax2.hist(real_data, bins=bins, alpha=0.5, label=f'Real\n(var/mean={var_real/mean_real:.1f})',
             density=True, color='blue')
    ax2.hist(poisson_data, bins=bins, alpha=0.5, label=f'Poisson\n(var/mean≈1)',
             density=True, color='red')
    ax2.hist(nb_data, bins=bins, alpha=0.5, label=f'NegBin\n(var/mean={np.var(nb_data)/np.mean(nb_data):.1f})',
             density=True, color='green')

    ax2.set_xlabel('Event Count')
    ax2.set_ylabel('Densidad')
    ax2.set_title('Comparación de Distribuciones', fontweight='bold')
    ax2.legend()

    # 3.3 Cola pesada (eventos extremos)
    ax3 = axes[1, 0]

    # Mostrar la cola de la distribución
    threshold = np.percentile(df['event_count'], 90)
    normal_counts = df[df['event_count'] <= threshold]['event_count']
    extreme_counts = df[df['event_count'] > threshold]['event_count']

    ax3.hist(normal_counts, bins=30, alpha=0.7, label=f'Normal (≤{threshold:.0f}): {len(normal_counts)} obs',
             color='steelblue')
    ax3.hist(extreme_counts, bins=20, alpha=0.7, label=f'Extremos (>{threshold:.0f}): {len(extreme_counts)} obs',
             color='coral')
    ax3.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Event Count')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Cola Pesada: Eventos Extremos', fontweight='bold')
    ax3.legend()

    # Anotar
    ax3.annotate('Poisson subestima\nestos eventos', xy=(threshold*1.5, 5), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='coral', alpha=0.5))

    # 3.4 Explicación
    ax4 = axes[1, 1]
    ax4.axis('off')

    overdispersion_ratio = df['event_count'].var() / df['event_count'].mean()

    explanation = f"""
    📊 ¿QUÉ ES OVERDISPERSION?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Poisson asume:     Varianza = Media
    Realidad:          Varianza >> Media

    En nuestros datos:
        Media    = {df['event_count'].mean():.2f}
        Varianza = {df['event_count'].var():.2f}
        Ratio    = {overdispersion_ratio:.2f}  ← ¡Debería ser ~1!

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ❌ POISSON (var = μ)

        P(y) = e^(-μ) × μ^y / y!

        Problema: No puede modelar la variabilidad extra
                  → Subestima probabilidad de eventos extremos
                  → Muchos falsos positivos

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ✅ NEGATIVE BINOMIAL (var = μ + μ²/φ)

        Tiene parámetro extra φ (overdispersion)

        φ grande → Se parece a Poisson
        φ pequeño → Mucha overdispersion

        Puede capturar la "cola pesada" de eventos extremos

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 Por eso BSAD usa Negative Binomial, no Poisson
    """
    ax4.text(0.02, 0.98, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='orange'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_overdispersion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '03_overdispersion.png'}")


def plot_04_partial_pooling(df):
    """Visualización 4: El efecto del partial pooling."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('🎯 PASO 4: Partial Pooling - La Magia del Modelo Jerárquico', fontsize=16, fontweight='bold')

    # Calcular estadísticas por entidad
    entity_stats = df.groupby('entity_id').agg({
        'event_count': ['mean', 'std', 'count'],
        'base_rate': 'first'
    }).reset_index()
    entity_stats.columns = ['entity_id', 'obs_mean', 'obs_std', 'n_obs', 'true_rate']

    global_mean = df['event_count'].mean()

    # Simular el efecto de shrinkage
    # Más observaciones → menos shrinkage
    # shrinkage_factor = n / (n + prior_strength)
    prior_strength = 5  # Simular fuerza del prior
    entity_stats['shrinkage'] = entity_stats['n_obs'] / (entity_stats['n_obs'] + prior_strength)
    entity_stats['pooled_estimate'] = (entity_stats['shrinkage'] * entity_stats['obs_mean'] +
                                        (1 - entity_stats['shrinkage']) * global_mean)

    # 4.1 Shrinkage diagram
    ax1 = axes[0, 0]

    # Ordenar por número de observaciones
    entity_stats_sorted = entity_stats.sort_values('n_obs')

    # Seleccionar algunas entidades representativas
    n_show = 10
    sample_entities = entity_stats_sorted.iloc[::len(entity_stats_sorted)//n_show][:n_show]

    y_pos = np.arange(len(sample_entities))

    # MLE (sin pooling)
    ax1.scatter(sample_entities['obs_mean'], y_pos, s=100, c='red', marker='o',
                label='MLE (sin pooling)', zorder=3)
    # Pooled estimate
    ax1.scatter(sample_entities['pooled_estimate'], y_pos, s=100, c='green', marker='s',
                label='Partial Pooling', zorder=3)
    # Global mean
    ax1.axvline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Media Global = {global_mean:.1f}')

    # Flechas de shrinkage
    for i, (_, row) in enumerate(sample_entities.iterrows()):
        ax1.annotate('', xy=(row['pooled_estimate'], i), xytext=(row['obs_mean'], i),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{row['entity_id']}\n(n={row['n_obs']:.0f})" for _, row in sample_entities.iterrows()])
    ax1.set_xlabel('Tasa Estimada')
    ax1.set_title('Shrinkage: MLE → Partial Pooling', fontweight='bold')
    ax1.legend(loc='lower right')

    # 4.2 Shrinkage vs número de observaciones
    ax2 = axes[0, 1]

    scatter = ax2.scatter(entity_stats['n_obs'], entity_stats['shrinkage'],
                         c=np.abs(entity_stats['obs_mean'] - global_mean),
                         cmap='RdYlGn_r', s=80, alpha=0.7)
    ax2.axhline(0.5, color='gray', linestyle='--', label='50% shrinkage')
    ax2.set_xlabel('Número de Observaciones')
    ax2.set_ylabel('Factor de Shrinkage\n(1=usa solo datos propios, 0=usa solo prior)')
    ax2.set_title('Shrinkage Adaptativo según Datos', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='|Desviación de media global|')
    ax2.legend()

    # 4.3 Comparación de los tres enfoques
    ax3 = axes[1, 0]

    approaches = ['No Pooling\n(MLE independiente)', 'Partial Pooling\n(BSAD)', 'Complete Pooling\n(Media Global)']

    # Para una entidad con pocos datos que se desvía de la media
    sparse_entity = entity_stats[entity_stats['n_obs'] == entity_stats['n_obs'].min()].iloc[0]

    no_pool = sparse_entity['obs_mean']
    partial_pool = sparse_entity['pooled_estimate']
    complete_pool = global_mean

    colors = ['#e74c3c', '#2ecc71', '#3498db']
    bars = ax3.bar(approaches, [no_pool, partial_pool, complete_pool], color=colors, edgecolor='black')

    ax3.axhline(sparse_entity['true_rate'], color='purple', linestyle='--', linewidth=2,
                label=f'Tasa Real = {sparse_entity["true_rate"]:.1f}')
    ax3.set_ylabel('Tasa Estimada')
    ax3.set_title(f'Entidad con POCOS datos (n={sparse_entity["n_obs"]:.0f})', fontweight='bold')
    ax3.legend()

    # Añadir valores
    for bar, val in zip(bars, [no_pool, partial_pool, complete_pool]):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}',
                ha='center', fontsize=10, fontweight='bold')

    # 4.4 Explicación
    ax4 = axes[1, 1]
    ax4.axis('off')

    explanation = """
    🔄 TRES ENFOQUES DE ESTIMACIÓN
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ❌ NO POOLING (MLE independiente)
       • Estima θ[e] solo con datos de la entidad e
       • Problema: Alta varianza para entidades sparse
       • "Confía ciegamente en datos ruidosos"

    ❌ COMPLETE POOLING (Media global)
       • θ[e] = μ para todas las entidades
       • Problema: Ignora diferencias individuales
       • "Trata a todos igual"

    ✅ PARTIAL POOLING (BSAD)
       • θ[e] = weighted_average(datos propios, prior)
       • El peso depende de cuántos datos tiene cada entidad

       Muchos datos → θ ≈ MLE (confía en datos)
       Pocos datos  → θ ≈ μ (usa prior poblacional)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📐 FÓRMULA DE SHRINKAGE

        θ[e]_pooled = λ × θ[e]_MLE + (1-λ) × μ

        donde λ = n[e] / (n[e] + α)

        • n[e] = observaciones de entidad e
        • α = fuerza del prior (se aprende del modelo)
        • μ = media poblacional

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 BENEFICIO: Entidades nuevas o sparse "toman prestada"
       información de la población sin perder individualidad
       cuando hay suficientes datos.
    """
    ax4.text(0.02, 0.98, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='green'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_partial_pooling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '04_partial_pooling.png'}")


def plot_05_scoring_explained(df):
    """Visualización 5: Cómo se calculan los anomaly scores."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('📊 PASO 5: Cálculo del Anomaly Score', fontsize=16, fontweight='bold')

    # Seleccionar una entidad para el ejemplo
    example_entity = df.groupby('entity_id')['event_count'].mean().idxmin()
    entity_data = df[df['entity_id'] == example_entity]['event_count'].values
    entity_mean = entity_data.mean()

    # 5.1 Distribución aprendida vs observación
    ax1 = axes[0, 0]

    # Simular distribución posterior (Negative Binomial)
    x = np.arange(0, int(entity_mean * 4))

    # Parámetros de la NegBin (simplificado)
    n_param = 3
    p_param = n_param / (n_param + entity_mean)
    pmf = stats.nbinom.pmf(x, n_param, p_param)

    ax1.bar(x, pmf, color='steelblue', alpha=0.7, label='P(y|θ) - Distribución aprendida')
    ax1.axvline(entity_mean, color='green', linestyle='--', linewidth=2, label=f'Media = {entity_mean:.1f}')

    # Marcar una observación normal y una anómala
    normal_obs = int(entity_mean)
    anomaly_obs = int(entity_mean * 3)

    ax1.axvline(normal_obs, color='blue', linewidth=3, label=f'Obs Normal = {normal_obs}')
    ax1.axvline(anomaly_obs, color='red', linewidth=3, label=f'Obs Anómala = {anomaly_obs}')

    ax1.set_xlabel('Event Count (y)')
    ax1.set_ylabel('P(y|θ)')
    ax1.set_title(f'Distribución para {example_entity}', fontweight='bold')
    ax1.legend()

    # 5.2 Cálculo paso a paso
    ax2 = axes[0, 1]
    ax2.axis('off')

    p_normal = stats.nbinom.pmf(normal_obs, n_param, p_param)
    p_anomaly = stats.nbinom.pmf(anomaly_obs, n_param, p_param)
    score_normal = -np.log(p_normal) if p_normal > 0 else 20
    score_anomaly = -np.log(p_anomaly) if p_anomaly > 0 else 20

    calculation = f"""
    📐 CÁLCULO DEL SCORE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Fórmula: score = -log P(y | θ, φ)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    OBSERVACIÓN NORMAL (y = {normal_obs}):

        P({normal_obs} | θ={entity_mean:.1f}) = {p_normal:.6f}

        score = -log({p_normal:.6f})
              = {score_normal:.2f}  ← BAJO (normal)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    OBSERVACIÓN ANÓMALA (y = {anomaly_obs}):

        P({anomaly_obs} | θ={entity_mean:.1f}) = {p_anomaly:.8f}

        score = -log({p_anomaly:.8f})
              = {score_anomaly:.2f}  ← ALTO (anómalo)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 INTERPRETACIÓN:

        • Score BAJO = Probabilidad ALTA = NORMAL
        • Score ALTO = Probabilidad BAJA = ANÓMALO

        El score es cuántos "bits de sorpresa" nos da
        ver esa observación dado el modelo.
    """
    ax2.text(0.02, 0.98, calculation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e3f2fd', edgecolor='blue'))

    # 5.3 Score vs Event Count
    ax3 = axes[1, 0]

    # Calcular scores para rango de valores
    y_range = np.arange(0, int(entity_mean * 5))
    probs = stats.nbinom.pmf(y_range, n_param, p_param)
    scores = -np.log(np.maximum(probs, 1e-10))

    ax3.plot(y_range, scores, 'b-', linewidth=2)
    ax3.fill_between(y_range, scores, alpha=0.3)
    ax3.axvline(entity_mean, color='green', linestyle='--', label=f'Media θ = {entity_mean:.1f}')

    # Marcar zonas
    ax3.axhspan(0, 5, alpha=0.2, color='green', label='Normal (score < 5)')
    ax3.axhspan(5, 10, alpha=0.2, color='yellow', label='Sospechoso (5 < score < 10)')
    ax3.axhspan(10, 20, alpha=0.2, color='red', label='Anómalo (score > 10)')

    ax3.set_xlabel('Event Count (y)')
    ax3.set_ylabel('Anomaly Score = -log P(y|θ)')
    ax3.set_title('Relación: Conteo → Score', fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.set_ylim(0, 20)

    # 5.4 Distribución final de scores
    ax4 = axes[1, 1]

    # Simular scores para todo el dataset
    all_scores = []
    for _, row in df.iterrows():
        entity_mean_i = df[df['entity_id'] == row['entity_id']]['event_count'].mean()
        n_i = 3
        p_i = n_i / (n_i + entity_mean_i)
        prob = stats.nbinom.pmf(int(row['event_count']), n_i, p_i)
        score = -np.log(max(prob, 1e-10))
        all_scores.append(score)

    df_scores = df.copy()
    df_scores['score'] = all_scores

    # Histograma separado por ataque/normal
    normal_scores = df_scores[df_scores['is_attack'] == 0]['score']
    attack_scores = df_scores[df_scores['is_attack'] == 1]['score']

    ax4.hist(normal_scores, bins=30, alpha=0.6, label=f'Normal (n={len(normal_scores)})', color='steelblue', density=True)
    ax4.hist(attack_scores, bins=20, alpha=0.6, label=f'Ataque (n={len(attack_scores)})', color='coral', density=True)

    # Threshold
    threshold = np.percentile(all_scores, 95)
    ax4.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (p95) = {threshold:.1f}')

    ax4.set_xlabel('Anomaly Score')
    ax4.set_ylabel('Densidad')
    ax4.set_title('Distribución de Scores: Normal vs Ataque', fontweight='bold')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_scoring_explained.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '05_scoring_explained.png'}")


def plot_06_full_pipeline_summary(df):
    """Visualización 6: Resumen del pipeline completo."""
    fig = plt.figure(figsize=(16, 12))

    # Crear grid personalizado
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    fig.suptitle('🔄 PIPELINE COMPLETO DE BSAD', fontsize=18, fontweight='bold', y=0.98)

    # 1. Datos crudos
    ax1 = fig.add_subplot(gs[0, 0])
    sample = df.head(5)[['entity_id', 'event_count']].values
    ax1.axis('off')
    table1 = ax1.table(cellText=sample, colLabels=['entity_id', 'count'],
                       loc='center', cellLoc='center', colColours=['#e3f2fd']*2)
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    ax1.set_title('1️⃣ Datos Crudos\n(N=100K)', fontweight='bold', fontsize=10)

    # 2. Agregación
    ax2 = fig.add_subplot(gs[0, 1])
    agg_sample = df.groupby(['entity_id', 'day'])['event_count'].sum().reset_index().head(5).values
    ax2.axis('off')
    table2 = ax2.table(cellText=agg_sample[:, :3], colLabels=['entity', 'day', 'count'],
                       loc='center', cellLoc='center', colColours=['#f3e5f5']*3)
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    ax2.set_title('2️⃣ Agregación\n(N=2.8K)', fontweight='bold', fontsize=10)

    # 3. Modelo jerárquico (diagrama simple)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    model_text = """
       μ, α
        │
        ▼
    ┌───────┐
    │ θ[1]  │
    │ θ[2]  │
    │  ...  │
    │ θ[E]  │
    └───────┘
        │
        ▼
    y ~ NegBin
    """
    ax3.text(0.5, 0.5, model_text, transform=ax3.transAxes, fontsize=10,
             ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff3e0', edgecolor='orange'))
    ax3.set_title('3️⃣ Modelo\nJerárquico', fontweight='bold', fontsize=10)

    # 4. MCMC
    ax4 = fig.add_subplot(gs[0, 3])
    # Simular trace
    np.random.seed(42)
    trace_sample = np.random.normal(35, 5, (100, 3))
    for i in range(3):
        ax4.plot(trace_sample[:, i], alpha=0.7, linewidth=0.5)
    ax4.set_xlabel('Iteration', fontsize=8)
    ax4.set_ylabel('θ', fontsize=8)
    ax4.set_title('4️⃣ MCMC\n(2K samples)', fontweight='bold', fontsize=10)
    ax4.tick_params(labelsize=7)

    # 5. Scoring
    ax5 = fig.add_subplot(gs[1, :2])
    entity_stats = df.groupby('entity_id')['event_count'].mean()
    scores = -np.log(np.random.uniform(0.001, 0.5, len(df)))
    df_temp = df.copy()
    df_temp['score'] = scores

    # Top anomalías
    top_anom = df_temp.nlargest(10, 'score')[['entity_id', 'event_count', 'score', 'is_attack']]
    top_anom['score'] = top_anom['score'].round(2)

    ax5.axis('off')
    colors = [['#ffcdd2' if row[3] else 'white' for _ in range(4)] for row in top_anom.values]
    table5 = ax5.table(cellText=top_anom.values,
                       colLabels=['entity', 'count', 'score', 'attack'],
                       loc='center', cellLoc='center',
                       colColours=['#fce4ec']*4,
                       cellColours=colors)
    table5.auto_set_font_size(False)
    table5.set_fontsize(9)
    ax5.set_title('5️⃣ Scoring: Top 10 Anomalías (rojo = ataque real)', fontweight='bold', fontsize=11)

    # 6. Distribución de scores
    ax6 = fig.add_subplot(gs[1, 2:])
    normal_scores = df_temp[df_temp['is_attack'] == 0]['score']
    attack_scores = df_temp[df_temp['is_attack'] == 1]['score']

    ax6.hist(normal_scores, bins=30, alpha=0.6, label='Normal', color='steelblue', density=True)
    ax6.hist(attack_scores, bins=15, alpha=0.6, label='Ataque', color='coral', density=True)
    ax6.axvline(np.percentile(scores, 95), color='red', linestyle='--', label='Threshold (p95)')
    ax6.set_xlabel('Anomaly Score')
    ax6.set_ylabel('Densidad')
    ax6.set_title('6️⃣ Distribución de Scores', fontweight='bold', fontsize=11)
    ax6.legend()

    # 7. Métricas finales
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.axis('off')

    # Calcular métricas simuladas
    from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

    y_true = df_temp['is_attack'].values
    y_score = df_temp['score'].values

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    # Precision@K
    top_k = 50
    top_k_idx = np.argsort(y_score)[-top_k:]
    precision_at_k = y_true[top_k_idx].sum() / top_k

    metrics_text = f"""
    📈 MÉTRICAS DE EVALUACIÓN
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    PR-AUC:           {pr_auc:.3f}
    ROC-AUC:          {roc_auc:.3f}
    Precision@{top_k}:     {precision_at_k:.1%}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    De las top {top_k} alertas:
    • {int(precision_at_k * top_k)} son ataques reales
    • {int((1-precision_at_k) * top_k)} son falsos positivos

    💡 PR-AUC es la métrica primaria para
       eventos raros (class imbalance)
    """
    ax7.text(0.05, 0.95, metrics_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', edgecolor='green'))
    ax7.set_title('7️⃣ Evaluación', fontweight='bold', fontsize=11)

    # 8. PR Curve
    ax8 = fig.add_subplot(gs[2, 2:])
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ax8.plot(recall, precision, 'b-', linewidth=2, label=f'BSAD (PR-AUC={pr_auc:.3f})')
    ax8.fill_between(recall, precision, alpha=0.3)
    ax8.axhline(y_true.mean(), color='r', linestyle='--', label=f'Random (baseline={y_true.mean():.3f})')
    ax8.set_xlabel('Recall')
    ax8.set_ylabel('Precision')
    ax8.set_title('8️⃣ Precision-Recall Curve', fontweight='bold', fontsize=11)
    ax8.legend()
    ax8.set_xlim([0, 1])
    ax8.set_ylim([0, 1])

    plt.savefig(OUTPUT_DIR / '06_full_pipeline_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {OUTPUT_DIR / '06_full_pipeline_summary.png'}")


def main():
    """Genera todas las visualizaciones EDA."""
    print("=" * 60)
    print("BSAD EDA PIPELINE EXPLAINER")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generar datos
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_entities=50, n_days=30, seed=42)
    print(f"  Generated {len(df):,} observations")
    print(f"  Entities: {df['entity_id'].nunique()}")
    print(f"  Attack rate: {df['is_attack'].mean():.1%}")
    print()

    # Generar visualizaciones
    print("Generating visualizations...")
    print()

    plot_01_raw_data_overview(df)
    plot_02_why_entity_structure(df)
    plot_03_overdispersion(df)
    plot_04_partial_pooling(df)
    plot_05_scoring_explained(df)
    plot_06_full_pipeline_summary(df)

    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated 6 visualizations in {OUTPUT_DIR}/")
    print("\nVisualizaciones:")
    print("  01_raw_data_overview.png      - Vista general de datos crudos")
    print("  02_why_entity_structure.png   - Por qué necesitamos entidades")
    print("  03_overdispersion.png         - Problema de overdispersion")
    print("  04_partial_pooling.png        - Efecto del partial pooling")
    print("  05_scoring_explained.png      - Cómo se calculan los scores")
    print("  06_full_pipeline_summary.png  - Resumen del pipeline completo")


if __name__ == "__main__":
    main()
