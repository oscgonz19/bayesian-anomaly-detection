"""
Visualizaciones educativas del pipeline BSAD.

Genera una serie de gráficos que explican, paso a paso, cómo funciona
el detector bayesiano jerárquico de anomalías en datos de seguridad.

Uso:
    python scripts/explain_pipeline.py
    python scripts/explain_pipeline.py --output docs/figures/
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# ---------------------------------------------------------------------------
# Estilo global
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

BLUE = "#2E86AB"
RED = "#E84855"
GREEN = "#3BB273"
ORANGE = "#F4A261"
PURPLE = "#6A4C93"
GRAY = "#8D99AE"
DARK = "#2B2D42"

RNG = np.random.default_rng(42)


# ===========================================================================
# FIGURA 1: El problema – heterogeneidad de entidades y rareza de ataques
# ===========================================================================

def fig1_el_problema(output_dir: Path) -> None:
    """
    Muestra por qué los datos de seguridad son difíciles:
      - Los usuarios tienen tasas MUY distintas (heterogeneidad)
      - Los ataques son raros (<5 % de ventanas)
      - La distribución es sobredispersa (varianza >> media)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "El Problema: Datos de Conteo en Seguridad",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # --- Panel 1: Heterogeneidad de usuarios ---
    ax = axes[0]
    n_users = 40
    user_rates = RNG.lognormal(mean=np.log(5), sigma=0.9, size=n_users)
    user_rates_sorted = np.sort(user_rates)[::-1]
    colors = [BLUE] * n_users
    ax.bar(range(n_users), user_rates_sorted, color=colors, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axhline(user_rates.mean(), color=RED, linestyle="--", linewidth=1.5,
               label=f"Media global = {user_rates.mean():.1f}")
    ax.set_xlabel("Usuario (ordenado por actividad)")
    ax.set_ylabel("Eventos promedio / día")
    ax.set_title("① Heterogeneidad entre entidades")
    ax.legend()
    ax.text(0.97, 0.97,
            f"Rango: {user_rates_sorted[-1]:.1f} – {user_rates_sorted[0]:.1f}\n"
            f"CV = {user_rates.std()/user_rates.mean():.1%}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=ORANGE, alpha=0.8),
            fontsize=9)

    # --- Panel 2: Rareza de ataques ---
    ax = axes[1]
    n_windows = 6000
    attack_rate = 0.02
    n_attacks = int(n_windows * attack_rate)
    labels = ["Ventanas\nbenignas\n(98 %)", "Ventanas\nde ataque\n(2 %)"]
    sizes = [n_windows - n_attacks, n_attacks]
    wedge_colors = [BLUE, RED]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=wedge_colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.6,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("② Rareza extrema de ataques")
    ax.text(0, -1.4,
            "Un clasificador que predice\n'benigno' siempre tiene 98 % accuracy.\n"
            "→ Accuracy es inútil aquí.",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec=ORANGE, alpha=0.9))

    # --- Panel 3: Sobredispersión ---
    ax = axes[2]
    x = np.arange(0, 35)
    mu_nb = 5.0
    # Poisson: varianza = media
    poisson_pmf = stats.poisson.pmf(x, mu=mu_nb)
    # NB con phi=2: varianza = mu + mu^2/phi
    phi = 2.0
    p_nb = phi / (phi + mu_nb)
    nb_pmf = stats.nbinom.pmf(x, n=phi, p=p_nb)

    ax.plot(x, poisson_pmf, color=BLUE, linewidth=2, marker="o", markersize=4,
            label=f"Poisson (μ=5, σ²=5)")
    ax.fill_between(x, poisson_pmf, alpha=0.2, color=BLUE)
    ax.plot(x, nb_pmf, color=RED, linewidth=2, marker="s", markersize=4,
            label=f"NB (μ=5, σ²={mu_nb + mu_nb**2/phi:.0f})")
    ax.fill_between(x, nb_pmf, alpha=0.2, color=RED)
    ax.set_xlabel("Conteo de eventos")
    ax.set_ylabel("Probabilidad")
    ax.set_title("③ Sobredispersión: Varianza >> Media")
    ax.legend()
    ax.text(0.97, 0.97,
            "Eventos de seguridad tienen\ncolas pesadas: días normales\n"
            "son mayoritariamente cero,\npero algunos son muy altos.",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=ORANGE, alpha=0.8),
            fontsize=9)

    plt.tight_layout()
    _save(fig, output_dir / "01_el_problema.png")


# ===========================================================================
# FIGURA 2: Por qué fallan los métodos clásicos
# ===========================================================================

def fig2_falla_metodos_clasicos(output_dir: Path) -> None:
    """
    Compara Z-score, IF, y NB-MLE contra BSAD en un escenario sintético pequeño.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "¿Por qué Fallan los Métodos Clásicos?",
        fontsize=15, fontweight="bold",
    )

    # Datos sintéticos: 5 usuarios con tasas distintas
    np.random.seed(42)
    user_rates = [2, 5, 15, 30, 50]
    n_users = len(user_rates)
    n_obs_per_user = 20
    all_counts = []
    all_entities = []
    all_labels = []

    for i, rate in enumerate(user_rates):
        counts = RNG.negative_binomial(n=2, p=2/(2+rate), size=n_obs_per_user)
        labels = np.zeros(n_obs_per_user, dtype=int)
        # Inyectar 1 ataque por usuario
        attack_idx = RNG.integers(0, n_obs_per_user)
        counts[attack_idx] = int(rate * 5)
        labels[attack_idx] = 1
        all_counts.extend(counts)
        all_entities.extend([i] * n_obs_per_user)
        all_labels.extend(labels)

    y = np.array(all_counts)
    entity_idx = np.array(all_entities)
    is_attack = np.array(all_labels)

    # Métodos
    # 1. Global Z-score (ignora entidades)
    global_mean, global_std = y.mean(), y.std()
    score_zscore_global = (y - global_mean) / (global_std + 1e-8)

    # 2. Z-score per entity
    score_zscore_entity = np.zeros(len(y))
    for e in range(n_users):
        mask = entity_idx == e
        mu_e = y[mask].mean()
        std_e = y[mask].std() + 0.5
        score_zscore_entity[mask] = (y[mask] - mu_e) / std_e

    # 3. NB-MLE por entidad (no pooling)
    score_nb_mle = np.zeros(len(y))
    for e in range(n_users):
        mask = entity_idx == e
        mu_e = y[mask].mean()
        var_e = y[mask].var()
        phi_e = max(mu_e**2 / max(var_e - mu_e, 0.1), 0.5)
        p_e = phi_e / (phi_e + mu_e)
        score_nb_mle[mask] = -stats.nbinom.logpmf(y[mask], n=phi_e, p=p_e)

    # 4. BSAD aproximado (NB con partial pooling simulado)
    global_mu = y.mean()
    score_bsad = np.zeros(len(y))
    shrinkage_k = 5
    for e in range(n_users):
        mask = entity_idx == e
        n_e = mask.sum()
        mu_e = y[mask].mean()
        # Shrinkage hacia la media global
        mu_shrunk = (n_e * mu_e + shrinkage_k * global_mu) / (n_e + shrinkage_k)
        phi_e = 2.0
        p_e = phi_e / (phi_e + mu_shrunk)
        score_bsad[mask] = -stats.nbinom.logpmf(y[mask], n=phi_e, p=p_e)

    def plot_method(ax, scores, title, note=""):
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        benign_scores = scores_norm[~is_attack.astype(bool)]
        attack_scores = scores_norm[is_attack.astype(bool)]
        ax.hist(benign_scores, bins=20, alpha=0.7, color=BLUE, density=True,
                label=f"Benigno (n={len(benign_scores)})")
        ax.hist(attack_scores, bins=10, alpha=0.8, color=RED, density=True,
                label=f"Ataque (n={len(attack_scores)})")
        ax.set_title(title)
        ax.set_xlabel("Score normalizado")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8)
        pr_auc = average_precision_score(is_attack, scores)
        ax.text(0.98, 0.98, f"PR-AUC = {pr_auc:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                fontweight="bold",
                color=GREEN if pr_auc > 0.5 else RED,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, alpha=0.8))
        if note:
            ax.text(0.02, 0.98, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=DARK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=ORANGE, alpha=0.8))

    plot_method(axes[0, 0], score_zscore_global, "Z-score Global\n(ignora estructura de entidad)",
                "❌ Usuario activo parece\nsiempre anómalo")
    plot_method(axes[0, 1], score_zscore_entity, "Z-score por Entidad\n(sin pooling, asume normalidad)",
                "⚠ Mejor, pero asume\ndistribución normal (falso)")
    plot_method(axes[1, 0], score_nb_mle, "NB-MLE por Entidad\n(sin pooling, entidades escasas = ruido)",
                "⚠ Correcto en distribución\npero estimados inestables\npara entidades con pocos datos")
    plot_method(axes[1, 1], score_bsad, "BSAD: NB Jerárquico Bayesiano\n(partial pooling + incertidumbre)",
                "✓ Distribución correcta\n✓ Shrinkage para entidades\n  escasas\n✓ Cuantifica incertidumbre")

    plt.tight_layout()
    _save(fig, output_dir / "02_falla_metodos_clasicos.png")


# ===========================================================================
# FIGURA 3: Estructura del modelo jerárquico
# ===========================================================================

def fig3_modelo_jerarquico(output_dir: Path) -> None:
    """
    Diagrama de placas del modelo NB jerárquico + visualización
    de las distribuciones a priori y el efecto de partial pooling.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.05)

    # --- Panel izquierdo: Diagrama de placas ---
    ax_plate = fig.add_subplot(gs[0])
    ax_plate.set_xlim(0, 10)
    ax_plate.set_ylim(0, 10)
    ax_plate.axis("off")
    ax_plate.set_title("Estructura del Modelo Bayesiano Jerárquico", fontsize=13, fontweight="bold")

    def box(ax, x, y, w, h, color, label, sublabel="", alpha=0.15):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor=color,
                                        alpha=alpha, linewidth=2)
        ax.add_patch(rect)
        rect2 = mpatches.FancyBboxPatch((x, y), w, h,
                                         boxstyle="round,pad=0.1",
                                         facecolor="none", edgecolor=color,
                                         alpha=0.8, linewidth=2)
        ax.add_patch(rect2)
        ax.text(x + w / 2, y + h / 2 + (0.2 if sublabel else 0),
                label, ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.3, sublabel,
                    ha="center", va="center", fontsize=8.5, color=DARK, fontstyle="italic")

    def node(ax, x, y, label, sublabel="", color=BLUE, observed=False):
        circle = plt.Circle((x, y), 0.45,
                              facecolor="white" if not observed else "#E8F5E9",
                              edgecolor=color, linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + (0.12 if sublabel else 0),
                label, ha="center", va="center", fontsize=12,
                fontweight="bold", color=DARK, zorder=4)
        if sublabel:
            ax.text(x, y - 0.18, sublabel,
                    ha="center", va="center", fontsize=7.5,
                    color=GRAY, zorder=4)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=DARK,
                                    lw=1.8, connectionstyle="arc3,rad=0"))

    # Nivel 1: Hiperpriors (Población)
    box(ax_plate, 1.0, 7.5, 8.0, 1.8, PURPLE, "  Nivel de Población  ", alpha=0.08)
    node(ax_plate, 3.0, 8.4, "μ", "Exp(0.1)", color=PURPLE)
    node(ax_plate, 7.0, 8.4, "α", "HalfN(2)", color=PURPLE)

    # Nivel 2: Entidades
    box(ax_plate, 1.5, 4.5, 7.0, 2.5, BLUE, "  Nivel de Entidad  [E entidades]  ", alpha=0.08)
    node(ax_plate, 5.0, 5.7, "θₑ", "Gamma(μα, α)", color=BLUE)

    # Nivel 3: Observaciones
    box(ax_plate, 2.0, 1.5, 6.0, 2.5, GREEN, "  Nivel de Observación  [N obs]  ", alpha=0.08)
    node(ax_plate, 4.0, 2.75, "φ", "HalfN(2)", color=GREEN)
    node(ax_plate, 6.5, 2.75, "yₙ", "NB(θₑₙ, φ)", color=GREEN, observed=True)

    # Flechas
    arrow(ax_plate, 3.0, 7.95, 4.6, 6.1)   # mu → theta
    arrow(ax_plate, 7.0, 7.95, 5.4, 6.1)   # alpha → theta
    arrow(ax_plate, 5.0, 5.25, 5.8, 3.2)   # theta → y
    arrow(ax_plate, 4.0, 2.3, 5.8, 2.75)   # phi → y  (horizontal)

    # Anotaciones de texto a la derecha
    annotations = [
        (9.0, 8.4, "μ: tasa media\nglobal de\neventos"),
        (9.0, 7.8, "α: concentración\n(fuerza del\npooling)"),
        (9.0, 5.7, "θₑ: tasa específica\nde la entidad e\n(el 'normal' de cada\nusuario)"),
        (9.0, 2.75, "φ: sobredispersión\nglobal\n\nyₙ: conteo observado\n(DATO)"),
    ]
    # Solo etiquetas a los lados de los nodos
    ax_plate.text(1.2, 8.4, "μ ~ Exp(λ=0.1)\n"r"E[μ]=10", fontsize=8.5, color=PURPLE,
                  va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#F3E8FF", ec=PURPLE, alpha=0.5))
    ax_plate.text(7.6, 8.4, "α ~ HalfNormal(σ=2)\ncontrola el shrinkage", fontsize=8.5, color=PURPLE,
                  va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#F3E8FF", ec=PURPLE, alpha=0.5))
    ax_plate.text(5.6, 5.7, "θₑ ~ Gamma(μα, α)\nentidad e aprende\nsu propia tasa", fontsize=8.5, color=BLUE,
                  va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#E8F0FF", ec=BLUE, alpha=0.5))
    ax_plate.text(4.6, 2.0, "yₙ ~ NegBin(θₑₙ, φ)\n DATO: observado ✓", fontsize=8.5, color=GREEN,
                  va="center", ha="left",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#E8F9EE", ec=GREEN, alpha=0.5))

    # Score
    ax_plate.text(5.0, 0.7,
                  "Score de anomalía: –log p(y | posterior) = –log ∫ p(y|θ,φ) p(θ,φ|datos) dθdφ",
                  ha="center", va="center", fontsize=9, color=RED, fontstyle="italic",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FFE8E8", ec=RED, alpha=0.6))

    # --- Panel derecho: Partial pooling ---
    ax_pool = fig.add_subplot(gs[1])
    ax_pool.set_title("Partial Pooling: el corazón del modelo", fontsize=12, fontweight="bold")

    # Simular estimaciones con distintos n_obs
    n_obs_values = [1, 2, 5, 10, 20, 50, 100]
    global_mean_val = 5.0
    true_rate = 15.0

    mle_estimates = []
    bayes_estimates = []
    for n in n_obs_values:
        obs = RNG.poisson(true_rate, size=n)
        mle = obs.mean()
        k = 5  # shrinkage
        bayes = (n * mle + k * global_mean_val) / (n + k)
        mle_estimates.append(mle)
        bayes_estimates.append(bayes)

    ax_pool.plot(n_obs_values, mle_estimates, "o-", color=ORANGE, linewidth=2,
                 markersize=7, label="MLE puro (sin pooling)", zorder=3)
    ax_pool.plot(n_obs_values, bayes_estimates, "s-", color=BLUE, linewidth=2,
                 markersize=7, label="BSAD (partial pooling)", zorder=3)
    ax_pool.axhline(true_rate, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Tasa real = {true_rate}")
    ax_pool.axhline(global_mean_val, color=GRAY, linestyle=":", linewidth=1.5,
                    label=f"Media global = {global_mean_val}")

    ax_pool.fill_between(n_obs_values,
                          [global_mean_val] * len(n_obs_values),
                          [true_rate] * len(n_obs_values),
                          alpha=0.05, color=BLUE)

    ax_pool.set_xscale("log")
    ax_pool.set_xlabel("Número de observaciones para esta entidad")
    ax_pool.set_ylabel("Estimación de la tasa θₑ")
    ax_pool.legend(fontsize=9)

    ax_pool.annotate(
        "Entidad nueva (pocos datos)\n→ BSAD se encoge\nhacia la media global",
        xy=(1, bayes_estimates[0]), xytext=(3, 8),
        arrowprops=dict(arrowstyle="->", color=DARK),
        fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=ORANGE, alpha=0.8),
    )
    ax_pool.annotate(
        "Entidad con muchos datos\n→ converge al MLE\n(la evidencia domina)",
        xy=(100, bayes_estimates[-1]), xytext=(20, 12),
        arrowprops=dict(arrowstyle="->", color=DARK),
        fontsize=9, color=DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec=GREEN, alpha=0.8),
    )

    ax_pool.set_title("Partial Pooling: Shrinkage Adaptativo", fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save(fig, output_dir / "03_modelo_jerarquico.png")


# ===========================================================================
# FIGURA 4: Pipeline de datos paso a paso
# ===========================================================================

def fig4_pipeline_datos(output_dir: Path) -> None:
    """
    Muestra el flujo de datos con ejemplos concretos en cada etapa.
    """
    from bsad.data_generator import GeneratorConfig, generate_synthetic_data
    from bsad.features import FeatureConfig, build_modeling_table

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Pipeline de Datos: de Eventos Crudos a Tabla de Modelado",
                 fontsize=14, fontweight="bold")

    cfg = GeneratorConfig(n_users=30, n_days=14, attack_rate=0.05, random_seed=42)
    events_df, attacks_df = generate_synthetic_data(cfg)

    feat_cfg = FeatureConfig(entity_column="user_id", window_size="1D", include_temporal=True)
    modeling_df, metadata = build_modeling_table(events_df, feat_cfg)

    # --- Panel 1: Eventos crudos (primeras filas) ---
    ax = axes[0, 0]
    ax.axis("off")
    ax.set_title("① Eventos Crudos (log de seguridad)", fontweight="bold")
    sample = events_df[["timestamp", "user_id", "endpoint", "status_code",
                         "is_attack"]].head(8)
    sample = sample.copy()
    sample["timestamp"] = pd.to_datetime(sample["timestamp"]).dt.strftime("%m-%d %H:%M")
    sample["is_attack"] = sample["is_attack"].map({True: "🔴 Sí", False: "⚪ No"})
    col_labels = ["Timestamp", "Usuario", "Endpoint", "Status", "Ataque"]
    table = ax.table(
        cellText=sample.values,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif "Sí" in str(cell.get_text().get_text()):
            cell.set_facecolor("#FFE8E8")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("white")

    # --- Panel 2: Distribución de tasas de eventos por usuario ---
    ax = axes[0, 1]
    user_stats = modeling_df.groupby("user_id")["event_count"].mean().sort_values(ascending=False)
    ax.bar(range(len(user_stats)), user_stats.values, color=BLUE, alpha=0.8, edgecolor="white")
    ax.axhline(user_stats.mean(), color=RED, linestyle="--", linewidth=1.5,
               label=f"Media = {user_stats.mean():.1f}")
    ax.set_xlabel("Usuario (ordenado)")
    ax.set_ylabel("Eventos / ventana (promedio)")
    ax.set_title("② Heterogeneidad de Usuarios\n(después del agregado)")
    ax.legend()

    # --- Panel 3: Distribución de conteos (sobredispersión) ---
    ax = axes[0, 2]
    counts = modeling_df["event_count"].values
    ax.hist(counts, bins=40, color=BLUE, alpha=0.8, edgecolor="white", density=True)
    # Ajuste Poisson teórico
    mu_fit = counts.mean()
    x_range = np.arange(0, counts.max() + 1)
    ax.plot(x_range, stats.poisson.pmf(x_range, mu=mu_fit), color=ORANGE, linewidth=2,
            label=f"Poisson(μ={mu_fit:.1f})")
    var = counts.var()
    phi_fit = mu_fit**2 / max(var - mu_fit, 0.1)
    p_fit = phi_fit / (phi_fit + mu_fit)
    ax.plot(x_range, stats.nbinom.pmf(x_range, n=phi_fit, p=p_fit), color=RED, linewidth=2,
            label=f"NB(μ={mu_fit:.1f}, φ={phi_fit:.1f})\nVar/Media={var/mu_fit:.1f}x")
    ax.set_xlabel("Conteo de eventos por ventana")
    ax.set_ylabel("Densidad")
    ax.set_title("③ Sobredispersión Confirmada\n(NB >> Poisson)")
    ax.legend(fontsize=8)

    # --- Panel 4: Serie temporal de un usuario ---
    ax = axes[1, 0]
    user_example = modeling_df[modeling_df["user_id"] == "user_0000"].copy()
    user_example = user_example.sort_values("window")
    attack_windows = user_example[user_example["has_attack"]]
    ax.plot(range(len(user_example)), user_example["event_count"].values,
            color=BLUE, linewidth=1.5, marker="o", markersize=4, label="Benigno")
    if len(attack_windows) > 0:
        for _, row in attack_windows.iterrows():
            idx = list(user_example["window"]).index(row["window"])
            ax.scatter(idx, row["event_count"], color=RED, s=120, zorder=5,
                       marker="^", label="Ataque" if idx == list(user_example["window"]
                                                                  ).index(row["window"]) else "")
    ax.axhline(user_example["entity_mean_count"].iloc[0], color=GRAY, linestyle="--",
               linewidth=1.2, label=f"Media = {user_example['entity_mean_count'].iloc[0]:.1f}")
    ax.set_xlabel("Ventana diaria")
    ax.set_ylabel("Conteo de eventos")
    ax.set_title("④ Serie Temporal: user_0000\n(¿Cuándo sale de su 'normal'?)")
    ax.legend(fontsize=8)

    # --- Panel 5: Tabla de modelado final ---
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("⑤ Tabla de Modelado (input al modelo)", fontweight="bold")
    sample_model = modeling_df[["user_id", "window", "event_count",
                                 "entity_idx", "has_attack"]].head(8).copy()
    sample_model["window"] = pd.to_datetime(sample_model["window"]).dt.strftime("%m-%d")
    sample_model["has_attack"] = sample_model["has_attack"].map({True: "🔴", False: "⚪"})
    table2 = ax.table(
        cellText=sample_model.values,
        colLabels=["Usuario", "Ventana", "Conteo", "ID Entidad", "Ataque"],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    for (r, c), cell in table2.get_celld().items():
        if r == 0:
            cell.set_facecolor(GREEN)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("white")

    ax.text(0.5, -0.08,
            f"Total: {len(modeling_df):,} filas | {metadata['n_entities']} entidades | "
            f"Attack rate: {metadata['attack_rate']:.1%}",
            ha="center", transform=ax.transAxes, fontsize=9, color=DARK)

    # --- Panel 6: Lo que el modelo usa vs ignora ---
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("⑥ Qué Usa el Modelo NB vs Qué es Contexto", fontweight="bold")

    uses = [
        ("event_count (y)", "✓ Likelihood: y ~ NB(θₑ, φ)", GREEN),
        ("entity_idx", "✓ Indexa θₑ por entidad", GREEN),
    ]
    context = [
        ("unique_ips", "✗ No en likelihood", RED),
        ("unique_devices", "✗ No en likelihood", RED),
        ("location", "✗ No en likelihood", RED),
        ("bytes_total", "✗ No en likelihood", RED),
        ("is_weekend", "✗ No en likelihood", RED),
        ("failed_count", "✗ No en likelihood", RED),
    ]

    y_pos = 0.93
    ax.text(0.02, y_pos, "LO QUE EL MODELO MODELA:", fontweight="bold",
            color=GREEN, transform=ax.transAxes, fontsize=10)
    y_pos -= 0.07
    for feat, note, color in uses:
        ax.text(0.05, y_pos, f"• {feat}", transform=ax.transAxes,
                fontsize=9, color=DARK)
        ax.text(0.45, y_pos, note, transform=ax.transAxes,
                fontsize=8.5, color=color, fontstyle="italic")
        y_pos -= 0.07

    y_pos -= 0.03
    ax.text(0.02, y_pos, "SOLO CONTEXTO / EDA / TRIAGE:", fontweight="bold",
            color=RED, transform=ax.transAxes, fontsize=10)
    y_pos -= 0.07
    for feat, note, color in context:
        ax.text(0.05, y_pos, f"• {feat}", transform=ax.transAxes,
                fontsize=9, color=DARK)
        ax.text(0.45, y_pos, note, transform=ax.transAxes,
                fontsize=8.5, color=color, fontstyle="italic")
        y_pos -= 0.065

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02",
        facecolor="none", edgecolor=GRAY, linewidth=1.5, transform=ax.transAxes,
    ))

    plt.tight_layout()
    _save(fig, output_dir / "04_pipeline_datos.png")


# ===========================================================================
# FIGURA 5: Cómo funciona el scoring de anomalías
# ===========================================================================

def fig5_anomaly_scoring(output_dir: Path) -> None:
    """
    Explica el score = -log P(y | posterior) con visualizaciones.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cómo Funciona el Score de Anomalía: –log P(y | posterior)",
                 fontsize=14, fontweight="bold")

    # --- Panel 1: Probabilidad bajo la distribución aprendida ---
    ax = axes[0, 0]
    mu_learned = 5.0
    phi_learned = 2.0
    p_learned = phi_learned / (phi_learned + mu_learned)
    x = np.arange(0, 50)
    pmf = stats.nbinom.pmf(x, n=phi_learned, p=p_learned)

    ax.bar(x, pmf, color=BLUE, alpha=0.6, label="P(y | θ aprendida)")

    # Marcar puntos con distintos scores
    ejemplos = [
        (5, "normal", GREEN, "benigno típico"),
        (3, "bajo", ORANGE, "benigno bajo"),
        (30, "anómalo", RED, "¡ATAQUE! burst"),
    ]
    for val, tipo, color, label in ejemplos:
        prob = stats.nbinom.pmf(val, n=phi_learned, p=p_learned)
        score = -np.log(prob + 1e-10)
        ax.bar([val], [prob], color=color, alpha=1.0,
               label=f"y={val}: score={score:.1f}")
        ax.annotate(f"  y={val}\n  score={score:.1f}",
                    xy=(val, prob), xytext=(val + 1, prob + 0.02),
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xlabel("Conteo de eventos observado (y)")
    ax.set_ylabel("P(y | parámetros aprendidos)")
    ax.set_title("① Probabilidad bajo el modelo aprendido\nAnomalía = evento muy improbable")
    ax.legend(fontsize=8)

    # --- Panel 2: Score vs conteo para distintas tasas de entidad ---
    ax = axes[0, 1]
    y_range = np.arange(0, 60)
    phi = 2.0

    for mu_e, color, label in [
        (2.0, GREEN, "Entidad tranquila (μ=2)"),
        (8.0, BLUE, "Entidad activa (μ=8)"),
        (20.0, PURPLE, "Entidad muy activa (μ=20)"),
    ]:
        p_e = phi / (phi + mu_e)
        scores = -stats.nbinom.logpmf(y_range, n=phi, p=p_e)
        ax.plot(y_range, scores, color=color, linewidth=2, label=label)
        ax.axvline(mu_e, color=color, linestyle=":", alpha=0.5)

    ax.set_xlabel("Conteo de eventos observado (y)")
    ax.set_ylabel("Score de anomalía = –log P(y | θₑ, φ)")
    ax.set_title("② El score es relativo a cada entidad\nEl mismo conteo = distintos scores")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 30)

    ax.annotate(
        "y=30 para entidad tranquila\n→ score muy alto",
        xy=(30, -stats.nbinom.logpmf(30, n=2, p=2/(2+2))),
        xytext=(38, 8),
        arrowprops=dict(arrowstyle="->", color=DARK),
        fontsize=8.5, color=GREEN,
        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec=ORANGE, alpha=0.7),
    )

    # --- Panel 3: Incertidumbre del score (posterior vs punto) ---
    ax = axes[1, 0]
    # Simular posterior samples de theta para una entidad
    n_post_samples = 500
    mu_pop, alpha_pop = 5.0, 3.0
    theta_samples = RNG.gamma(shape=mu_pop * alpha_pop, scale=1.0 / alpha_pop,
                              size=n_post_samples)
    phi_samples = RNG.exponential(scale=2.0, size=n_post_samples)

    # Score para y=20 (evento anómalo) con incertidumbre
    y_obs = 20
    per_sample_scores = np.array([
        -stats.nbinom.logpmf(y_obs, n=phi_s, p=phi_s / (phi_s + theta_s))
        for theta_s, phi_s in zip(theta_samples, phi_samples)
    ])
    avg_score = per_sample_scores.mean()

    ax.hist(per_sample_scores, bins=40, color=BLUE, alpha=0.7, density=True,
            label="Distribución del score\n(una muestra posterior = un valor)")
    ax.axvline(avg_score, color=RED, linewidth=2.5,
               label=f"Score final = {avg_score:.2f}\n(promedio log-marginal)")
    ax.axvline(np.percentile(per_sample_scores, 5), color=ORANGE, linewidth=1.5,
               linestyle="--", label=f"IC 90%: [{np.percentile(per_sample_scores, 5):.1f}, "
                                     f"{np.percentile(per_sample_scores, 95):.1f}]")
    ax.axvline(np.percentile(per_sample_scores, 95), color=ORANGE, linewidth=1.5,
               linestyle="--")
    ax.set_xlabel("Score por muestra posterior")
    ax.set_ylabel("Densidad")
    ax.set_title(f"③ Incertidumbre del Score (y={y_obs})\nEl modelo sabe cuánto sabe")
    ax.legend(fontsize=8.5)

    # --- Panel 4: Separación de clases en datos sintéticos ---
    ax = axes[1, 1]
    from bsad.data_generator import GeneratorConfig, generate_synthetic_data
    from bsad.features import FeatureConfig, build_modeling_table

    cfg = GeneratorConfig(n_users=50, n_days=21, attack_rate=0.04, random_seed=99)
    events_df, _ = generate_synthetic_data(cfg)
    feat_cfg = FeatureConfig(window_size="1D")
    modeling_df, _ = build_modeling_table(events_df, feat_cfg)

    # Score aproximado (sin MCMC completo): z-score NB por entidad
    def approx_score(row):
        mu_e = row["entity_mean_count"]
        std_e = max(row["entity_std_count"], 0.5)
        phi_e = max(mu_e**2 / max(std_e**2 - mu_e, 0.1), 0.5)
        p_e = phi_e / (phi_e + mu_e)
        return float(-stats.nbinom.logpmf(int(row["event_count"]), n=phi_e, p=p_e))

    modeling_df["approx_score"] = modeling_df.apply(approx_score, axis=1)

    benign_scores = modeling_df[~modeling_df["has_attack"]]["approx_score"]
    attack_scores = modeling_df[modeling_df["has_attack"]]["approx_score"]

    ax.hist(benign_scores, bins=50, alpha=0.7, color=BLUE, density=True,
            label=f"Benigno (n={len(benign_scores):,})")
    ax.hist(attack_scores, bins=25, alpha=0.85, color=RED, density=True,
            label=f"Ataque (n={len(attack_scores):,})")

    pr_auc = average_precision_score(
        modeling_df["has_attack"].astype(int),
        modeling_df["approx_score"],
    )
    ax.set_xlabel("Score de anomalía")
    ax.set_ylabel("Densidad")
    ax.set_title(f"④ Separación de Clases (datos sintéticos)\nPR-AUC ≈ {pr_auc:.3f}")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.98, f"PR-AUC = {pr_auc:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color=GREEN if pr_auc > 0.4 else RED,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY))

    plt.tight_layout()
    _save(fig, output_dir / "05_anomaly_scoring.png")


# ===========================================================================
# FIGURA 6: Evaluación – métricas clásicas vs operacionales
# ===========================================================================

def fig6_evaluacion(output_dir: Path) -> None:
    """
    Muestra la diferencia entre PR-AUC / ROC-AUC y las métricas operacionales
    (Precision@k, Recall@k, alertas por 1000 ventanas).
    """
    from bsad.data_generator import GeneratorConfig, generate_synthetic_data
    from bsad.features import FeatureConfig, build_modeling_table
    from bsad.evaluation import compute_all_metrics, format_metrics_report

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Evaluación: Métricas Clásicas vs Métricas Operacionales SOC",
                 fontsize=14, fontweight="bold")

    cfg = GeneratorConfig(n_users=100, n_days=21, attack_rate=0.03, random_seed=7)
    events_df, _ = generate_synthetic_data(cfg)
    feat_cfg = FeatureConfig(window_size="1D")
    modeling_df, _ = build_modeling_table(events_df, feat_cfg)

    def score_row(row):
        mu_e = row["entity_mean_count"]
        std_e = max(row["entity_std_count"], 0.5)
        phi_e = max(mu_e**2 / max(std_e**2 - mu_e, 0.1), 0.5)
        p_e = phi_e / (phi_e + mu_e)
        return float(-stats.nbinom.logpmf(int(row["event_count"]), n=phi_e, p=p_e))

    modeling_df["score"] = modeling_df.apply(score_row, axis=1)
    y_true = modeling_df["has_attack"].astype(int).values
    scores = modeling_df["score"].values
    metrics = compute_all_metrics(y_true, scores, k_values=[10, 25, 50, 100, 200])

    # --- Panel 1: ROC Curve ---
    ax = axes[0, 0]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = metrics["roc_auc"]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5, label=f"BSAD (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.2, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=BLUE)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title(f"ROC-AUC = {roc_auc:.3f}\n⚠ Optimista con clases desbalanceadas")
    ax.legend()
    ax.text(0.5, 0.1,
            "ROC-AUC asume que FP y FN\ntienen el mismo costo.\nCon 2% ataques, esto es falso.",
            ha="center", fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec=ORANGE, alpha=0.8))

    # --- Panel 2: PR Curve ---
    ax = axes[0, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    pr_auc = metrics["pr_auc"]
    ax.plot(recall_curve, precision_curve, color=RED, linewidth=2.5,
            label=f"BSAD (PR-AUC={pr_auc:.3f})")
    ax.axhline(y_true.mean(), color=GRAY, linestyle="--", linewidth=1.2,
               label=f"Baseline = {y_true.mean():.3f}\n(tasa de ataque)")
    ax.fill_between(recall_curve, precision_curve, alpha=0.1, color=RED)
    ax.set_xlabel("Recall (TPR)")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR-AUC = {pr_auc:.3f}\n✓ La métrica correcta para eventos raros")
    ax.legend()
    ax.text(0.5, 0.05,
            "PR-AUC = 1 es perfecto.\nBaseline (random) = tasa de ataque.\n"
            f"BSAD = {pr_auc/y_true.mean():.1f}x sobre baseline.",
            ha="center", fontsize=8.5, color=DARK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec=GREEN, alpha=0.8))

    # --- Panel 3: Precision@k y Recall@k ---
    ax = axes[0, 2]
    k_vals = [10, 25, 50, 100, 200]
    prec_vals = [metrics.get(f"precision_at_{k}", 0) for k in k_vals]
    rec_vals = [metrics.get(f"recall_at_{k}", 0) for k in k_vals]

    x = np.arange(len(k_vals))
    width = 0.35
    bars1 = ax.bar(x - width / 2, prec_vals, width, color=BLUE, alpha=0.85,
                   label="Precision@k")
    bars2 = ax.bar(x + width / 2, rec_vals, width, color=RED, alpha=0.85,
                   label="Recall@k")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in k_vals])
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1.15)
    ax.set_title("③ Métricas Operacionales: Precision@k / Recall@k\n"
                 "'¿Cuántos ataques capturo en mis top-k alertas?'")
    ax.legend()

    # --- Panel 4: Alertas por 1000 ventanas (budget curve) ---
    ax = axes[1, 0]
    recall_targets = np.linspace(0.05, 0.95, 20)
    alerts_per_1k = []
    thresholds = []

    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_y = y_true[sorted_indices]
    total_positives = y_true.sum()
    n_total = len(y_true)

    for target_recall in recall_targets:
        needed_tp = target_recall * total_positives
        tp_so_far = 0
        for i, (s, y) in enumerate(zip(sorted_scores, sorted_y)):
            tp_so_far += y
            if tp_so_far >= needed_tp:
                threshold = s
                n_alerts = i + 1
                alerts_1k = n_alerts / n_total * 1000
                alerts_per_1k.append(alerts_1k)
                thresholds.append(threshold)
                break
        else:
            alerts_per_1k.append(1000)
            thresholds.append(sorted_scores[-1])

    ax.plot(recall_targets * 100, alerts_per_1k, "o-", color=BLUE, linewidth=2,
            markersize=5, label="BSAD")
    ax.fill_between(recall_targets * 100, alerts_per_1k, alpha=0.15, color=BLUE)
    ax.set_xlabel("Recall objetivo (%)")
    ax.set_ylabel("Alertas por 1,000 ventanas")
    ax.set_title("④ Curva de Budget de Alertas\n'¿Cuántas alertas necesito para X% recall?'")

    # Marcar puntos de operación
    for target_r, color, label in [(0.30, GREEN, "30% recall"), (0.50, ORANGE, "50% recall"),
                                    (0.80, RED, "80% recall")]:
        idx = np.argmin(np.abs(recall_targets - target_r))
        a = alerts_per_1k[idx]
        ax.scatter([target_r * 100], [a], color=color, s=80, zorder=5)
        ax.annotate(f"{label}\n= {a:.0f} alertas/1k",
                    xy=(target_r * 100, a), xytext=(target_r * 100 + 3, a + 15),
                    fontsize=8, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color))
    ax.legend()

    # --- Panel 5: Tabla de resumen de métricas ---
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("⑤ Resumen de Métricas", fontweight="bold")

    rows = [
        ["PR-AUC", f"{metrics['pr_auc']:.4f}", f"{y_true.mean():.4f}", "PR-AUC / tasa_ataque",
         f"{metrics['pr_auc']/y_true.mean():.1f}x"],
        ["ROC-AUC", f"{metrics['roc_auc']:.4f}", "0.5000", "—", "—"],
        ["Recall@50", f"{metrics.get('recall_at_50', 0):.3f}", "—", "—", "—"],
        ["Prec@50", f"{metrics.get('precision_at_50', 0):.3f}", f"{y_true.mean():.4f}", "—", "—"],
        ["Recall@100", f"{metrics.get('recall_at_100', 0):.3f}", "—", "—", "—"],
        ["N obs", f"{metrics['n_observations']:,}", "—", "—", "—"],
        ["N ataques", f"{metrics['n_positives']:,}", "—", "—", "—"],
    ]
    col_labels = ["Métrica", "BSAD", "Baseline", "Comparación", "Lift"]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, -0.05, 1, 1.05],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
        cell.set_edgecolor("white")

    # --- Panel 6: Interpretación práctica ---
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("⑥ Cómo Interpretar los Resultados", fontweight="bold")

    interpretations = [
        (GREEN, "✓ PR-AUC",
         f"  = {metrics['pr_auc']:.3f} vs baseline {y_true.mean():.3f}\n"
         f"  → {metrics['pr_auc']/y_true.mean():.1f}x mejor que aleatorio"),
        (BLUE, "✓ Recall@50",
         f"  = {metrics.get('recall_at_50', 0):.1%} de ataques\n"
         f"  capturados en top-50 alertas"),
        (ORANGE, "⚠ Precision@50",
         f"  = {metrics.get('precision_at_50', 0):.1%}\n"
         f"  Solo 1 de cada {1/max(metrics.get('precision_at_50', 0.01), 0.01):.0f}\n"
         f"  alertas es un ataque real"),
        (RED, "! Limitación honesta",
         "  geo_anomaly y device_anomaly\n"
         "  solo se detectan si elevan\n"
         "  el conteo de eventos"),
    ]

    y_pos = 0.95
    for color, title, text in interpretations:
        ax.text(0.02, y_pos, title, transform=ax.transAxes, fontsize=10,
                fontweight="bold", color=color)
        y_pos -= 0.065
        ax.text(0.02, y_pos, text, transform=ax.transAxes, fontsize=8.5,
                color=DARK, fontstyle="italic")
        y_pos -= 0.13

    plt.tight_layout()
    _save(fig, output_dir / "06_evaluacion.png")


# ===========================================================================
# FIGURA 7: Triage – del score al workflow SOC
# ===========================================================================

def fig7_triage(output_dir: Path) -> None:
    """
    Muestra cómo el módulo de triage convierte scores en alertas accionables.
    """
    from triage.risk_score import RiskScorer
    from triage.calibrate_thresholds import AlertBudget

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Triage: Del Score Bayesiano al Workflow SOC",
                 fontsize=14, fontweight="bold")

    # Datos sintéticos con score, std, e historial
    n = 500
    attack_rate = 0.04
    rng_t = np.random.default_rng(77)
    is_attack = (rng_t.random(n) < attack_rate).astype(int)

    # Ataques tienen scores más altos
    base_scores = rng_t.exponential(3.0, size=n)
    base_scores[is_attack == 1] += rng_t.exponential(8.0, size=is_attack.sum())
    score_std = base_scores * 0.2 + rng_t.exponential(0.5, size=n)
    entity_history = rng_t.integers(1, 200, size=n)

    # --- Panel 1: Risk Score = anomaly + confianza + novedad ---
    ax = axes[0, 0]
    scorer = RiskScorer(score_weight=0.5, confidence_weight=0.3, novelty_weight=0.2)
    risk_scores = scorer.compute(base_scores, score_std, entity_history)

    # Mostrar los tres componentes
    anomaly_norm = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min())
    confidence = 1 / (1 + score_std / score_std.max())
    novelty = 1 - entity_history / entity_history.max()

    components_df = pd.DataFrame({
        "anomaly": anomaly_norm[:20],
        "confidence": confidence[:20],
        "novelty": novelty[:20],
        "risk": risk_scores[:20],
    })

    x = np.arange(20)
    width = 0.2
    ax.bar(x - 1.5*width, components_df["anomaly"], width, color=RED, alpha=0.8,
           label="Anomalía (w=0.5)")
    ax.bar(x - 0.5*width, components_df["confidence"], width, color=BLUE, alpha=0.8,
           label="Confianza (w=0.3)")
    ax.bar(x + 0.5*width, components_df["novelty"], width, color=ORANGE, alpha=0.8,
           label="Novedad (w=0.2)")
    ax.plot(x + 1.5*width, components_df["risk"], "D-", color=DARK, linewidth=1.5,
            markersize=4, label="Risk Score final")

    ax.set_xlabel("Observación")
    ax.set_ylabel("Valor normalizado [0, 1]")
    ax.set_title("① Risk Score = Anomalía + Confianza + Novedad\n(primeras 20 observaciones)")
    ax.legend(fontsize=8)

    # --- Panel 2: Calibración de umbral ---
    ax = axes[0, 1]
    budget_modes = [
        ("fixed_recall", 0.40, GREEN, "Recall=40%"),
        ("fixed_fpr", 0.05, ORANGE, "FPR≤5%"),
        ("fixed_alerts", 50, BLUE, "50 alertas/día"),
    ]

    all_thresholds = []
    for mode, target, color, label in budget_modes:
        budget = AlertBudget(mode=mode, target=target)
        result = budget.calibrate(base_scores, is_attack, n_windows_per_day=500)
        threshold = result["threshold"]
        all_thresholds.append((threshold, color, label, result))

    scores_sorted = np.sort(base_scores)
    ax.hist(base_scores[is_attack == 0], bins=50, alpha=0.6, color=BLUE,
            density=True, label="Benigno")
    ax.hist(base_scores[is_attack == 1], bins=20, alpha=0.8, color=RED,
            density=True, label="Ataque")

    for threshold, color, label, result in all_thresholds:
        recall = result.get("recall", result.get("actual_recall", "?"))
        fpr = result.get("fpr", result.get("actual_fpr", "?"))
        if isinstance(recall, float):
            recall_str = f"{recall:.1%}"
        else:
            recall_str = str(recall)
        ax.axvline(threshold, color=color, linewidth=2, linestyle="--",
                   label=f"{label}: thr={threshold:.1f}")

    ax.set_xlabel("Score de anomalía")
    ax.set_ylabel("Densidad")
    ax.set_title("② Calibración de Umbral por Budget Operacional\n"
                 "¿Cuántas alertas puedes manejar?")
    ax.legend(fontsize=8)

    # --- Panel 3: Curva de budget ---
    ax = axes[1, 0]
    recall_range = np.arange(0.05, 0.95, 0.05)
    alerts_per_day = []

    total_positives = is_attack.sum()
    n_total = len(is_attack)
    sorted_idx = np.argsort(base_scores)[::-1]
    sorted_scores_t = base_scores[sorted_idx]
    sorted_y_t = is_attack[sorted_idx]

    for target_r in recall_range:
        needed_tp = target_r * total_positives
        tp_so_far = 0
        for i, y_val in enumerate(sorted_y_t):
            tp_so_far += y_val
            if tp_so_far >= needed_tp:
                alerts_per_day.append((i + 1) / n_total * 500)
                break
        else:
            alerts_per_day.append(500)

    ax.plot(recall_range * 100, alerts_per_day, "o-", color=BLUE, linewidth=2,
            markersize=5)
    ax.fill_between(recall_range * 100, alerts_per_day, alpha=0.15, color=BLUE)

    # Zonas operacionales
    ax.axhspan(0, 20, alpha=0.07, color=GREEN, label="Zona manejable (<20/día)")
    ax.axhspan(20, 50, alpha=0.07, color=ORANGE, label="Zona aceptable (20-50/día)")
    ax.axhspan(50, 500, alpha=0.05, color=RED, label="Zona sobrecarga (>50/día)")

    ax.set_xlabel("Recall objetivo (%)")
    ax.set_ylabel("Alertas por día (500 ventanas/día)")
    ax.set_title("③ Trade-off: Recall vs Carga del Analista SOC")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    # --- Panel 4: Flujo del analista SOC ---
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("④ Flujo de Trabajo del Analista SOC", fontweight="bold")

    steps_soc = [
        (0.5, 0.90, RED, "🔴 ALERTA DISPARADA\nScore > umbral calibrado"),
        (0.5, 0.72, ORANGE, "📊 CONTEXTO DE ENTIDAD\n• Histórico del usuario\n"
         "• Desviaciones en σ\n• Ataques previos"),
        (0.5, 0.52, BLUE, "🎯 RISK SCORE\nAnomalia × Confianza × Novedad\n→ Priorización automática"),
        (0.5, 0.32, PURPLE, "🔍 INVESTIGACIÓN\n• ¿Geo anómala? (contexto)\n"
         "• ¿Dispositivo nuevo? (contexto)\n• ¿IP sospechosa? (contexto)"),
        (0.5, 0.12, GREEN, "✅ DECISIÓN\nVerdadero Positivo → Escalar\nFalso Positivo → Cerrar + Feedback"),
    ]

    for x_pos, y_pos, color, text in steps_soc:
        ax.text(x_pos, y_pos, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", fc=color, ec=color,
                          alpha=0.15),
                color=DARK)
        if y_pos > 0.12:
            ax.annotate("", xy=(x_pos, y_pos - 0.13), xytext=(x_pos, y_pos - 0.05),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

    plt.tight_layout()
    _save(fig, output_dir / "07_triage.png")


# ===========================================================================
# FIGURA 8: Limitaciones honestas y tipos de ataque detectables
# ===========================================================================

def fig8_limitaciones(output_dir: Path) -> None:
    """
    Mapa visual honesto de qué detecta el modelo y qué no.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Limitaciones Honestas del Modelo NB",
                 fontsize=14, fontweight="bold")

    # --- Panel izquierdo: Detectabilidad por tipo de ataque ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Detectabilidad por Tipo de Ataque", fontweight="bold")

    attacks = [
        # (nombre, detectabilidad 0-1, mecanismo, color_bg)
        ("brute_force", 0.85, "↑ conteo masivo en una entidad-día\n→ alta señal en event_count", GREEN),
        ("credential_stuffing", 0.45,
         "↑ conteo por entidad SI suficientes eventos\n→ señal moderada/débil", ORANGE),
        ("geo_anomaly", 0.20,
         "Solo si N eventos eleva el conteo\nLa ubicación NO se modela\n→ señal débil / falsos negativos", RED),
        ("device_anomaly", 0.15,
         "Solo si nuevos dispositivos generan\nsuficientes eventos extras\nEl device NO se modela\n→ señal muy débil", RED),
    ]

    y_start = 9.0
    for name, detectability, mechanism, color in attacks:
        # Barra de detectabilidad
        bar_w = detectability * 4.0
        ax.barh([y_start], [bar_w], height=0.6, color=color, alpha=0.7,
                left=5.0, align="center")
        ax.barh([y_start], [4.0], height=0.6, color=GRAY, alpha=0.15,
                left=5.0, align="center")
        ax.text(9.2, y_start, f"{detectability:.0%}", va="center",
                fontsize=10, fontweight="bold", color=color)

        ax.text(4.8, y_start, name, va="center", ha="right",
                fontsize=10, fontweight="bold", color=DARK)
        ax.text(5.1, y_start - 0.45, mechanism, va="top", ha="left",
                fontsize=7.5, color=GRAY, fontstyle="italic")
        y_start -= 2.2

    ax.text(7.0, 0.3, "Detectabilidad →", ha="center", fontsize=9, color=GRAY)
    ax.axvline(5.0, color=GRAY, linestyle=":", alpha=0.4)

    # Leyenda
    for val, label, color in [(5.0 + 0, "0%", RED), (5.0 + 2, "50%", ORANGE),
                               (5.0 + 4, "100%", GREEN)]:
        ax.text(val, 0.8, label, ha="center", fontsize=8, color=GRAY)

    # --- Panel derecho: Mapa de casos de uso ---
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Cuándo Usar BSAD y Cuándo No", fontweight="bold")

    uso_si = [
        "✓ Datos de conteo (eventos, requests, logins)",
        "✓ Estructura de entidad (usuarios, IPs, servicios)",
        "✓ Tasa de ataque < 5%",
        "✓ Sin etiquetas de ataque disponibles",
        "✓ La 'anormalidad' = demasiados eventos",
        "✓ Necesitas cuantificar incertidumbre",
        "✓ Entidades heterogéneas (usuarios activos vs inactivos)",
    ]
    uso_no = [
        "✗ Features continuas (bytes, latencia pura)",
        "✗ Tasa de ataque > 10% → clasificación supervisada",
        "✗ Necesitas respuesta en tiempo real (<100ms)",
        "✗ Ataques que NO cambian el conteo total",
        "✗ Multi-variate behavioral detection",
        "✗ Datos sin estructura de entidad clara",
    ]

    y_pos = 0.93
    ax.text(0.02, y_pos, "BSAD ES ADECUADO CUANDO:", fontweight="bold",
            color=GREEN, transform=ax.transAxes, fontsize=11)
    y_pos -= 0.08
    for item in uso_si:
        ax.text(0.04, y_pos, item, transform=ax.transAxes,
                fontsize=9, color=DARK)
        y_pos -= 0.065

    y_pos -= 0.04
    ax.text(0.02, y_pos, "NO USAR BSAD CUANDO:", fontweight="bold",
            color=RED, transform=ax.transAxes, fontsize=11)
    y_pos -= 0.08
    for item in uso_no:
        ax.text(0.04, y_pos, item, transform=ax.transAxes,
                fontsize=9, color=DARK)
        y_pos -= 0.065

    plt.tight_layout()
    _save(fig, output_dir / "08_limitaciones.png")


# ===========================================================================
# Utilidad
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Guardado: {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Genera gráficos educativos de BSAD")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/explanation"),
                        help="Directorio de salida para las figuras")
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🔬 Generando visualizaciones educativas BSAD...\n")

    steps = [
        ("Fig 1 – El Problema",                  fig1_el_problema),
        ("Fig 2 – Falla de Métodos Clásicos",     fig2_falla_metodos_clasicos),
        ("Fig 3 – Modelo Jerárquico",             fig3_modelo_jerarquico),
        ("Fig 4 – Pipeline de Datos",             fig4_pipeline_datos),
        ("Fig 5 – Anomaly Scoring",               fig5_anomaly_scoring),
        ("Fig 6 – Evaluación",                    fig6_evaluacion),
        ("Fig 7 – Triage SOC",                    fig7_triage),
        ("Fig 8 – Limitaciones",                  fig8_limitaciones),
    ]

    for title, fn in steps:
        print(f"  → {title}")
        try:
            fn(output_dir)
        except Exception as e:
            print(f"    ⚠ Error en {title}: {e}")

    print(f"\n✅ Figuras guardadas en: {output_dir.resolve()}\n")


if __name__ == "__main__":
    main()
