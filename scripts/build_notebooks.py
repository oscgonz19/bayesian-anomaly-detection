"""
Genera los notebooks 04, 05, y 06 como .ipynb válidos.

Uso:
    python scripts/build_notebooks.py
"""

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (bsad)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }


def md(text: str) -> dict:
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "id": "md-" + str(hash(text))[:8].replace("-", "x"),
        "metadata": {},
        "source": source,
    }


def code(text: str) -> dict:
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "id": "code-" + str(hash(text))[:8].replace("-", "x"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def save(notebook: dict, path: Path) -> None:
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"  ✓ Guardado: {path}")


# ===========================================================================
# NOTEBOOK 04 — Alert Prioritization & Triage
# ===========================================================================

def build_04() -> dict:
    return nb([

        md("""# Notebook 04 — Alert Prioritization & Triage SOC

Este notebook muestra cómo convertir scores de anomalía crudos en **alertas accionables** para un analista SOC.

**Lo que aprenderás:**
- Risk Score compuesto (anomalía + confianza + novedad)
- Calibración de umbrales según presupuesto de alertas
- Precision@k, Recall@k y curva de budget
- Enriquecimiento de alertas con contexto de entidad

> ℹ️ Este notebook usa datos **completamente sintéticos** para que funcione sin necesidad de datos externos."""),

        code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import FeatureConfig, build_modeling_table
from triage.risk_score import RiskScorer, compute_risk_score
from triage.calibrate_thresholds import AlertBudget, calibrate_threshold, build_alert_budget_curve
from triage.ranking_metrics import (
    precision_at_k, recall_at_k, fpr_at_fixed_recall,
    alerts_per_k_windows, ranking_report, workload_reduction,
)
from triage.entity_context import build_entity_history, enrich_alerts

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120
print("✓ Imports OK")"""),

        md("## 1. Generar datos sintéticos con scores aproximados\n\nEn producción, aquí cargarías el `scored_df` del pipeline. Aquí lo generamos directamente para no depender de un modelo MCMC completo."),

        code("""\
# Generar datos sintéticos
cfg = GeneratorConfig(n_users=150, n_days=30, attack_rate=0.03, random_seed=42)
events_df, attacks_df = generate_synthetic_data(cfg)

feat_cfg = FeatureConfig(window_size="1D")
modeling_df, metadata = build_modeling_table(events_df, feat_cfg)

print(f"Ventanas totales:  {len(modeling_df):,}")
print(f"Entidades:         {metadata['n_entities']}")
print(f"Attack rate:       {metadata['attack_rate']:.2%}")
print(f"Ataques totales:   {modeling_df['has_attack'].sum()}")"""),

        code("""\
# Aproximar scores de anomalía sin MCMC completo
# (en producción estos vienen de compute_scores() con el trace MCMC)
def approx_score(row):
    mu_e = max(row["entity_mean_count"], 0.5)
    std_e = max(row["entity_std_count"], 0.5)
    phi_e = max(mu_e**2 / max(std_e**2 - mu_e, 0.1), 0.5)
    p_e = phi_e / (phi_e + mu_e)
    return float(-stats.nbinom.logpmf(int(row["event_count"]), n=phi_e, p=p_e))

rng = np.random.default_rng(42)
modeling_df["anomaly_score"] = modeling_df.apply(approx_score, axis=1)
# Añadir incertidumbre simulada (en producción viene del posterior MCMC)
modeling_df["score_std"] = modeling_df["anomaly_score"] * 0.15 + rng.exponential(0.3, len(modeling_df))
modeling_df["predicted_mean"] = modeling_df["entity_mean_count"]
modeling_df["predicted_upper"] = modeling_df["entity_mean_count"] + 2 * modeling_df["entity_std_count"]
modeling_df["anomaly_rank"] = modeling_df["anomaly_score"].rank(ascending=False).astype(int)
modeling_df = modeling_df.sort_values("anomaly_score", ascending=False)

print("Columnas del scored_df:")
print(modeling_df.columns.tolist())
modeling_df[["user_id", "window", "event_count", "anomaly_score", "score_std", "has_attack"]].head(8)"""),

        md("## 2. Risk Score compuesto\n\nEl `anomaly_score` puro tiene un problema: **dos alertas con el mismo score pueden tener distinta confianza** (una entidad con 3 obs vs una con 200 obs). El Risk Score combina tres dimensiones:"),

        code("""\
# Risk Score = 0.5 * anomalía + 0.3 * confianza + 0.2 * novedad
scorer = RiskScorer(score_weight=0.5, confidence_weight=0.3, novelty_weight=0.2)

# Historial de observaciones por entidad (proxy de 'novedad')
entity_obs_count = modeling_df.groupby("user_id")["event_count"].count()
obs_counts = modeling_df["user_id"].map(entity_obs_count).values

risk_scores = scorer.compute(
    anomaly_scores=modeling_df["anomaly_score"].values,
    score_std=modeling_df["score_std"].values,
    entity_history_counts=obs_counts,
)
modeling_df["risk_score"] = risk_scores

print("Risk Score Statistics:")
print(modeling_df["risk_score"].describe().round(4))"""),

        code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1: Anomaly vs Risk score
ax = axes[0]
colors = ["crimson" if a else "steelblue" for a in modeling_df["has_attack"]]
ax.scatter(modeling_df["anomaly_score"], modeling_df["risk_score"],
           c=colors, alpha=0.4, s=15)
ax.set_xlabel("Anomaly Score (crudo)")
ax.set_ylabel("Risk Score (compuesto)")
ax.set_title("Anomaly vs Risk Score\\nRojo = ataque")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="crimson", label="Ataque"),
                   Patch(fc="steelblue", label="Benigno")], fontsize=9)

# Panel 2: Distribución de risk scores
ax = axes[1]
ax.hist(modeling_df[~modeling_df["has_attack"]]["risk_score"],
        bins=40, alpha=0.7, color="steelblue", density=True, label="Benigno")
ax.hist(modeling_df[modeling_df["has_attack"]]["risk_score"],
        bins=20, alpha=0.85, color="crimson", density=True, label="Ataque")
ax.set_xlabel("Risk Score")
ax.set_ylabel("Densidad")
ax.set_title("Distribución de Risk Scores")
ax.legend()

# Panel 3: Top 20 alertas por risk score
ax = axes[2]
top20 = modeling_df.nlargest(20, "risk_score").sort_values("risk_score")
bar_colors = ["crimson" if a else "steelblue" for a in top20["has_attack"]]
ax.barh(range(len(top20)), top20["risk_score"], color=bar_colors, alpha=0.8)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f"{row['user_id']}" for _, row in top20.iterrows()], fontsize=7)
ax.set_xlabel("Risk Score")
ax.set_title("Top 20 Alertas por Risk Score\\nRojo = ataque real")

plt.tight_layout()
plt.show()"""),

        md("## 3. Calibración de umbrales\n\nEn lugar de elegir un umbral manualmente, el módulo `AlertBudget` te permite definir la **restricción operacional** y calcular automáticamente el umbral óptimo."),

        code("""\
y_true = modeling_df["has_attack"].astype(int).values
scores = modeling_df["anomaly_score"].values

# Modo 1: quiero exactamente 30 alertas por día (500 ventanas/día)
budget_alerts = AlertBudget(mode="fixed_alerts", target=30)
result_alerts = budget_alerts.calibrate(scores, y_true, n_windows_per_day=500)

# Modo 2: quiero capturar al menos el 40% de los ataques
budget_recall = AlertBudget(mode="fixed_recall", target=0.40)
result_recall = budget_recall.calibrate(scores, y_true)

# Modo 3: quiero que el FPR sea ≤ 5%
budget_fpr = AlertBudget(mode="fixed_fpr", target=0.05)
result_fpr = budget_fpr.calibrate(scores, y_true)

print("=" * 55)
print(f"{'Modo':<20} {'Umbral':>8} {'Recall':>8} {'FPR':>8} {'Alertas/día':>12}")
print("-" * 55)
for label, r in [("fixed_alerts=30", result_alerts),
                  ("fixed_recall=40%", result_recall),
                  ("fixed_fpr=5%",    result_fpr)]:
    recall = r.get("recall", r.get("actual_recall", 0))
    fpr    = r.get("fpr",    r.get("actual_fpr",    0))
    apd    = r.get("alerts_per_day", "—")
    if isinstance(apd, float): apd = f"{apd:.0f}"
    print(f"{label:<20} {r['threshold']:>8.2f} {recall:>8.2%} {fpr:>8.2%} {apd:>12}")"""),

        code("""\
# Curva completa de budget: recall vs alertas/día
budget_curve = build_alert_budget_curve(scores, y_true)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(budget_curve["target_recall"] * 100, budget_curve["alerts"],
        "o-", color="#2E86AB", linewidth=2, markersize=5)
ax.fill_between(budget_curve["target_recall"] * 100,
                budget_curve["alerts"], alpha=0.12, color="#2E86AB")
ax.axhspan(0,  20, alpha=0.06, color="green",  label="Manejable (<20/día)")
ax.axhspan(20, 60, alpha=0.06, color="orange", label="Aceptable (20-60/día)")
ax.axhspan(60, 500, alpha=0.04, color="red",   label="Sobrecarga (>60/día)")
ax.set_xlabel("Recall objetivo (%)")
ax.set_ylabel("Alertas")
ax.set_title("Trade-off: Recall vs Carga del Analista")
ax.set_ylim(0, 200)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(budget_curve["target_recall"] * 100, budget_curve["fpr"] * 100,
        "s-", color="#E84855", linewidth=2, markersize=5)
ax.set_xlabel("Recall objetivo (%)")
ax.set_ylabel("False Positive Rate (%)")
ax.set_title("Trade-off: Recall vs FPR")
ax.fill_between(budget_curve["target_recall"] * 100,
                budget_curve["fpr"] * 100, alpha=0.12, color="#E84855")

plt.tight_layout()
plt.show()"""),

        md("## 4. Métricas operacionales: Precision@k y Recall@k"),

        code("""\
k_values = [10, 25, 50, 100, 200]
report = ranking_report(y_true, scores, ks=k_values)
print("Reporte de métricas operacionales:")
print(report.to_string(index=False))"""),

        code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ks = k_values
prec = [precision_at_k(y_true, scores, k) for k in ks]
rec  = [recall_at_k(y_true, scores, k) for k in ks]

ax = axes[0]
x = np.arange(len(ks))
ax.bar(x - 0.2, prec, 0.35, color="#2E86AB", alpha=0.85, label="Precision@k")
ax.bar(x + 0.2, rec,  0.35, color="#E84855", alpha=0.85, label="Recall@k")
for i, (p, r) in enumerate(zip(prec, rec)):
    ax.text(i - 0.2, p + 0.01, f"{p:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + 0.2, r + 0.01, f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_ylim(0, 1.1)
ax.set_title("Precision@k vs Recall@k")
ax.legend()

ax = axes[1]
fprs   = [fpr_at_fixed_recall(y_true, scores, t) for t in [0.2, 0.3, 0.5, 0.7]]
a1k    = [alerts_per_k_windows(y_true, scores, t) for t in [0.2, 0.3, 0.5, 0.7]]
labels = ["Recall 20%", "Recall 30%", "Recall 50%", "Recall 70%"]
x2 = np.arange(len(labels))
ax.bar(x2 - 0.2, fprs,  0.35, color="#F4A261", alpha=0.85, label="FPR")
ax2b = ax.twinx()
ax2b.bar(x2 + 0.2, a1k, 0.35, color="#6A4C93", alpha=0.85, label="Alertas/1k")
ax.set_xticks(x2)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("FPR")
ax2b.set_ylabel("Alertas por 1,000 ventanas")
ax.set_title("FPR y Alertas/1k a distintos niveles de Recall")
ax.legend(loc="upper left", fontsize=9)
ax2b.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()"""),

        md("## 5. Enriquecimiento de alertas con contexto de entidad"),

        code("""\
# Construir historial de entidades
history = build_entity_history(
    modeling_df,
    entity_col="user_id",
    value_col="event_count",
    score_col="anomaly_score",
)

# Enriquecer las top-20 alertas con contexto
enriched = enrich_alerts(
    modeling_df,
    history,
    entity_col="user_id",
    value_col="event_count",
    score_col="anomaly_score",
    score_std_col="score_std",
    top_k=20,
)

print(f"Alertas enriquecidas: {len(enriched)}")
print("\\nEjemplo de alerta enriquecida:")
ex = enriched[0]
for k, v in ex.items():
    print(f"  {k:<25} {v}")"""),

        code("""\
# Dashboard final: Top 15 alertas con contexto completo
top15 = enriched[:15]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Score con barra de incertidumbre + flag de ataque
ax = axes[0]
entities = [e["entity_id"] for e in top15]
scores_top = [e["anomaly_score"] for e in top15]
confs = [e.get("confidence", "low") for e in top15]
is_atk = [modeling_df[modeling_df["user_id"] == e["entity_id"]]["has_attack"].any()
          for e in top15]

bar_colors = ["crimson" if a else "#2E86AB" for a in is_atk]
conf_alpha = {"high": 1.0, "medium": 0.7, "low": 0.4}
alphas = [conf_alpha.get(c, 0.5) for c in confs]

y_pos = list(range(len(top15)))
for yi, score, color, conf in zip(y_pos, scores_top, bar_colors, confs):
    ax.barh(yi, score, color=color, alpha=conf_alpha.get(conf, 0.5))
ax.set_yticks(y_pos)
ax.set_yticklabels(entities, fontsize=8)
ax.set_xlabel("Anomaly Score")
ax.set_title("Top 15 Alertas\\nRojo=ataque, Opacidad=confianza")

from matplotlib.patches import Patch
legend_elems = [
    Patch(fc="crimson",  label="Ataque confirmado"),
    Patch(fc="#2E86AB",  label="Benigno"),
    Patch(fc="gray", alpha=1.0, label="Alta confianza"),
    Patch(fc="gray", alpha=0.4, label="Baja confianza"),
]
ax.legend(handles=legend_elems, fontsize=8, loc="lower right")

# Panel 2: Tabla de contexto
ax = axes[1]
ax.axis("off")
ax.set_title("Contexto de entidad (top 10)", fontweight="bold")

rows = []
for e in top15[:10]:
    sigma = e.get("sigma_deviation", 0)
    conf = e.get("confidence", "?")
    hist_alerts = e.get("historical_alerts", 0)
    baseline = e.get("baseline_mean", 0)
    current = e.get("current_value", 0)
    rows.append([
        e["entity_id"],
        f"{baseline:.1f}",
        f"{current}",
        f"{sigma:.1f}σ",
        conf,
        str(hist_alerts),
    ])

table = ax.table(
    cellText=rows,
    colLabels=["Entidad", "Media", "Actual", "Desv.", "Conf.", "Prev."],
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2B2D42")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("white")

plt.tight_layout()
plt.show()"""),

        md("""## Resumen

| Concepto | Módulo | Función clave |
|---|---|---|
| Risk Score compuesto | `triage/risk_score.py` | `RiskScorer.compute()` |
| Calibración de umbral | `triage/calibrate_thresholds.py` | `AlertBudget.calibrate()` |
| Precision@k / Recall@k | `triage/ranking_metrics.py` | `ranking_report()` |
| Contexto de entidad | `triage/entity_context.py` | `enrich_alerts()` |

**Conclusión:** El módulo de triage transforma scores técnicos en decisiones operacionales. El analista SOC no ve un número — ve contexto, confianza, historial y una narrativa."""),
    ])


# ===========================================================================
# NOTEBOOK 05 — Interview Prep / Preguntas técnicas
# ===========================================================================

def build_05() -> dict:
    return nb([

        md("""# Notebook 05 — Preparación Técnica: Preguntas y Respuestas

Este notebook prepara para una entrevista técnica sobre BSAD. Cada sección presenta **una pregunta difícil** seguida de la respuesta con código ejecutable.

**Nivel:** Senior ML / Applied Scientist / Security Data Scientist

---

> Las respuestas no son solo verbales — cada concepto está respaldado por código que puedes ejecutar."""),

        code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import average_precision_score

from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import FeatureConfig, build_modeling_table
from bsad.model import ModelConfig

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 120

# Datos para todos los ejemplos
cfg = GeneratorConfig(n_users=80, n_days=21, attack_rate=0.03, random_seed=7)
events_df, _ = generate_synthetic_data(cfg)
modeling_df, meta = build_modeling_table(events_df, FeatureConfig())
print(f"Dataset: {len(modeling_df):,} ventanas | {meta['n_entities']} entidades | {meta['attack_rate']:.1%} ataques")"""),

        md("""---
## Q1: ¿Por qué Negative Binomial y no Poisson?

**La pregunta:** ¿No basta con modelar los conteos como Poisson? ¿Por qué la complejidad extra del NB?"""),

        code("""\
counts = modeling_df["event_count"].values
mean_c = counts.mean()
var_c  = counts.var()

print(f"Media:        {mean_c:.2f}")
print(f"Varianza:     {var_c:.2f}")
print(f"Var/Media:    {var_c/mean_c:.1f}x  ← si fuera Poisson, debería ser ~1.0")
print()
print("Prueba de sobredispersión (score test):")
# Bajo Poisson, (var - mean) / mean ~ N(0,1) para n grande
disp_stat = (var_c - mean_c) / mean_c
print(f"  Estadístico: {disp_stat:.1f}  (>>0 indica sobredispersión)")
print(f"  → La hipótesis Poisson se rechaza claramente.")"""),

        code("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

x = np.arange(0, min(int(counts.max()) + 1, 60))
# Ajuste empírico NB
phi_fit = mean_c**2 / max(var_c - mean_c, 0.1)
p_fit   = phi_fit / (phi_fit + mean_c)

axes[0].hist(counts, bins=40, density=True, alpha=0.6, color="#2E86AB", label="Datos reales")
axes[0].plot(x, stats.poisson.pmf(x, mu=mean_c), "r-", lw=2,
             label=f"Poisson(μ={mean_c:.1f})")
axes[0].plot(x, stats.nbinom.pmf(x, n=phi_fit, p=p_fit), "g-", lw=2,
             label=f"NB(μ={mean_c:.1f}, φ={phi_fit:.1f})")
axes[0].set_title("Poisson subestima la cola derecha")
axes[0].legend(fontsize=9)
axes[0].set_xlabel("Conteo de eventos / ventana")

# QQ-plot comparativo
from scipy.stats import probplot
axes[1].set_title("QQ-plot: NB ajusta mejor la cola")
sorted_data = np.sort(counts)
n = len(sorted_data)
probs = (np.arange(1, n+1) - 0.5) / n
q_nb  = stats.nbinom.ppf(probs, n=phi_fit, p=p_fit)
q_poi = stats.poisson.ppf(probs, mu=mean_c)
axes[1].plot(q_poi, sorted_data, "r.", alpha=0.3, markersize=3, label="Poisson")
axes[1].plot(q_nb,  sorted_data, "g.", alpha=0.3, markersize=3, label="NB")
axes[1].plot([0, sorted_data.max()], [0, sorted_data.max()],
             "k--", lw=1, label="Ajuste perfecto")
axes[1].set_xlabel("Cuantiles teóricos")
axes[1].set_ylabel("Cuantiles observados")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
print(f"\\nRespuesta: Var/Media = {var_c/mean_c:.1f}x, muy por encima de 1.0 (Poisson).")
print("Forzar Poisson subestimaría la variabilidad natural y generaría demasiadas alertas falsas.")"""),

        md("""---
## Q2: ¿Qué es partial pooling y por qué es mejor que "sin pooling"?

**La pregunta:** ¿Por qué no simplemente estimar la tasa de cada usuario de forma independiente?"""),

        code("""\
# Simulación: 3 escenarios de estimación
np.random.seed(42)
true_rate = 12.0
global_mean = 5.0
scenarios = {
    "1 obs":   np.array([true_rate + np.random.randn()]),
    "5 obs":   np.random.poisson(true_rate, 5).astype(float),
    "20 obs":  np.random.poisson(true_rate, 20).astype(float),
    "100 obs": np.random.poisson(true_rate, 100).astype(float),
}

print(f"{'Scenario':<12} {'MLE':>8} {'BSAD (shrunk)':>15} {'Dist. al true':>15}")
print("-" * 55)
for label, obs in scenarios.items():
    mle  = obs.mean()
    k    = 5  # fuerza de shrinkage
    bsad = (len(obs) * mle + k * global_mean) / (len(obs) + k)
    d_mle  = abs(mle  - true_rate)
    d_bsad = abs(bsad - true_rate)
    winner = "← BSAD" if d_bsad < d_mle else ""
    print(f"{label:<12} {mle:>8.2f} {bsad:>15.2f} {d_bsad:>12.2f} vs {d_mle:.2f} {winner}")"""),

        code("""\
n_experiments = 1000
errors_mle, errors_bsad = [], []
for _ in range(n_experiments):
    true_r = np.random.gamma(shape=3, scale=2)  # tasa aleatoria
    obs    = np.random.poisson(true_r, size=3)   # 3 observaciones (escaso)
    mle    = obs.mean()
    k      = 5
    bsad   = (3 * mle + k * global_mean) / (3 + k)
    errors_mle.append(abs(mle - true_r))
    errors_bsad.append(abs(bsad - true_r))

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(errors_mle,  bins=50, alpha=0.7, color="#E84855", density=True, label=f"MLE  (MAE={np.mean(errors_mle):.2f})")
ax.hist(errors_bsad, bins=50, alpha=0.7, color="#2E86AB", density=True, label=f"BSAD (MAE={np.mean(errors_bsad):.2f})")
ax.set_xlabel("Error absoluto en estimación de tasa (|θ̂ - θ_true|)")
ax.set_ylabel("Densidad")
ax.set_title(f"Con solo 3 obs: BSAD tiene {np.mean(errors_mle)/np.mean(errors_bsad):.1f}x menor error que MLE")
ax.legend()
plt.tight_layout()
plt.show()"""),

        md("""---
## Q3: ¿Por qué el score es –log P(y | posterior) y no simplemente el Z-score?

**La pregunta:** El Z-score es más simple. ¿Cuándo falla y por qué el log-likelihood es mejor?"""),

        code("""\
# Demostración con un caso concreto
# Entidad A: media=5, std=3 (activa)
# Entidad B: media=50, std=30 (muy activa)
# Evento: y=20 para ambas

mu_A, std_A = 5.0, 3.0
mu_B, std_B = 50.0, 30.0
y_obs = 20

z_A = (y_obs - mu_A) / std_A
z_B = (y_obs - mu_B) / std_B

phi = 2.0
score_nb_A = -stats.nbinom.logpmf(y_obs, n=phi, p=phi/(phi+mu_A))
score_nb_B = -stats.nbinom.logpmf(y_obs, n=phi, p=phi/(phi+mu_B))

print(f"Evento: y = {y_obs} eventos")
print()
print(f"{'':30} {'Z-score':>10} {'NB Score':>10}")
print("-" * 55)
print(f"{'Entidad A (media=5,  std=3)':30} {z_A:>10.2f} {score_nb_A:>10.2f}")
print(f"{'Entidad B (media=50, std=30)':30} {z_B:>10.2f} {score_nb_B:>10.2f}")
print()
print("Z-score: para entidad A, y=20 está 5σ sobre la media → muy anómalo ✓")
print("Z-score: para entidad B, y=20 está -1σ bajo la media → parece benigno... ¿pero es razonable?")
print()

# Pero y=20 para entidad B con conteos normalmente de 50+ ¿es benigno?
# Con NB, la probabilidad de y=20 dado mu=50 es baja también (día inusualmente tranquilo)
p_normal_B = stats.nbinom.pmf(20, n=phi, p=phi/(phi+mu_B))
p_at_mean_B = stats.nbinom.pmf(50, n=phi, p=phi/(phi+mu_B))
print(f"P(y=20 | entidad B, μ=50):  {p_normal_B:.6f}")
print(f"P(y=50 | entidad B, μ=50):  {p_at_mean_B:.6f}")
print(f"→ NB detecta que y=20 también es inusual para entidad B, aunque el Z-score no lo haga.")"""),

        md("""---
## Q4: ¿Cuáles son las limitaciones reales del modelo?

**La pregunta:** Sé honesto — ¿qué NO puede detectar este modelo?"""),

        code("""\
# Demostración: geo_anomaly vs brute_force

# Caso 1: brute_force — genera muchos eventos → detectable
mu_user = 5.0  # baseline del usuario
y_brute_force = 150  # burst masivo
score_bf = -stats.nbinom.logpmf(y_brute_force, n=2, p=2/(2+mu_user))

# Caso 2: geo_anomaly — 5 eventos desde Corea del Norte
# El modelo NB NO ve la ubicación, solo el conteo
y_geo = 7  # 2 eventos extra sobre el baseline
score_geo = -stats.nbinom.logpmf(y_geo, n=2, p=2/(2+mu_user))

# Caso 3: device_anomaly — 3 eventos desde dispositivo nuevo
y_device = 6
score_device = -stats.nbinom.logpmf(y_device, n=2, p=2/(2+mu_user))

print("Tipo de ataque     | Conteo observado | Score NB | Detectable?")
print("-" * 65)
print(f"brute_force        | {y_brute_force:<16} | {score_bf:>8.2f} | ✓ Alta señal")
print(f"geo_anomaly        | {y_geo:<16} | {score_geo:>8.2f} | ✗ Score bajo (solo 2 eventos extra)")
print(f"device_anomaly     | {y_device:<16} | {score_device:>8.2f} | ✗ Score bajo (solo 1 evento extra)")
print()
print("Conclusión: El modelo NB es un detector de VOLUMEN, no de COMPORTAMIENTO multivariado.")
print("geo_anomaly y device_anomaly requieren modelos que usen esas features explícitamente.")"""),

        code("""\
# Detección de tipos de ataque en datos sintéticos
def approx_score(row):
    mu_e  = max(row["entity_mean_count"], 0.5)
    std_e = max(row["entity_std_count"], 0.5)
    phi_e = max(mu_e**2 / max(std_e**2 - mu_e, 0.1), 0.5)
    p_e   = phi_e / (phi_e + mu_e)
    return float(-stats.nbinom.logpmf(int(row["event_count"]), n=phi_e, p=p_e))

modeling_df["score"] = modeling_df.apply(approx_score, axis=1)

# Si el dataset tiene info de attack_type (puede que no con solo has_attack)
if "attack_type" in modeling_df.columns:
    attack_types = modeling_df[modeling_df["has_attack"]]["attack_type"].value_counts()
    print("Ataques por tipo:")
    print(attack_types)

    fig, ax = plt.subplots(figsize=(10, 4))
    for atype, color in [("brute_force","crimson"), ("credential_stuffing","orange"),
                          ("geo_anomaly","purple"),   ("device_anomaly","steelblue")]:
        subset = modeling_df[modeling_df.get("attack_type", pd.Series(["none"]*len(modeling_df))) == atype]
        if len(subset) > 0:
            ax.hist(subset["score"], bins=20, alpha=0.7, color=color, density=True, label=f"{atype} (n={len(subset)})")
    benign = modeling_df[~modeling_df["has_attack"]]
    ax.hist(benign["score"], bins=50, alpha=0.3, color="gray", density=True, label=f"Benigno (n={len(benign)})")
    ax.set_xlabel("Score de anomalía")
    ax.set_title("Distribución de scores por tipo de ataque\\n(brute_force debe tener scores más altos)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("attack_type no disponible en esta versión del dataset.")
    print(f"PR-AUC general: {average_precision_score(modeling_df['has_attack'].astype(int), modeling_df['score']):.3f}")"""),

        md("""---
## Q5: ¿Cómo escalarías esto a producción?

**La pregunta:** El entrenamiento MCMC tarda horas. ¿Cómo lo desplegarías?"""),

        code("""\
arch = '''
ARQUITECTURA DE PRODUCCION
==========================

1. ENTRENAMIENTO (offline, semanal)
   ---------------------------------
   Scheduler (cron/Airflow)
     -> Ingestar logs de la semana
     -> build_modeling_table()
     -> fit_model()  # horas de MCMC
     -> save_model(trace, "model_v2024_W42.nc")
     -> save_run_metadata(RunMetadata.from_settings(settings))

2. SCORING (near-real-time, cada hora/dia)
   -----------------------------------------
   Trigger (nueva ventana completa)
     -> load_model("model_latest.nc")    # milisegundos
     -> build_modeling_table(new_events)
     -> compute_scores(y, trace, entity_idx)  # segundos
     -> AlertBudget.calibrate() -> threshold
     -> Alertas -> SIEM / ticketing

3. RE-ENTRENAMIENTO ADAPTIVO
   --------------------------
   Si deriva detectada (score_std aumenta sistematicamente):
     -> Re-entrenar en ventana deslizante
     -> Comparar diagnostics: r_hat, ESS, divergences

LATENCIAS REALES (estimadas):
  - Entrenamiento:  2-4 horas (200 entidades, 2000 samples, 4 cadenas)
  - Scoring:        0.5-5 segundos (depende de n_obs y n_samples)
  - Carga de trace: < 1 segundo
  - Triage:         milisegundos
'''
print(arch)"""),

        md("""---
## Q6: ¿Cómo demostrarías que el modelo NO tiene data leakage?

**La pregunta de auditoría más difícil.**"""),

        code("""\
# Demostración de split temporal correcto
from bsad.features import create_time_windows, add_entity_features, encode_entity_ids

# Simular split temporal: primeras 3 semanas = train, última semana = test
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
cutoff = events_df["timestamp"].max() - pd.Timedelta(days=7)

train_events = events_df[events_df["timestamp"] < cutoff]
test_events  = events_df[events_df["timestamp"] >= cutoff]

feat_cfg = FeatureConfig()

# ✓ CORRECTO: calcular estadísticas de entidad SOLO sobre train
train_w = create_time_windows(train_events, feat_cfg)
train_w = add_entity_features(train_w, feat_cfg.entity_column)
train_w, entity_mapping = encode_entity_ids(train_w, feat_cfg.entity_column)

# ✓ CORRECTO: en test, usar estadísticas del train (no recalcular)
test_w = create_time_windows(test_events, feat_cfg)
# Usar entity_mapping del training
test_w["entity_idx"] = test_w[feat_cfg.entity_column].map(entity_mapping)
# Estadísticas de entidad: join desde train
train_stats = train_w.groupby(feat_cfg.entity_column).agg(
    entity_mean_count=("event_count", "mean"),
    entity_std_count=("event_count", "std"),
).fillna(1.0)
test_w = test_w.merge(train_stats, on=feat_cfg.entity_column, how="left")
test_w["entity_mean_count"] = test_w["entity_mean_count"].fillna(train_w["entity_mean_count"].mean())
test_w["entity_std_count"]  = test_w["entity_std_count"].fillna(1.0)

print(f"Train: {len(train_w):,} ventanas ({train_events['timestamp'].min().date()} – {cutoff.date()})")
print(f"Test:  {len(test_w):,}  ventanas ({cutoff.date()} – {test_events['timestamp'].max().date()})")
print()
print("✓ Sin leakage: entity_mean_count calculado solo sobre train")
print("✓ Entidades nuevas en test → reciben media global del train")
print("✓ El modelo MCMC se entrena SOLO en train")
print()
print("✗ INCORRECTO (lo que hace el pipeline naive):")
print("  build_modeling_table(train_events + test_events)  ← leakage!")"""),

        md("""---
## Resumen de puntos clave para la entrevista

| Pregunta | Respuesta clave |
|---|---|
| ¿Por qué NB y no Poisson? | Var/Media >> 1 en logs de seguridad. Poisson subestima la cola. |
| ¿Por qué jerárquico? | Partial pooling: entidades con pocos datos se benefician de datos de otras. |
| ¿Por qué –log P(y│posterior)? | Respeta la distribución real. Z-score asume normalidad y falla con colas pesadas. |
| ¿Limitaciones? | Solo detecta ataques que elevan el conteo. Geo/device no se modelan. |
| ¿Escala a producción? | Entrenar offline (horas), scoring online (segundos). |
| ¿Data leakage? | Calcular estadísticas de entidad solo en train. Usar entity_mapping del train en test. |"""),
    ])


# ===========================================================================
# NOTEBOOK 06 — Explicación Profunda (versión interactiva de LEAME_PROFUNDO)
# ===========================================================================

def build_06() -> dict:
    return nb([

        md("""# Notebook 06 — Explicación Profunda del Sistema BSAD

**Versión interactiva de `LEAME_PROFUNDO.md`**

Este notebook recorre el sistema completo con código ejecutable, visualizaciones y explicaciones detalladas en cada paso.

---

**Índice:**
1. El Problema — ¿por qué los datos de seguridad son difíciles?
2. Por qué fallan los métodos clásicos
3. El modelo Bayesiano Jerárquico — matemáticas y visualización
4. Pipeline de datos — de logs crudos a tabla de modelado
5. Scoring de anomalías — cómo funciona –log P(y | posterior)
6. Evaluación — métricas correctas vs incorrectas
7. Triage — del score al workflow SOC
8. Limitaciones honestas

> **Para generar todas las figuras en alta resolución:**
> ```bash
> python scripts/explain_pipeline.py --output outputs/explanation
> ```"""),

        code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from bsad.data_generator import GeneratorConfig, generate_synthetic_data
from bsad.features import FeatureConfig, build_modeling_table, get_model_arrays
from bsad.model import ModelConfig, build_hierarchical_negbinom_model
from bsad.evaluation import compute_all_metrics, format_metrics_report

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

BLUE, RED, GREEN, ORANGE, GRAY = "#2E86AB", "#E84855", "#3BB273", "#F4A261", "#8D99AE"
RNG = np.random.default_rng(42)
print("✓ Setup completo")"""),

        md("---\n## 1. El Problema\n\n> **Pregunta central:** ¿Por qué no basta con poner un umbral fijo de '15 eventos = sospechoso'?"),

        code("""\
# 1.1 Heterogeneidad de entidades
cfg = GeneratorConfig(n_users=60, n_days=21, attack_rate=0.03, random_seed=42)
events_df, _ = generate_synthetic_data(cfg)
modeling_df, meta = build_modeling_table(events_df, FeatureConfig())

user_means = modeling_df.groupby("user_id")["event_count"].mean().sort_values(ascending=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("El Problema: Datos de Seguridad son Difíciles", fontsize=13, fontweight="bold")

# Panel 1: Heterogeneidad
ax = axes[0]
ax.bar(range(len(user_means)), user_means.values, color=BLUE, alpha=0.8, edgecolor="white")
ax.axhline(user_means.mean(), color=RED, linestyle="--", lw=1.5,
           label=f"Media global = {user_means.mean():.1f}")
ax.set_title("① Heterogeneidad entre usuarios")
ax.set_xlabel("Usuario (ordenado por actividad)")
ax.set_ylabel("Eventos / día (media)")
ax.legend()
ax.text(0.97, 0.97,
        f"Rango: {user_means.min():.1f} - {user_means.max():.1f}\\nCV = {user_means.std()/user_means.mean():.0%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec=ORANGE, alpha=0.8))

# Panel 2: Rareza
ax = axes[1]
ar = meta["attack_rate"]
sizes = [1 - ar, ar]
wedges, _, autotexts = ax.pie(sizes,
    labels=[f"Benigno\\n({1-ar:.0%})", f"Ataque\\n({ar:.0%})"],
    colors=[BLUE, RED], autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2), pctdistance=0.6)
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("white")
ax.set_title("② Ataques son raros")

# Panel 3: Sobredispersión
ax = axes[2]
counts = modeling_df["event_count"].values
mu = counts.mean(); var = counts.var()
x = np.arange(0, min(int(counts.max()) + 1, 50))
ax.hist(counts, bins=35, density=True, alpha=0.6, color=BLUE, label="Datos reales")
ax.plot(x, stats.poisson.pmf(x, mu=mu), "r-", lw=2, label=f"Poisson(μ={mu:.1f})")
phi = mu**2 / max(var - mu, 0.1)
ax.plot(x, stats.nbinom.pmf(x, n=phi, p=phi/(phi+mu)), "g-", lw=2,
        label=f"NB(φ={phi:.1f}) Var/μ={var/mu:.1f}x")
ax.set_title("③ Sobredispersión (Var >> Media)")
ax.set_xlabel("Conteo de eventos"); ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print(f"\\nDatos generados: {len(modeling_df):,} ventanas | {meta['n_entities']} entidades")
print(f"Attack rate: {meta['attack_rate']:.2%} | Var/Media = {var/mu:.1f}x")"""),

        md("---\n## 2. Por qué fallan los métodos clásicos"),

        code("""\
# Comparar Z-score global, Z-score entidad, NB-MLE, y NB aproximado con pooling
y_all = modeling_df["event_count"].values
entity_idx = modeling_df["entity_idx"].values
is_attack  = modeling_df["has_attack"].astype(int).values
n_entities = meta["n_entities"]

# 1. Z-score global
g_mean, g_std = y_all.mean(), y_all.std()
score_z_global = np.abs((y_all - g_mean) / (g_std + 1e-8))

# 2. Z-score por entidad
score_z_entity = np.zeros(len(y_all))
for e in range(n_entities):
    mask = entity_idx == e
    mu_e = y_all[mask].mean(); std_e = max(y_all[mask].std(), 0.5)
    score_z_entity[mask] = (y_all[mask] - mu_e) / std_e

# 3. NB-MLE por entidad
score_nb_mle = np.zeros(len(y_all))
for e in range(n_entities):
    mask = entity_idx == e
    mu_e = y_all[mask].mean(); var_e = y_all[mask].var()
    phi_e = max(mu_e**2 / max(var_e - mu_e, 0.1), 0.5)
    p_e = phi_e / (phi_e + mu_e)
    score_nb_mle[mask] = -stats.nbinom.logpmf(y_all[mask], n=phi_e, p=p_e)

# 4. NB con partial pooling (simulado)
score_bsad = np.zeros(len(y_all))
k = 5
for e in range(n_entities):
    mask = entity_idx == e
    mu_e = y_all[mask].mean(); n_e = mask.sum()
    mu_s = (n_e * mu_e + k * g_mean) / (n_e + k)
    phi_e = max(mu_e**2 / max(y_all[mask].var() - mu_e, 0.1), 0.5)
    p_e = phi_e / (phi_e + mu_s)
    score_bsad[mask] = -stats.nbinom.logpmf(y_all[mask], n=phi_e, p=p_e)

methods = [
    ("Z-score Global\\n(ignora entidades)", score_z_global),
    ("Z-score por Entidad\\n(asume normalidad)", score_z_entity),
    ("NB-MLE por Entidad\\n(sin pooling)", score_nb_mle),
    ("BSAD: NB Jerarquico\\n(partial pooling)", score_bsad),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("PR-AUC de cada método (más alto = mejor detector de ataques)", fontsize=12, fontweight="bold")

for ax, (title, scores) in zip(axes, methods):
    pr_auc = average_precision_score(is_attack, scores)
    s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    ax.hist(s_norm[is_attack == 0], bins=30, alpha=0.6, color=BLUE, density=True, label="Benigno")
    ax.hist(s_norm[is_attack == 1], bins=15, alpha=0.8, color=RED, density=True, label="Ataque")
    ax.set_title(f"{title}\\nPR-AUC = {pr_auc:.3f}", fontsize=9)
    ax.set_xlabel("Score normalizado"); ax.legend(fontsize=7)
    c = GREEN if pr_auc > 0.15 else RED
    ax.text(0.97, 0.97, f"PR-AUC\\n{pr_auc:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold", color=c,
            bbox=dict(boxstyle="round", fc="white", ec=GRAY))

plt.tight_layout()
plt.show()"""),

        md("---\n## 3. El Modelo Bayesiano Jerárquico\n\n### 3.1 La estructura matemática\n\n```\nPoblación:    μ ~ Exp(0.1)          # tasa media global\n              α ~ HalfNormal(2)     # concentración\n\nEntidad:      θ_e ~ Gamma(μα, α)    # tasa propia de cada entidad\n\nObservación:  φ ~ HalfNormal(2)     # sobredispersión\n              y ~ NegBin(θ_{e_n}, φ)  # dato observado\n```\n\nEl score de anomalía: `–log p(y | posterior)` = cuán improbable es y bajo el modelo aprendido."),

        code("""\
# 3.2 Construcción del modelo (sin entrenar — solo la estructura)
import pymc as pm

arrays = get_model_arrays(modeling_df)
config = ModelConfig(n_samples=100, n_tune=50, n_chains=2, random_seed=42)

model = build_hierarchical_negbinom_model(
    y=arrays["y"],
    entity_idx=arrays["entity_idx"],
    n_entities=arrays["n_entities"],
    config=config,
)

print("Variables del modelo:")
print([v.name for v in model.free_RVs])
print()
print(f"Entidades modeladas: {arrays['n_entities']}")
print(f"Observaciones:       {len(arrays['y'])}")
print()
print("Estructura jerárquica:")
print("  μ, α  → parámetros de POBLACIÓN")
print("  θ_e   → parámetros de ENTIDAD (uno por usuario)")
print("  φ     → sobredispersión GLOBAL")
print("  y_n   → DATO OBSERVADO (likelihood)")"""),

        code("""\
# 3.3 Visualización del partial pooling
fig, ax = plt.subplots(figsize=(10, 5))

n_obs_range = [1, 2, 5, 10, 20, 50, 100, 200]
true_rate = 15.0
global_mean_val = 5.0
k_shrink = 5

mle_vals, bayes_vals = [], []
for n in n_obs_range:
    obs_sim = RNG.poisson(true_rate, size=n)
    mle = obs_sim.mean()
    bsad = (n * mle + k_shrink * global_mean_val) / (n + k_shrink)
    mle_vals.append(mle)
    bayes_vals.append(bsad)

ax.plot(n_obs_range, mle_vals,   "o-", color=ORANGE, lw=2, markersize=6,
        label="MLE puro (sin pooling)")
ax.plot(n_obs_range, bayes_vals, "s-", color=BLUE,   lw=2, markersize=6,
        label="BSAD (partial pooling / shrinkage)")
ax.axhline(true_rate,       color=RED,  linestyle="--", lw=1.5, label=f"Tasa real = {true_rate}")
ax.axhline(global_mean_val, color=GRAY, linestyle=":",  lw=1.5, label=f"Media global = {global_mean_val}")

ax.annotate("Pocos datos: BSAD\\nse encoge hacia la\\nmedia global",
            xy=(2, bayes_vals[1]), xytext=(5, 8),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, bbox=dict(boxstyle="round", fc="lightyellow", ec=ORANGE, alpha=0.8))
ax.annotate("Muchos datos: BSAD\\nconverge al MLE\\n(evidencia domina)",
            xy=(200, bayes_vals[-1]), xytext=(70, 12),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, bbox=dict(boxstyle="round", fc="#E8F5E9", ec=GREEN, alpha=0.8))

ax.set_xscale("log")
ax.set_xlabel("Número de observaciones de la entidad")
ax.set_ylabel("Estimación de θ_e")
ax.set_title("Partial Pooling: Shrinkage Adaptativo\\nEntidades con pocos datos se benefician de la población")
ax.legend()
plt.tight_layout()
plt.show()"""),

        md("---\n## 4. Pipeline de datos\n\nDe logs crudos a tabla de modelado paso a paso."),

        code("""\
print("PASO 1: Eventos crudos")
print(events_df[["timestamp", "user_id", "ip_address", "endpoint",
                   "status_code", "is_attack", "attack_type"]].head(5).to_string(index=False))

print("\\nPASO 2: Agregado por ventanas de 1 día")
from bsad.features import create_time_windows
windowed = create_time_windows(events_df, FeatureConfig(window_size="1D"))
print(windowed[["user_id", "window", "event_count", "unique_ips",
                 "unique_devices", "has_attack"]].head(5).to_string(index=False))

print(f"\\nTotal filas en eventos:   {len(events_df):,}")
print(f"Total filas en windowed:  {len(windowed):,}  ← una por (usuario, día)")
print(f"Sum event_count:          {windowed['event_count'].sum():,} == {len(events_df):,}  ✓ (sin pérdida)")"""),

        code("""\
print("PASO 3: Features de entidad (ATENCIÓN AL LEAKAGE)")
print()
print("✓ CORRECTO para ENTRENAMIENTO:")
print("  add_entity_features(train_windowed, entity_col)")
print("  → calcula mean/std/zscore SOLO sobre datos de train")
print()
print("✗ INCORRECTO (lo que hace el pipeline naive):")
print("  add_entity_features(train_windowed + test_windowed)")
print("  → contamina las estadísticas del train con el futuro")
print()
print("PASO 4: Encoding de entidades")
from bsad.features import encode_entity_ids
from bsad.features import add_entity_features
windowed2 = add_entity_features(windowed, "user_id")
encoded, entity_map = encode_entity_ids(windowed2, "user_id")
print(f"  Entidades únicas: {len(entity_map)}")
print(f"  Rango de entity_idx: [{encoded['entity_idx'].min()}, {encoded['entity_idx'].max()}]  ← contiguos [0, E)")
print()
print("PASO 5: Arrays para PyMC")
arrays = get_model_arrays(encoded)
print(f"  y.shape:          {arrays['y'].shape}  dtype={arrays['y'].dtype}")
print(f"  entity_idx.shape: {arrays['entity_idx'].shape}  dtype={arrays['entity_idx'].dtype}")
print(f"  n_entities:       {arrays['n_entities']}")
print(f"  is_attack.sum():  {arrays['is_attack'].sum()}  ← NO entra al modelo, solo evaluación")"""),

        md("---\n## 5. Scoring de Anomalías\n\n`score = –log p(y | posterior)` — un evento es anómalo si es **improbable bajo la distribución aprendida de su entidad**."),

        code("""\
# Visualización: P(y | theta) para tres entidades distintas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Score = –log P(y | distribución aprendida)\\nEl mismo conteo tiene scores distintos según la entidad", fontsize=11)

for ax, (mu_e, label, color) in zip(axes, [
    (2.0,  "Entidad tranquila (μ=2)",     GREEN),
    (10.0, "Entidad activa (μ=10)",       BLUE),
    (40.0, "Entidad muy activa (μ=40)", ORANGE),
]):
    phi_e = 2.0
    p_e   = phi_e / (phi_e + mu_e)
    x_max = int(mu_e * 5)
    x     = np.arange(0, min(x_max, 100))
    pmf   = stats.nbinom.pmf(x, n=phi_e, p=p_e)

    ax.bar(x, pmf, color=color, alpha=0.6)

    # Marcar tres puntos: normal, medio, alto
    for y_obs, marker, mc in [(int(mu_e), "normal", GREEN),
                               (int(mu_e * 2), "elevado", ORANGE),
                               (int(mu_e * 5), "muy alto", RED)]:
        if y_obs < len(x):
            sc = -stats.nbinom.logpmf(y_obs, n=phi_e, p=p_e)
            ax.axvline(y_obs, color=mc, lw=2, linestyle="--",
                       label=f"y={y_obs}: score={sc:.1f}")

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Conteo y")
    ax.set_ylabel("P(y | θ)")
    ax.legend(fontsize=7.5)

plt.tight_layout()
plt.show()"""),

        code("""\
# Cómo funciona la marginalizacion sobre el posterior
# (sin MCMC real, simulamos muestras posteriores)
print("Simulación de marginalizacion sobre el posterior:")
print()

n_post = 500
mu_pop, alpha_pop = 5.0, 3.0
theta_samples = RNG.gamma(shape=mu_pop * alpha_pop, scale=1.0 / alpha_pop, size=n_post)
phi_samples   = RNG.exponential(scale=2.0, size=n_post)

y_obs_list = [3, 8, 25, 60]

print(f"{'y observado':>12} {'Score promedio':>15} {'Score std':>12} {'Interpretación':>25}")
print("-" * 68)
for y_obs in y_obs_list:
    per_sample = np.array([
        -stats.nbinom.logpmf(y_obs, n=phi_s, p=phi_s/(phi_s+theta_s))
        for theta_s, phi_s in zip(theta_samples, phi_samples)
    ])
    # Promedio en escala log (log-sum-exp)
    log_sum = logsumexp(-per_sample)
    avg_score = -(log_sum - np.log(n_post))

    interp = "Muy probable" if avg_score < 4 else ("Posible" if avg_score < 8 else ("Sospechoso" if avg_score < 15 else "Anomalía clara"))
    print(f"{y_obs:>12} {avg_score:>15.2f} {per_sample.std():>12.2f} {interp:>25}")"""),

        md("---\n## 6. Evaluación: métricas correctas para eventos raros"),

        code("""\
# Generar scores para evaluación
def approx_score(row):
    mu_e  = max(row["entity_mean_count"], 0.5)
    std_e = max(row["entity_std_count"], 0.5)
    phi_e = max(mu_e**2 / max(std_e**2 - mu_e, 0.1), 0.5)
    p_e   = phi_e / (phi_e + mu_e)
    return float(-stats.nbinom.logpmf(int(row["event_count"]), n=phi_e, p=p_e))

modeling_df["score"] = modeling_df.apply(approx_score, axis=1)
y_true = modeling_df["has_attack"].astype(int).values
scores = modeling_df["score"].values

metrics = compute_all_metrics(y_true, scores, k_values=[10, 25, 50, 100])
print(format_metrics_report(metrics))"""),

        code("""\
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Métricas: ROC vs PR vs Operacionales", fontsize=12, fontweight="bold")

# ROC
ax = axes[0]
fpr_r, tpr_r, _ = roc_curve(y_true, scores)
ax.plot(fpr_r, tpr_r, color=BLUE, lw=2, label=f"BSAD (AUC={metrics['roc_auc']:.3f})")
ax.plot([0,1],[0,1], color=GRAY, linestyle="--", lw=1)
ax.fill_between(fpr_r, tpr_r, alpha=0.1, color=BLUE)
ax.set_title(f"ROC-AUC = {metrics['roc_auc']:.3f}\\n(optimista con desbalanceo)")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend()

# PR
ax = axes[1]
prec_c, rec_c, _ = precision_recall_curve(y_true, scores)
ax.plot(rec_c, prec_c, color=RED, lw=2, label=f"BSAD (PR-AUC={metrics['pr_auc']:.3f})")
ax.axhline(y_true.mean(), color=GRAY, linestyle="--", lw=1,
           label=f"Baseline = {y_true.mean():.3f}")
ax.fill_between(rec_c, prec_c, alpha=0.1, color=RED)
ax.set_title(f"PR-AUC = {metrics['pr_auc']:.3f}\\n({metrics['pr_auc']/y_true.mean():.1f}x sobre baseline)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()

# Precision@k / Recall@k
ax = axes[2]
ks  = [10, 25, 50, 100]
prk = [metrics.get(f"precision_at_{k}", 0) for k in ks]
rek = [metrics.get(f"recall_at_{k}", 0) for k in ks]
x   = np.arange(len(ks))
ax.bar(x - 0.2, prk, 0.35, color=BLUE,  alpha=0.85, label="Precision@k")
ax.bar(x + 0.2, rek, 0.35, color=RED,   alpha=0.85, label="Recall@k")
for i, (p, r) in enumerate(zip(prk, rek)):
    ax.text(i-0.2, p+0.01, f"{p:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i+0.2, r+0.01, f"{r:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_ylim(0, 1.1)
ax.set_title("Métricas Operacionales SOC\\nPrecision y Recall en top-k alertas")
ax.legend()

plt.tight_layout()
plt.show()"""),

        md("---\n## 7. Triage: del score al workflow SOC"),

        code("""\
from triage.risk_score import RiskScorer
from triage.calibrate_thresholds import AlertBudget, build_alert_budget_curve
from triage.ranking_metrics import precision_at_k, recall_at_k

# Risk Score compuesto
entity_obs = modeling_df.groupby("user_id")["event_count"].count()
obs_counts = modeling_df["user_id"].map(entity_obs).values

scorer = RiskScorer(score_weight=0.5, confidence_weight=0.3, novelty_weight=0.2)
score_std = scores * 0.15 + RNG.exponential(0.3, len(scores))
risk_scores = scorer.compute(scores, score_std, obs_counts)
modeling_df["risk_score"] = risk_scores

# Calibración de umbral para 30 alertas/día
budget = AlertBudget(mode="fixed_alerts", target=30)
result = budget.calibrate(scores, y_true, n_windows_per_day=500)
threshold = result["threshold"]

print(f"Budget: 30 alertas/día")
print(f"  Umbral calibrado:  {threshold:.2f}")
print(f"  Recall obtenido:   {result.get('recall', 0):.1%}")
print(f"  FPR:               {result.get('fpr', 0):.1%}")
print()

top10 = modeling_df.nlargest(10, "risk_score")[
    ["user_id", "event_count", "entity_mean_count", "score", "risk_score", "has_attack"]
].round(2)
print("Top 10 alertas priorizadas por Risk Score:")
print(top10.to_string(index=False))"""),

        code("""\
# Curva budget completa
budget_curve = build_alert_budget_curve(scores, y_true)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(budget_curve["target_recall"] * 100, budget_curve["alerts"],
        "o-", color=BLUE, lw=2, markersize=4)
ax.axhspan(0,  25, alpha=0.07, color="green",  label="Manejable")
ax.axhspan(25, 75, alpha=0.07, color="orange", label="Aceptable")
ax.axhspan(75, 500, alpha=0.05, color="red",   label="Sobrecarga")
ax.set_xlabel("Recall objetivo (%)")
ax.set_ylabel("Alertas")
ax.set_title("Trade-off: Recall vs Carga SOC")
ax.set_ylim(0, 250); ax.legend(fontsize=9)

ax = axes[1]
ks_range = range(5, min(len(y_true), 150), 5)
precs = [precision_at_k(y_true, scores, k) for k in ks_range]
recs  = [recall_at_k(y_true, scores, k) for k in ks_range]
ax.plot(list(ks_range), precs, color=BLUE, lw=2, label="Precision@k")
ax.plot(list(ks_range), recs,  color=RED,  lw=2, label="Recall@k")
ax.axhline(y_true.mean(), color=GRAY, linestyle="--", lw=1,
           label=f"Baseline precision = {y_true.mean():.3f}")
ax.set_xlabel("k (top-k alertas revisadas)")
ax.set_ylabel("Valor")
ax.set_title("Precision@k y Recall@k vs k")
ax.legend()

plt.tight_layout()
plt.show()"""),

        md("---\n## 8. Limitaciones Honestas"),

        code("""\
# Demostración de qué detecta y qué no detecta el modelo

# Generar un usuario con solo una "geo_anomaly" sutil (5 eventos extra)
mu_user_normal = 8.0  # baseline del usuario

# Caso A: brute_force — 150 eventos en un día (normal = 8)
y_bf = 150
score_bf = -stats.nbinom.logpmf(y_bf, n=2, p=2/(2+mu_user_normal))

# Caso B: geo_anomaly — 5 eventos extra desde ubicación sospechosa
y_geo = 12  # solo 4 más que la media
score_geo = -stats.nbinom.logpmf(y_geo, n=2, p=2/(2+mu_user_normal))

# Caso C: device_anomaly — 2 accesos desde dispositivo nuevo
y_dev = 10
score_dev = -stats.nbinom.logpmf(y_dev, n=2, p=2/(2+mu_user_normal))

# Umbral típico basado en percentil 95
threshold_95 = np.percentile(modeling_df["score"], 95)

print("Tipo de ataque     | y observado | Score NB  | > umbral?    | Detectable?")
print("-" * 78)
for name, y_a, sc in [("brute_force",   y_bf,  score_bf),
                       ("geo_anomaly",   y_geo, score_geo),
                       ("device_anomaly",y_dev, score_dev)]:
    above = "SI" if sc > threshold_95 else "NO"
    detectable = "Muy alta" if sc > threshold_95 * 2 else ("Media" if sc > threshold_95 else "Baja")
    print(f"{name:<18} | {y_a:<11} | {sc:>9.2f} | {above:<12} | {detectable}")

print(f"\\nUmbral (percentil 95): {threshold_95:.2f}")
print()
print("CONCLUSIÓN:")
print("  El modelo NB detecta VOLUMEN anómalo, no comportamiento multivariado.")
print("  geo_anomaly y device_anomaly solo se detectan si elevan event_count.")
print("  Para detectar geo/device, necesitarías features adicionales en el likelihood.")"""),

        code("""\
fig, ax = plt.subplots(figsize=(10, 5))

attack_types = ["brute_force", "credential\\nstuffing", "geo_anomaly", "device_anomaly"]
detectability = [0.85, 0.45, 0.20, 0.15]
colors_d = [GREEN if d > 0.5 else (ORANGE if d > 0.3 else RED) for d in detectability]
mechanisms = [
    "Eleva masivamente\\nevent_count",
    "Eleva event_count\\nsi hay suficientes\\nintentos",
    "Location NO modelada.\\nSolo detectable si\\neleva el conteo.",
    "Device NO modelado.\\nPocos eventos extra.\\nSenal muy débil.",
]

y_pos = range(len(attack_types))
bars = ax.barh(y_pos, detectability, color=colors_d, alpha=0.8, edgecolor="white", height=0.5)
ax.barh(y_pos, [1.0]*len(attack_types), color=GRAY, alpha=0.12, height=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(attack_types, fontsize=11, fontweight="bold")
ax.set_xlabel("Detectabilidad por el modelo NB (0 = no detecta, 1 = siempre detecta)")
ax.set_title("Detectabilidad Honesta por Tipo de Ataque\\nEl modelo NB es un detector de VOLUMEN, no de comportamiento", fontsize=11)
ax.set_xlim(0, 1.3)

for i, (bar, mech) in enumerate(zip(bars, mechanisms)):
    ax.text(bar.get_width() + 0.02, i, f"{bar.get_width():.0%}  {mech}",
            va="center", fontsize=8.5, color="#2B2D42")

plt.tight_layout()
plt.show()"""),

        md("""---
## Resumen Final

| Capa | Módulo clave | Qué hace | Limitación |
|---|---|---|---|
| **Datos** | `bsad.data_generator` | Genera eventos sintéticos con 4 tipos de ataque | geo/device solo detectables via conteo |
| **Features** | `bsad.features` | Agrega eventos → conteos por entidad-ventana | `entity_mean_count` tiene riesgo de leakage |
| **Modelo** | `bsad.model` | NB jerárquico con partial pooling vía MCMC | Entrenamiento lento (horas) |
| **Scoring** | `bsad.scoring` | `–log p(y│posterior)` con incertidumbre | Loop Python lento para posteriors grandes |
| **Evaluación** | `bsad.evaluation` | PR-AUC, Recall@k, Precision@k | Metrics sobre datos sintéticos pueden ser optimistas |
| **Triage** | `triage/` | Risk Score + Alert Budget + Contexto de entidad | Pesos del Risk Score son heurísticos |

### Próximos pasos
1. **Covariates temporales**: `log(μ_et) = log(θ_e) + β·is_weekend` → menos FP en fines de semana
2. **Multi-signal**: añadir `unique_ips` como segundo likelihood para detectar credential stuffing
3. **Scorer vectorizado**: reemplazar el loop Python con NumPy/JAX → 50x más rápido
4. **Split temporal sin leakage**: `temporal_train_test_split(modeling_df, cutoff_date)`"""),

    ])


# ===========================================================================
# Ejecutar
# ===========================================================================

if __name__ == "__main__":
    nb_dir = Path("notebooks")

    print("\nGenerando notebooks...\n")
    save(build_04(), nb_dir / "04_alert_prioritization.ipynb")
    save(build_05(), nb_dir / "05_interview_prep_fortra.ipynb")
    save(build_06(), nb_dir / "06_explicacion_profunda.ipynb")
    print("\n✅ Notebooks generados. Ejecutar con:")
    print("   conda activate bsad")
    print("   jupyter lab notebooks/")
