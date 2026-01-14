"""
Streamlit dashboard for Bayesian Security Anomaly Detection.

Run with: streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="BSAD - Anomaly Detection Dashboard",
    page_icon="🔒",
    layout="wide",
)


def load_data():
    """Load scored data and metrics if available."""
    data = {}

    scores_path = Path("outputs/scores.parquet")
    if scores_path.exists():
        data["scores"] = pd.read_parquet(scores_path)

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    return data


def main():
    st.title("🔒 Bayesian Security Anomaly Detection")
    st.markdown("---")

    # Load data
    data = load_data()

    if not data:
        st.warning("No results found. Run `make demo` first to generate data.")
        st.code("make demo", language="bash")
        return

    # Sidebar filters
    st.sidebar.header("Filters")

    if "scores" in data:
        scores_df = data["scores"]

        # Score threshold filter
        min_score = float(scores_df["anomaly_score"].min())
        max_score = float(scores_df["anomaly_score"].max())
        score_threshold = st.sidebar.slider(
            "Minimum Anomaly Score",
            min_value=min_score,
            max_value=max_score,
            value=min_score,
        )

        # Attack filter
        show_attacks_only = st.sidebar.checkbox("Show attacks only", value=False)

        # Apply filters
        filtered_df = scores_df[scores_df["anomaly_score"] >= score_threshold]
        if show_attacks_only:
            filtered_df = filtered_df[filtered_df["has_attack"]]

    # Main content
    col1, col2, col3, col4 = st.columns(4)

    if "metrics" in data:
        metrics = data["metrics"]
        col1.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        col2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        col3.metric("Recall@50", f"{metrics.get('recall_at_50', 0):.3f}")
        col4.metric("Recall@100", f"{metrics.get('recall_at_100', 0):.3f}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Top Anomalies", "📊 Score Distribution", "📈 Metrics"])

    with tab1:
        st.subheader("Top Anomalies")

        if "scores" in data:
            n_display = st.slider("Number of anomalies to display", 10, 100, 25)

            display_cols = [
                "anomaly_rank",
                "user_id",
                "window",
                "event_count",
                "anomaly_score",
                "has_attack",
                "attack_type",
            ]
            display_cols = [c for c in display_cols if c in filtered_df.columns]

            top_df = filtered_df.head(n_display)[display_cols]

            # Highlight attacks
            def highlight_attacks(row):
                if row.get("has_attack", False):
                    return ["background-color: #ffcccb"] * len(row)
                return [""] * len(row)

            st.dataframe(
                top_df.style.apply(highlight_attacks, axis=1),
                use_container_width=True,
            )

            # Summary stats
            n_attacks_in_top = filtered_df.head(n_display)["has_attack"].sum()
            st.info(f"Attacks in top {n_display}: {n_attacks_in_top} ({n_attacks_in_top/n_display:.1%})")

    with tab2:
        st.subheader("Anomaly Score Distribution")

        if "scores" in data:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))

            benign = scores_df[~scores_df["has_attack"]]["anomaly_score"]
            attacks = scores_df[scores_df["has_attack"]]["anomaly_score"]

            ax.hist(benign, bins=50, alpha=0.7, label=f"Benign (n={len(benign)})", color="steelblue")
            ax.hist(attacks, bins=50, alpha=0.7, label=f"Attack (n={len(attacks)})", color="crimson")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Count")
            ax.legend()
            ax.set_title("Score Distribution by Class")

            st.pyplot(fig)

    with tab3:
        st.subheader("Evaluation Metrics")

        if "metrics" in data:
            metrics = data["metrics"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Dataset Statistics")
                st.write(f"- Total observations: {metrics.get('n_observations', 0):,}")
                st.write(f"- Attack windows: {metrics.get('n_positives', 0):,}")
                st.write(f"- Attack rate: {metrics.get('attack_rate', 0):.2%}")

            with col2:
                st.markdown("### Performance Metrics")
                st.write(f"- PR-AUC: {metrics.get('pr_auc', 0):.3f}")
                st.write(f"- ROC-AUC: {metrics.get('roc_auc', 0):.3f}")

                st.markdown("#### Recall@K")
                for k in [10, 25, 50, 100]:
                    key = f"recall_at_{k}"
                    if key in metrics:
                        st.write(f"- Recall@{k}: {metrics[key]:.3f}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        Bayesian Security Anomaly Detection |
        <a href='https://github.com/yourusername/bayesian-security-anomaly-detection'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
