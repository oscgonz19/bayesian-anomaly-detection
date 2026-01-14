"""
Pipeline orchestrator for BSAD.

The Pipeline class calls steps in order and manages state.
This is the ONE place that orchestrates the entire flow.
"""

from dataclasses import dataclass, field
from pathlib import Path

import arviz as az
import pandas as pd
from rich.console import Console
from rich.table import Table

from bsad import io, steps
from bsad.config import Settings


@dataclass
class PipelineState:
    """
    Holds all intermediate data between pipeline steps.

    This makes the pipeline transparent - you can inspect any artifact.
    """
    events_df: pd.DataFrame | None = None
    attacks_df: pd.DataFrame | None = None
    modeling_df: pd.DataFrame | None = None
    metadata: dict | None = None
    arrays: dict | None = None
    trace: az.InferenceData | None = None
    scored_df: pd.DataFrame | None = None
    metrics: dict | None = None
    plots: dict = field(default_factory=dict)


class Pipeline:
    """
    Main pipeline orchestrator.

    Usage:
        settings = Settings(n_entities=100, n_days=14)
        pipeline = Pipeline(settings)
        pipeline.run_demo()
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.state = PipelineState()
        self.console = Console()

    # =========================================================================
    # Main Pipeline Methods
    # =========================================================================

    def run_demo(self) -> PipelineState:
        """
        Run the complete pipeline: generate -> train -> score -> evaluate -> report.

        This is the main entry point for the demo.
        """
        self._print_header("BAYESIAN SECURITY ANOMALY DETECTION - DEMO")
        self.settings.ensure_dirs()

        # Step 1: Generate data
        self._print_step(1, 5, "Generating synthetic security events")
        self.state.events_df, self.state.attacks_df = steps.generate_data(self.settings)
        io.save_events(self.state.events_df, self.state.attacks_df, self.settings.events_path)
        self.console.print(f"  Generated {len(self.state.events_df):,} events")
        self.console.print(f"  Attack events: {self.state.events_df['is_attack'].sum():,}")

        # Step 2: Build features
        self._print_step(2, 5, "Building feature table")
        self.state.modeling_df, self.state.metadata = steps.build_features(self.state.events_df, self.settings)
        self.state.arrays = steps.get_model_arrays(self.state.modeling_df)
        io.save_parquet(self.state.modeling_df, self.settings.modeling_table_path)
        self.console.print(f"  Entities: {self.state.metadata['n_entities']}")
        self.console.print(f"  Windows: {self.state.metadata['n_windows']}")

        # Step 3: Train model
        self._print_step(3, 5, "Training hierarchical Bayesian model")
        self.console.print(f"  Sampling {self.settings.n_samples} posterior draws...")
        self.state.trace = steps.train_model(self.state.arrays, self.settings)
        io.save_model(self.state.trace, self.settings.model_path)
        diagnostics = steps.get_diagnostics(self.state.trace)
        self.console.print(f"  R-hat: {diagnostics['r_hat_max']:.3f}, Divergences: {diagnostics['divergences']}")

        # Step 4: Score
        self._print_step(4, 5, "Scoring observations")
        scores = steps.compute_scores(
            self.state.arrays["y"],
            self.state.trace,
            self.state.arrays["entity_idx"],
        )
        intervals = steps.compute_intervals(self.state.trace, self.state.arrays["entity_idx"])
        self.state.scored_df = steps.create_scored_df(self.state.modeling_df, scores, intervals)
        io.save_parquet(self.state.scored_df, self.settings.scores_path)
        self.console.print(f"  Scored {len(self.state.scored_df):,} entity-windows")

        # Step 5: Evaluate
        self._print_step(5, 5, "Evaluating performance")
        self.state.metrics = steps.evaluate(self.state.scored_df)
        io.save_json(self.state.metrics, self.settings.metrics_path)
        self.console.print(f"  PR-AUC: {self.state.metrics['pr_auc']:.3f}")
        self.console.print(f"  Recall@50: {self.state.metrics.get('recall_at_50', 0):.3f}")
        self.console.print(f"  Recall@100: {self.state.metrics.get('recall_at_100', 0):.3f}")

        # Generate plots
        self.console.print("\n  Generating plots...")
        self.state.plots = steps.create_plots(
            self.state.scored_df,
            self.state.metrics,
            self.state.trace,
            self.settings.plots_dir,
        )

        # Report
        self._print_report()

        return self.state

    def run_train(self, events_path: Path | None = None) -> PipelineState:
        """
        Train model from existing event data.

        Use this when you have real data or pre-generated synthetic data.
        """
        self._print_header("BSAD - TRAIN MODEL")
        self.settings.ensure_dirs()

        # Load data
        events_path = events_path or self.settings.events_path
        self.console.print(f"Loading events from {events_path}")
        self.state.events_df = io.load_parquet(events_path)
        self.console.print(f"  Loaded {len(self.state.events_df):,} events")

        # Build features
        self.console.print("Building features...")
        self.state.modeling_df, self.state.metadata = steps.build_features(self.state.events_df, self.settings)
        self.state.arrays = steps.get_model_arrays(self.state.modeling_df)
        io.save_parquet(self.state.modeling_df, self.settings.modeling_table_path)

        # Train
        self.console.print(f"Training model ({self.settings.n_samples} samples)...")
        self.state.trace = steps.train_model(self.state.arrays, self.settings)
        io.save_model(self.state.trace, self.settings.model_path)

        diagnostics = steps.get_diagnostics(self.state.trace)
        self.console.print(f"  R-hat: {diagnostics['r_hat_max']:.3f}")
        self.console.print(f"  Model saved to {self.settings.model_path}")

        return self.state

    def run_score(self, model_path: Path | None = None) -> PipelineState:
        """
        Score observations using a trained model.

        Use this for batch scoring with a pre-trained model.
        """
        self._print_header("BSAD - SCORE ANOMALIES")
        self.settings.ensure_dirs()

        # Load model
        model_path = model_path or self.settings.model_path
        self.console.print(f"Loading model from {model_path}")
        self.state.trace = io.load_model(model_path)

        # Load modeling table
        self.console.print(f"Loading data from {self.settings.modeling_table_path}")
        self.state.modeling_df = io.load_parquet(self.settings.modeling_table_path)
        self.state.arrays = steps.get_model_arrays(self.state.modeling_df)

        # Score
        self.console.print("Computing anomaly scores...")
        scores = steps.compute_scores(
            self.state.arrays["y"],
            self.state.trace,
            self.state.arrays["entity_idx"],
        )
        intervals = steps.compute_intervals(self.state.trace, self.state.arrays["entity_idx"])
        self.state.scored_df = steps.create_scored_df(self.state.modeling_df, scores, intervals)
        io.save_parquet(self.state.scored_df, self.settings.scores_path)

        self.console.print(f"  Scored {len(self.state.scored_df):,} observations")
        self.console.print(f"  Saved to {self.settings.scores_path}")

        # Show top anomalies
        self._print_top_anomalies(5)

        return self.state

    def run_evaluate(self) -> PipelineState:
        """
        Evaluate scored results against ground truth.
        """
        self._print_header("BSAD - EVALUATE")

        # Load scores
        self.console.print(f"Loading scores from {self.settings.scores_path}")
        self.state.scored_df = io.load_parquet(self.settings.scores_path)

        # Evaluate
        self.console.print("Computing metrics...")
        self.state.metrics = steps.evaluate(self.state.scored_df)
        io.save_json(self.state.metrics, self.settings.metrics_path)

        # Print report
        self._print_metrics()

        return self.state

    # =========================================================================
    # Printing Helpers
    # =========================================================================

    def _print_header(self, title: str) -> None:
        self.console.print()
        self.console.print("=" * 60)
        self.console.print(title)
        self.console.print("=" * 60)
        self.console.print()

    def _print_step(self, step: int, total: int, description: str) -> None:
        self.console.print(f"\n[bold]Step {step}/{total}:[/bold] {description}")

    def _print_report(self) -> None:
        """Print final summary report."""
        self.console.print()
        self.console.print("=" * 60)
        self.console.print("DEMO COMPLETE")
        self.console.print("=" * 60)
        self.console.print()
        self.console.print("[bold]Generated Artifacts:[/bold]")
        self.console.print(f"  Events:   {self.settings.events_path}")
        self.console.print(f"  Model:    {self.settings.model_path}")
        self.console.print(f"  Scores:   {self.settings.scores_path}")
        self.console.print(f"  Metrics:  {self.settings.metrics_path}")
        self.console.print(f"  Plots:    {self.settings.plots_dir}")
        self.console.print()
        self._print_top_anomalies(5)

    def _print_top_anomalies(self, n: int = 5) -> None:
        """Print top N anomalies as a table."""
        if self.state.scored_df is None:
            return

        self.console.print(f"[bold]Top {n} Anomalies:[/bold]")
        table = Table()
        table.add_column("Entity", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("Truth", style="bold")

        for _, row in self.state.scored_df.head(n).iterrows():
            truth = "[red]ATTACK[/red]" if row["has_attack"] else "[green]benign[/green]"
            table.add_row(
                str(row.get("user_id", row.get("entity_idx", "?"))),
                f"{row['anomaly_score']:.2f}",
                str(int(row["event_count"])),
                truth,
            )

        self.console.print(table)

    def _print_metrics(self) -> None:
        """Print evaluation metrics."""
        if self.state.metrics is None:
            return

        m = self.state.metrics
        self.console.print()
        self.console.print("[bold]Evaluation Metrics:[/bold]")
        self.console.print(f"  PR-AUC:       {m['pr_auc']:.3f}")
        self.console.print(f"  ROC-AUC:      {m['roc_auc']:.3f}")
        self.console.print(f"  Recall@50:    {m.get('recall_at_50', 0):.3f}")
        self.console.print(f"  Recall@100:   {m.get('recall_at_100', 0):.3f}")
        self.console.print(f"  Attack rate:  {m['attack_rate']:.2%}")
