"""
CLI for BSAD pipeline.

Simple Typer CLI that instantiates Pipeline and calls methods.
"""

from pathlib import Path

import typer

from bsad.config import Settings
from bsad.pipeline import Pipeline

app = typer.Typer(
    name="bsad",
    help="Bayesian Security Anomaly Detection CLI",
    add_completion=False,
)


@app.command()
def demo(
    output_dir: Path = typer.Option(
        Path("outputs"),
        "--output-dir", "-o",
        help="Output directory for all artifacts",
    ),
    n_entities: int = typer.Option(
        200,
        "--n-entities", "-n",
        help="Number of entities (users) to generate",
    ),
    n_days: int = typer.Option(
        30,
        "--n-days", "-d",
        help="Number of days of data to generate",
    ),
    samples: int = typer.Option(
        2000,
        "--samples", "-s",
        help="Number of MCMC posterior samples",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """
    Run complete demo pipeline: generate -> train -> score -> evaluate.

    This is the main entry point for trying out the system.
    """
    settings = Settings(
        output_dir=output_dir,
        data_dir=output_dir / "data",
        n_entities=n_entities,
        n_days=n_days,
        n_samples=samples,
        random_seed=seed,
    )

    pipeline = Pipeline(settings)
    pipeline.run_demo()


@app.command()
def train(
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        help="Path to events parquet file",
    ),
    output: Path = typer.Option(
        Path("outputs/model.nc"),
        "--output", "-o",
        help="Output path for trained model",
    ),
    samples: int = typer.Option(
        2000,
        "--samples", "-s",
        help="Number of MCMC posterior samples",
    ),
    tune: int = typer.Option(
        1000,
        "--tune", "-t",
        help="Number of tuning samples",
    ),
    chains: int = typer.Option(
        2,
        "--chains", "-c",
        help="Number of MCMC chains",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
) -> None:
    """
    Train model on existing event data.

    Use this when you have real data or pre-generated synthetic data.
    """
    settings = Settings(
        output_dir=output.parent,
        data_dir=input_path.parent,
        n_samples=samples,
        n_tune=tune,
        n_chains=chains,
        random_seed=seed,
    )

    pipeline = Pipeline(settings)
    pipeline.run_train(events_path=input_path)


@app.command()
def score(
    model_path: Path = typer.Option(
        Path("outputs/model.nc"),
        "--model", "-m",
        help="Path to trained model file",
    ),
    output: Path = typer.Option(
        Path("outputs/scores.parquet"),
        "--output", "-o",
        help="Output path for scores",
    ),
) -> None:
    """
    Score observations using a trained model.

    Use this for batch scoring with a pre-trained model.
    """
    settings = Settings(output_dir=output.parent)

    pipeline = Pipeline(settings)
    pipeline.run_score(model_path=model_path)


@app.command()
def evaluate(
    scores_path: Path = typer.Option(
        Path("outputs/scores.parquet"),
        "--scores", "-s",
        help="Path to scored observations",
    ),
    output: Path = typer.Option(
        Path("outputs/metrics.json"),
        "--output", "-o",
        help="Output path for metrics JSON",
    ),
) -> None:
    """
    Evaluate scored results against ground truth.
    """
    from bsad import io, steps

    settings = Settings(output_dir=output.parent)
    pipeline = Pipeline(settings)

    # Load and evaluate
    pipeline.state.scored_df = io.load_parquet(scores_path)
    pipeline.state.metrics = steps.evaluate(pipeline.state.scored_df)
    io.save_json(pipeline.state.metrics, output)

    pipeline._print_metrics()


@app.command("generate-data")
def generate_data(
    output: Path = typer.Option(
        Path("data/events.parquet"),
        "--output", "-o",
        help="Output path for events parquet",
    ),
    n_entities: int = typer.Option(
        200,
        "--n-entities", "-n",
        help="Number of entities",
    ),
    n_days: int = typer.Option(
        30,
        "--n-days", "-d",
        help="Number of days",
    ),
    attack_rate: float = typer.Option(
        0.02,
        "--attack-rate", "-a",
        help="Fraction of entity-windows with attacks",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
) -> None:
    """
    Generate synthetic security event data.

    Useful for testing or creating training data.
    """
    from rich.console import Console

    from bsad import io, steps

    console = Console()
    console.print("Generating synthetic data...")

    settings = Settings(
        data_dir=output.parent,
        n_entities=n_entities,
        n_days=n_days,
        attack_rate=attack_rate,
        random_seed=seed,
    )
    settings.ensure_dirs()

    events_df, attacks_df = steps.generate_data(settings)
    io.save_events(events_df, attacks_df, output)

    console.print(f"  Generated {len(events_df):,} events")
    console.print(f"  Attacks: {events_df['is_attack'].sum():,}")
    console.print(f"  Saved to {output}")


if __name__ == "__main__":
    app()
