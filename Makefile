.PHONY: install install-dev demo clean clean-figures test lint format help env env-remove streamlit \
       viz-explore viz-features viz-model viz-results viz-eval viz-all viz-report \
       benchmark benchmark-quick robustness eda

# Default target
help:
	@echo "Bayesian Security Anomaly Detection"
	@echo ""
	@echo "Usage:"
	@echo "  make env          Create conda environment"
	@echo "  make env-remove   Remove conda environment"
	@echo "  make install      Install package (pip)"
	@echo "  make install-dev  Install with dev dependencies"
	@echo "  make demo         Run complete pipeline demo"
	@echo "  make demo-fast    Run quick demo (fewer samples)"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linters"
	@echo "  make format       Format code"
	@echo "  make streamlit    Run Streamlit dashboard"
	@echo "  make clean        Remove generated files"
	@echo ""
	@echo "Benchmark (reproducible comparison):"
	@echo "  make benchmark       Full benchmark vs baselines (NB, GLMM, IF, LOF)"
	@echo "  make benchmark-quick Quick benchmark (fewer samples, single rate)"
	@echo "  make robustness      Robustness analysis (drift, cold-start, rates)"
	@echo ""
	@echo "Visualization:"
	@echo "  make viz-explore  Data exploration visualizations"
	@echo "  make viz-features Feature engineering visualizations"
	@echo "  make viz-model    Model diagnostics visualizations"
	@echo "  make viz-results  Anomaly results visualizations"
	@echo "  make viz-eval     Evaluation metrics visualizations"
	@echo "  make viz-all      Run all visualizations"
	@echo "  make viz-report   Generate full PDF report"

# Conda environment
env:
	conda env create -f environment.yml
	@echo ""
	@echo "Environment created. Activate with:"
	@echo "  conda activate bsad"

env-remove:
	conda env remove -n bsad

# Installation (after activating conda env)
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Demo targets
demo:
	bsad demo --output-dir outputs --n-entities 200 --n-days 30 --samples 2000

demo-fast:
	bsad demo --output-dir outputs --n-entities 100 --n-days 14 --samples 500

# Individual pipeline steps
generate-data:
	bsad generate-data --n-entities 200 --n-days 30 --output data/events.parquet

train:
	bsad train --input data/events.parquet --output outputs/model.nc --samples 2000

score:
	bsad score --model outputs/model.nc --input outputs/modeling_table.parquet --output outputs/scores.parquet

evaluate:
	bsad evaluate --scores outputs/scores.parquet --output outputs/metrics.json

# Development
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/bsad/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Streamlit dashboard
streamlit:
	streamlit run app/streamlit_app.py

# Visualization targets
viz-explore:
	python dataviz/01_data_exploration.py --input data/events.parquet --output outputs/figures/exploration

viz-features:
	python dataviz/02_feature_analysis.py --input data/events.parquet --output outputs/figures/features

viz-model:
	python dataviz/03_model_diagnostics.py --model outputs/model.nc --output outputs/figures/diagnostics

viz-results:
	python dataviz/04_anomaly_results.py --scores outputs/scores.parquet --output outputs/figures/results

viz-eval:
	python dataviz/05_evaluation_plots.py --scores outputs/scores.parquet --output outputs/figures/evaluation

viz-all:
	python dataviz/06_full_report.py --data data/events.parquet --model outputs/model.nc --scores outputs/scores.parquet --output outputs/figures

viz-report:
	python dataviz/06_full_report.py --data data/events.parquet --model outputs/model.nc --scores outputs/scores.parquet --output outputs/figures --pdf

# Benchmark targets
benchmark:
	python scripts/benchmark.py --output outputs/benchmark --attack-rates 0.01 0.02 0.05 --seed 42

benchmark-quick:
	python scripts/benchmark.py --output outputs/benchmark --quick --seed 42

robustness:
	python scripts/robustness_analysis.py --output outputs/robustness --seed 42

# EDA Pipeline Explainer (pedagogical visualizations)
eda:
	python scripts/eda_pipeline_explainer.py

# Cleanup
clean:
	rm -rf outputs/
	rm -rf data/*.parquet
	rm -rf __pycache__
	rm -rf src/bsad/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-figures:
	rm -rf outputs/figures/
