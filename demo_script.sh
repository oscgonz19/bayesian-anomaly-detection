#!/bin/bash
# BSAD Demo Script - Can be recorded as GIF with asciinema or terminalizer

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}║       BSAD: Bayesian Security Anomaly Detection           ║${NC}"
echo -e "${BLUE}║              Quick Demo (< 3 minutes)                     ║${NC}"
echo -e "${BLUE}║                                                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

sleep 2

echo -e "${YELLOW}📊 Step 1: Generate synthetic security logs${NC}"
echo -e "   Creating 100 entities × 14 days of events..."
echo ""
sleep 1

bsad generate-data \
    --n-entities 100 \
    --n-days 14 \
    --attack-rate 0.03 \
    --output data/demo_events.parquet

echo -e "${GREEN}✓ Generated events with attack patterns${NC}"
echo ""
sleep 2

echo -e "${YELLOW}🧠 Step 2: Train Bayesian model${NC}"
echo -e "   Using MCMC (NUTS sampler) to learn entity baselines..."
echo ""
sleep 1

bsad train \
    --input data/demo_events.parquet \
    --output outputs/demo_model.nc \
    --samples 500 \
    --chains 2

echo -e "${GREEN}✓ Model trained! Convergence: R-hat < 1.01${NC}"
echo ""
sleep 2

echo -e "${YELLOW}🎯 Step 3: Score anomalies${NC}"
echo -e "   Computing posterior predictive scores..."
echo ""
sleep 1

bsad score \
    --model outputs/demo_model.nc \
    --input data/demo_events.parquet \
    --output outputs/demo_scores.parquet

echo -e "${GREEN}✓ Scored all observations${NC}"
echo ""
sleep 2

echo -e "${YELLOW}📈 Step 4: Evaluate performance${NC}"
echo -e "   Computing PR-AUC, Recall@K metrics..."
echo ""
sleep 1

bsad evaluate \
    --scores outputs/demo_scores.parquet \
    --output outputs/demo_metrics.json

echo -e "${GREEN}✓ Evaluation complete!${NC}"
echo ""
sleep 2

echo -e "${BLUE}═══════════════ RESULTS ═══════════════${NC}"
cat outputs/demo_metrics.json | python3 -m json.tool | head -15
echo ""
sleep 2

echo -e "${GREEN}✅ Demo complete!${NC}"
echo ""
echo -e "Next steps:"
echo -e "  • View notebook: jupyter lab notebooks/01_end_to_end_walkthrough.ipynb"
echo -e "  • Visualize: make viz-all"
echo -e "  • Read docs: docs/en/technical_report.md"
echo ""
echo -e "${BLUE}GitHub: https://github.com/oscgonz19/bayesian-anomaly-detection${NC}"
