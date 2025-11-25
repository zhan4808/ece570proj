#!/bin/bash
# One-command execution script for AutoRAG-Allocator

set -e  # Exit on error

# Default values
DATASET="nq"
BUDGET_COST=5.0
BUDGET_LAT=2000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --budget-cost)
            BUDGET_COST="$2"
            shift 2
            ;;
        --budget-lat)
            BUDGET_LAT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--dataset nq|hotpot] [--budget-cost 5.0] [--budget-lat 2000]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "AutoRAG-Allocator"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Budget: $BUDGET_COST ¢/query, $BUDGET_LAT ms"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Please create it from .env.example"
    echo "Continuing anyway..."
fi

# Run experiments
echo "Running experiments..."
python experiments/run_experiments.py

# Generate figures
echo ""
echo "Generating figures..."
python experiments/generate_figures.py

echo ""
echo "=========================================="
echo "Complete! Results saved to results/"
echo "=========================================="

