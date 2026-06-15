# QuantFactor REINFORCE

This project implements a framework for mining steady formulaic alpha factors using reinforcement learning. It includes data loading for A-share markets, an RPN formula engine, a masked token-by-token generation environment, a GRU-based policy network, and a factor pool for weighted combination.

## Quickstart

### Run with A-share data

```bash
python scripts/run_toy_experiment.py --config configs/ashare_reproduction.yaml
```

### GPU Support

The `--device` flag accepts `cpu`, `cuda`, or `auto`:

```bash
python scripts/run_toy_experiment.py --config configs/ashare_reproduction.yaml --device auto
```

### Project Structure

- `configs/`: Experiment configurations
- `scripts/`: Training scripts
- `src/qfr/`: Framework code
- `artifacts/`: Output directory

---

This work is inspired by [Zhao et al. (2025)](https://doi.org/10.1109/TSP.2025.3576781).