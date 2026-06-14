# QFR reproduction scaffold

This workspace now contains a minimal but runnable scaffold for reproducing the main pipeline of QuantFactor REINFORCE.

The scaffold is intentionally split into the same layers implied by the paper:

- data: real A-share panel loading / toy generation and split management
- formula engine: token vocabulary, RPN legality, expression parsing, factor evaluation
- environment: masked token-by-token formula generation
- policy: sequence model that predicts the next token
- training: REINFORCE with greedy baseline and IR-based reward shaping
- factor pool: weighted combination model with ridge-style fitting and pruning

What is implemented now:

- a real A-share daily loader for 2022-2025 style long-form market data under dataset/mainboard_4y_daily.pkl
- a toy market generator kept as a fallback baseline
- an RPN expression engine covering the paper's operator families
- a legality-aware environment that masks invalid actions
- a GRU policy trained with QFR-style sampled reward minus greedy baseline
- IC and IR reward computation, including time-varying IR thresholding
- a factor pool that fits combination weights and keeps the strongest factors
- a runnable training script that saves outputs under artifacts/

What is still not paper-complete:

- no benchmark-index-specific loader yet
- no PPO, TRPO, A3C baseline runners yet
- no exact AlphaGen compatibility layer yet
- no backtesting engine yet
- no multi-seed experiment harness yet

## Project layout

- configs/toy_reproduction.yaml: default toy experiment config
- configs/ashare_reproduction.yaml: default A-share experiment config
- scripts/run_toy_experiment.py: single-entry training script
- src/qfr/: framework code

## Run

Use the quant-rl environment:

```powershell
D:/anaconda3/envs/quant-rl/python.exe scripts/run_toy_experiment.py --config configs/ashare_reproduction.yaml
```

Outputs are written to artifacts/ashare_qfr/ by default.

GPU support:

You can run training on GPU when available. The `--device` flag accepts `cpu`, `cuda`, or `auto` (auto will pick CUDA if available):

```bash
python scripts/run_experiment.py --config configs/ashare_reproduction.yaml --device auto
```

## Suggested next build steps

1. Add benchmark-aware targets such as excess return over SSE or CSI series.
2. Add a persistent experiment logger for multiple seeds and datasets.
3. Add a backtest module so the mined factors can be evaluated exactly like the paper's investment simulation.
4. Add PPO and AlphaGen-style baselines for direct comparison.
