# Quant Trading Strategy Backtester

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Test](https://github.com/IsaacCheng9/quant-trading-strategy-backtester/actions/workflows/test.yml/badge.svg)](https://github.com/IsaacCheng9/quant-trading-strategy-backtester/actions/workflows/test.yml)

A quantitative trading strategy backtester with an interactive dashboard.
Enables users to implement, test, and visualise trading strategies using
historical market data, featuring customisable parameters and key performance
metrics. Developed with Python.

## Usage

### Installing Dependencies

Run the following command from the [project root](./) directory:

```bash
poetry install
```

### Running the Application Locally

Run the following command from the [project root](./) directory:

```bash
poe app
```

Alternatively, run it directly with Poetry (skipping the Poe alias):

```bash
poetry run streamlit run src/quant_trading_strategy_backtester/app.py
```
