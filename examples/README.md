# Examples

This directory contains example scripts demonstrating how to use the crypto-backtester-binance library.

## Prerequisites

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Examples

### 01_simple_backtest.py

A minimal buy-and-hold strategy that demonstrates the basic backtesting workflow:
- Setting up the `Backtester`
- Creating a simple strategy
- Running a backtest
- Viewing results and plots

```bash
python examples/01_simple_backtest.py
```

### 02_momentum_strategy.py

A momentum-based strategy that:
- Goes long when price is above the moving average
- Goes short when price is below the moving average
- Demonstrates using historical data within strategies

```bash
python examples/02_momentum_strategy.py
```

### 03_long_short_strategy.py

A market-neutral strategy that:
- Goes long BTC as a safe haven
- Shorts altcoins based on variance-adjusted weights
- Demonstrates multi-asset position management

```bash
python examples/03_long_short_strategy.py
```

## Strategy Interface

All strategies must implement the following interface:

```python
class MyStrategy:
    def __init__(self, symbols: list[str], lookback_days: int):
        self.symbols = symbols
        self.lookback_days = lookback_days

    def run_strategy(
        self,
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector
    ) -> list[dict]:
        """
        Called at each time step during the backtest.

        Returns a list of order dictionaries:
        {
            "symbol": str,           # e.g., "BTC-USDT"
            "instrument_type": str,  # "spot" or "future"
            "side": str,             # "LONG", "SHORT", or "CLOSE"
            "value": float,          # Optional: position value in USDT
            "alloc_frac": float,     # Optional: fraction of portfolio (0-1)
        }
        """
        return []
```

## Position Manager Interface

Position managers handle risk screening and position sizing:

```python
class MyPositionManager:
    def filter_orders(
        self,
        orders: list[dict],
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector,
    ) -> list[dict]:
        """
        Filter and size orders based on risk criteria.

        Returns modified orders with position sizes.
        """
        return orders
```
