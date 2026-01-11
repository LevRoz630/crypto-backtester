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

### position_manager.py

A reference implementation of a position manager with:
- Volatility-based risk screening (4h scaled vol threshold)
- Inverse-volatility position sizing
- Budget enforcement (10% of USDT balance per step)
- CLOSE order passthrough

This file is imported by the example scripts. Customize it for your own risk management needs.

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

Position managers are **user-defined** - you create them to match your risk management needs.
The only requirement is implementing the `filter_orders` method:

```python
class MyPositionManager:
    def filter_orders(
        self,
        orders: list[dict],
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector,
    ) -> list[dict] | None:
        """
        Process orders from strategy before execution.

        Args:
            orders: Raw orders from strategy
            oms_client: Access to balance, positions, current time
            data_manager: Access to historical data

        Returns:
            List of orders with 'value' field set (USDT notional), or None to skip
        """
        if not orders:
            return None
        # Simple equal-weight: 10% of balance split across orders
        budget = oms_client.balance["USDT"] * 0.1 / len(orders)
        return [{**order, "value": budget} for order in orders]
```

See `position_manager.py` for a more sophisticated reference implementation.
