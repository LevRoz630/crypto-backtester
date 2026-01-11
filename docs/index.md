# Crypto Backtester

A backtesting framework for cryptocurrency trading strategies.

## Quick Start

```python
from datetime import datetime, timedelta, timezone
from typing import Any

from crypto_backtester_binance import Backtester, HistoricalDataCollector


# Define your strategy
class HoldStrategy:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.has_bought = False

    def run_strategy(self, oms_client, data_manager):
        if self.has_bought:
            return []
        self.has_bought = True
        return [{"symbol": s, "instrument_type": "future", "side": "LONG"} for s in self.symbols]


# Define your position manager (user-defined risk management)
class SimplePositionManager:
    def filter_orders(
        self, orders: list[dict[str, Any]], oms_client: Any, data_manager: HistoricalDataCollector
    ) -> list[dict[str, Any]] | None:
        if not orders:
            return None
        budget = oms_client.balance["USDT"] * 0.1 / len(orders)
        return [{**order, "value": budget} for order in orders]


bt = Backtester(historical_data_dir="./historical_data")
strategy = HoldStrategy(symbols=["BTC-USDT", "ETH-USDT"])
pm = SimplePositionManager()

start_date = datetime.now(timezone.utc) - timedelta(days=30)
end_date = datetime.now(timezone.utc)

results = bt.run_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(hours=1),
    market_type="futures",
)
```

## Installation

```bash
pip install crypto-backtester-binance
```

## Building Documentation

```bash
mkdocs build
mkdocs serve  # Preview at http://127.0.0.1:8000
```

See [Overview](overview.md) for architecture details.
