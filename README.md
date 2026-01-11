# Crypto Backtester

[![PyPI version](https://badge.fury.io/py/crypto-backtester-binance.svg)](https://badge.fury.io/py/crypto-backtester-binance)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/crypto-backtester-binance)](https://pypi.org/project/crypto-backtester-binance/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/LevRoz630/crypto-backtester/workflows/Tests/badge.svg)](https://github.com/LevRoz630/crypto-backtester/actions)
[![Code Quality](https://github.com/LevRoz630/crypto-backtester/workflows/Code%20Quality/badge.svg)](https://github.com/LevRoz630/crypto-backtester/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A backtesting framework for cryptocurrency trading strategies on Binance. Supports spot and perpetual futures markets with risk management, position sizing, and performance analytics.

## Features

- **Historical Data Collection**: Automated collection and caching of OHLCV, trades, funding rates, and open interest data
- **Strategy Backtesting**: Run strategies over historical data with configurable time steps
- **Risk Management**: Built-in position manager with volatility-based risk screening and inverse-vol weighting
- **OMS Simulation**: Order management system that tracks positions, balances, and trade history
- **Performance Metrics**: Returns, Sharpe ratio, drawdown, and permutation testing
- **Multiple Market Types**: Support for both spot and perpetual futures markets

## Prerequisites

- Python 3.11 or higher
- pip or uv package manager

## Installation

### From PyPI (recommended)

```bash
pip install crypto-backtester-binance
```

### From Source

```bash
git clone https://github.com/LevRoz630/crypto-backtester.git
cd crypto-backtester
pip install -e .
```

With development dependencies:
```bash
pip install -e ".[dev,test,docs]"
```

### Data Directory

The framework will automatically download and cache historical data. You can specify a custom path when initializing the `Backtester`, or it defaults to `./historical_data`.

## Project Structure

```
crypto-backtester-binance/
├── src/crypto_backtester_binance/  # Core library
│   ├── backtester.py               # Main backtest orchestrator
│   ├── oms_simulation.py           # Order management system
│   ├── hist_data.py                # Historical data collector
│   ├── position_manager.py         # Risk management & position sizing
│   └── utils.py                    # Utility functions
├── examples/                       # Example scripts
│   ├── 01_simple_backtest.py       # Buy-and-hold strategy
│   ├── 02_momentum_strategy.py     # SMA crossover strategy
│   └── 03_long_short_strategy.py   # Long BTC / Short Alts
├── tests/                          # Test suite
├── docs/                           # Documentation
└── historical_data/                # Cached data (created on first run)
```

## Quick Start

### Running Your First Backtest

```python
from datetime import datetime, timedelta, UTC

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.hist_data import HistoricalDataCollector
from crypto_backtester_binance.oms_simulation import OMSClient
from crypto_backtester_binance.position_manager import PositionManager


# Define a simple strategy
class HoldStrategy:
    def __init__(self, symbols: list[str], lookback_days: int = 0):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.has_bought = False

    def run_strategy(self, oms_client: OMSClient, data_manager: HistoricalDataCollector):
        orders = []
        if not self.has_bought:
            for symbol in self.symbols:
                orders.append({"symbol": symbol, "instrument_type": "future", "side": "LONG"})
            self.has_bought = True
        return orders


# Initialize backtester
backtester = Backtester(historical_data_dir="./historical_data")

# Create strategy and position manager
strategy = HoldStrategy(symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"])
position_manager = PositionManager()

# Set backtest period
start_date = datetime.now(UTC) - timedelta(days=50)
end_date = datetime.now(UTC) - timedelta(days=1)

# Run backtest
results = backtester.run_backtest(
    strategy=strategy,
    position_manager=position_manager,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(days=1),
    market_type="futures",
)

# View results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")

# Generate plots
backtester.plot_portfolio_value()
backtester.plot_positions(results)
```

The first run will download historical data automatically. Subsequent runs use cached data.

### Running Example Scripts

```bash
# Simple buy-and-hold
python examples/01_simple_backtest.py

# Momentum strategy (SMA crossover)
python examples/02_momentum_strategy.py

# Long BTC / Short Alts
python examples/03_long_short_strategy.py
```

## Creating Custom Strategies

A strategy must implement the `run_strategy` method that returns a list of order dictionaries:

```python
from datetime import timedelta

from crypto_backtester_binance.hist_data import HistoricalDataCollector
from crypto_backtester_binance.oms_simulation import OMSClient


class MyStrategy:
    def __init__(self, symbols: list[str], lookback_days: int):
        self.symbols = symbols
        self.lookback_days = lookback_days

    def run_strategy(
        self,
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector,
    ) -> list[dict]:
        """
        Generate trading orders based on strategy logic.

        Returns:
            List of order dictionaries with keys:
            - symbol: str (e.g., "BTC-USDT")
            - instrument_type: str ("spot" or "future")
            - side: str ("LONG", "SHORT", or "CLOSE")
            - value: float (optional, USDT notional)
        """
        orders = []

        for symbol in self.symbols:
            # Load historical data
            data = data_manager.load_data_period(
                symbol=symbol,
                timeframe="1h",
                data_type="mark_ohlcv_futures",
                start_date=oms_client.current_time - timedelta(days=self.lookback_days),
                end_date=oms_client.current_time,
            )

            # Your strategy logic here...

            orders.append({
                "symbol": symbol,
                "instrument_type": "future",
                "side": "LONG",  # or "SHORT" or "CLOSE"
            })

        return orders
```

## Configuration

### Market Types

- **`"futures"`**: Uses perpetual futures data with margin-based positions

### Time Steps

Supported time deltas map to data timeframes:
- `timedelta(minutes=1)` → `"1m"`
- `timedelta(minutes=5)` → `"5m"`
- `timedelta(minutes=15)` → `"15m"` (default)
- `timedelta(minutes=30)` → `"30m"`
- `timedelta(hours=1)` → `"1h"`

### Position Manager

The default `PositionManager`:
- Risk screens orders using 4-hour volatility (sets `value=0` if scaled vol > 0.1)
- Sizes orders using inverse-volatility weighting
- Allocates 10% of USDT balance per backtest step
- Enforces cash constraints

You can create custom position managers by extending the `PositionManager` class.

## Documentation

### Building Documentation

If you installed with `[docs]`:

```bash
# Build static HTML
mkdocs build

# Serve locally for preview
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

### Documentation Structure

- **Overview**: Architecture and data flow (`docs/overview.md`)
- **API Reference**: Auto-generated from docstrings (`docs/api/`)
- **Examples**: Strategy implementations (`examples/`)

## Permutation Testing

Test strategy significance using randomized returns:

```python
results = backtester.run_permutation_backtest(
    strategy=strategy,
    position_manager=position_manager,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(days=1),
    market_type="futures",
    permutations=100,  # Number of random permutations
)

print(f"P-value: {results['p_value']:.4f}")
print(f"Observed Sharpe: {results['sharpe_ratio']:.2f}")
```

## Troubleshooting

### Data Download Issues

- Ensure you have internet connectivity for first-time data collection
- Check that the `historical_data` directory is writable
- Data is cached in Parquet format for fast subsequent loads

### Import Errors

If you see import errors, ensure the package is installed:

```bash
pip install crypto-backtester-binance
```

Or for development:
```bash
pip install -e .
```

### Memory Issues

For large backtests:
- Reduce `lookback_days` in your strategy
- Use longer `time_step` intervals
- Process data in chunks

## License

See `LICENSE` file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue on the repository.

