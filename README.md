# Crypto Backtester

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/LevRoz630/crypto-backtester-binance/workflows/Tests/badge.svg)](https://github.com/LevRoz630/crypto-backtester-binance/actions)
[![Code Quality](https://github.com/LevRoz630/crypto-backtester-binance/workflows/Code%20Quality/badge.svg)](https://github.com/LevRoz630/crypto-backtester-binance/actions)
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

### 1. Clone the Repository

```bash
git clone <repository-url>
cd crypto-backtester-binance
```

### 2. Install Dependencies

Using pip:
```bash
pip install -e .
```

Or with documentation dependencies:
```bash
pip install -e ".[docs]"
```

Using uv (recommended):
```bash
uv pip install -e .
```

### 3. Set Up Data Directory

Create a directory for storing historical data:

```bash
mkdir historical_data
```

The framework will automatically download and cache data in this directory. You can also specify a custom path when initializing the `Backtester`.

## Project Structure

```
crypto-backtester-binance/
├── src/                    # Core engine modules
│   ├── backtester.py       # Main backtest orchestrator
│   ├── oms_simulation.py   # Order management system
│   ├── hist_data.py        # Historical data collector
│   ├── position_manager.py # Risk management & position sizing
│   └── utils.py            # Utility functions
├── backtest/
│   ├── strategies/         # Strategy implementations
│   ├── position_managers/  # Custom position managers
│   └── example/            # Example scripts
├── docs/                   # Documentation
│   └── api/                # API reference
└── historical_data/        # Cached historical data (created on first run)
```

## Quick Start

### Running Your First Backtest

1. **Create a simple script** (`my_backtest.py`):

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from backtester import Backtester
from position_manager import PositionManager
from backtest.example.v1_hold import HoldStrategy

# Initialize backtester
backtester = Backtester(historical_data_dir="./historical_data")

# Create strategy
strategy = HoldStrategy(
    symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
    lookback_days=0
)

# Create position manager
position_manager = PositionManager()

# Set backtest period
start_date = datetime.now(timezone.utc) - timedelta(days=50)
end_date = datetime.now(timezone.utc) - timedelta(days=1)

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

# Plot positions
backtester.plot_positions(results)

# Save results
backtester.save_results(results, "my_backtest")
```

2. **Run the script**:

```bash
python my_backtest.py
```

The first run will download historical data automatically. Subsequent runs use cached data.

### Running Example Scripts

The repository includes example scripts in `backtest/example/`:

```bash
# Run hold strategy example
python backtest/example/example.py

# Run pairs trading strategy
python backtest/v1_pairs_bt.py

# Run long-short strategy
python backtest/v1_ls_bt.py
```

## Creating Custom Strategies

A strategy must implement the `run_strategy` method that returns a list of order dictionaries:

```python
from typing import List, Dict, Any
from oms_simulation import OMSClient
from hist_data import HistoricalDataCollector

class MyStrategy:
    def __init__(self, symbols: List[str], lookback_days: int):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.oms_client = None
        self.data_manager = None
    
    def run_strategy(
        self, 
        oms_client: OMSClient, 
        data_manager: HistoricalDataCollector
    ) -> List[Dict[str, Any]]:
        """
        Generate trading orders based on strategy logic.
        
        Returns:
            List of order dictionaries with keys:
            - symbol: str (e.g., "BTC-USDT")
            - instrument_type: str ("spot" or "future")
            - side: str ("LONG", "SHORT", or "CLOSE")
            - value: float (optional, USDT notional; PositionManager will size if omitted)
        """
        self.oms_client = oms_client
        self.data_manager = data_manager
        
        orders = []
        
        # Example: Load price data
        for symbol in self.symbols:
            data = self.data_manager.load_data_period(
                symbol=symbol,
                timeframe="1h",
                data_type="mark_ohlcv_futures",
                start=self.oms_client.current_time - timedelta(days=self.lookback_days),
                end=self.oms_client.current_time
            )
            
            # Your strategy logic here
            # ...
            
            # Add order
            orders.append({
                "symbol": symbol,
                "instrument_type": "future",
                "side": "LONG"  # or "SHORT" or "CLOSE"
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

- **Overview**: Architecture and data flow (`backtest/docs/docs.md`)
- **API Reference**: Auto-generated from docstrings (`docs/api/`)
- **Method Docs**: Detailed method documentation (`backtest/docs/`)

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

If you see import errors, ensure `src/` is in your Python path:

```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
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

