from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add src directory to path for package imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add project root to path for backtest module
project_root = src_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.position_manager import PositionManager
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
backtester.plot_positions()

# Save results
backtester.save_results(results, "my_backtest")