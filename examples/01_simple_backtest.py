#!/usr/bin/env python3
"""
Simple Buy-and-Hold Strategy Example

This example demonstrates the basic backtesting workflow:
1. Create a simple strategy that buys and holds
2. Run a backtest over historical data
3. View results and generate plots

Usage:
    python examples/01_simple_backtest.py
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.hist_data import HistoricalDataCollector
from crypto_backtester_binance.oms_simulation import OMSClient
from crypto_backtester_binance.position_manager import PositionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HoldStrategy:
    """
    Simple buy-and-hold strategy.

    Buys equal positions in all configured symbols at the start,
    then holds until the end of the backtest period.
    """

    def __init__(self, symbols: list[str], lookback_days: int = 0):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.has_bought = False

    def run_strategy(
        self, oms_client: OMSClient, data_manager: HistoricalDataCollector
    ) -> list[dict]:
        """
        Generate trading signals.

        Called at each time step during the backtest.
        Returns a list of order dictionaries.
        """
        orders = []

        # Only buy once at the beginning
        if not self.has_bought:
            usdt_balance = oms_client.balance.get("USDT", 0)
            if usdt_balance > 0:
                for symbol in self.symbols:
                    orders.append(
                        {
                            "symbol": symbol,
                            "instrument_type": "future",
                            "side": "LONG",
                        }
                    )
            self.has_bought = True

        return orders


def main():
    """Run the buy-and-hold backtest."""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    print("Strategy: Buy and Hold")
    print()

    # Configure historical data directory (at repo root)
    hist_dir = Path(__file__).parent.parent / "historical_data"

    # Initialize backtester
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Configure strategy
    strategy = HoldStrategy(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT"],
        lookback_days=0,
    )

    # Configure position manager (handles sizing and risk)
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

    # Display results
    backtester.print_results(results)

    # Generate plots
    backtester.plot_portfolio_value()
    backtester.plot_positions(results)

    # Save results to file
    backtester.save_results(results, "hold_strategy")

    print()
    print("Results saved to 'hold_strategy' files")


if __name__ == "__main__":
    main()
