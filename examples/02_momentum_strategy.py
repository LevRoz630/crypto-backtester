#!/usr/bin/env python3
"""
Momentum Strategy Example

This example demonstrates a simple momentum-based strategy:
- Goes long when price is above its moving average
- Goes short when price is below its moving average
- Rebalances daily

Usage:
    python examples/02_momentum_strategy.py
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.hist_data import HistoricalDataCollector
from crypto_backtester_binance.oms_simulation import OMSClient
from crypto_backtester_binance.position_manager import PositionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MomentumStrategy:
    """
    Simple momentum strategy based on moving average crossover.

    Goes long when current price > SMA, short when price < SMA.
    """

    def __init__(
        self,
        symbols: list[str],
        lookback_days: int = 20,
        sma_period: int = 20,
    ):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.sma_period = sma_period
        self.last_signal_day = None

    def _get_sma(self, symbol: str, oms_client: OMSClient) -> float | None:
        """Calculate Simple Moving Average for a symbol."""
        dm = oms_client.data_manager
        base_symbol = symbol.replace("-PERP", "")

        # Get historical data
        window_start = oms_client.current_time - pd.Timedelta(days=self.sma_period + 5)
        end_time = oms_client.current_time

        df = dm.perpetual_index_ohlcv_data.get(base_symbol)
        if df is None or df.empty:
            return None

        df = df[df["timestamp"].between(window_start, end_time, inclusive="left")]
        df = df.sort_values("timestamp")

        if len(df) < self.sma_period:
            return None

        # Resample to daily and calculate SMA
        df = df.set_index("timestamp")
        daily = df["close"].resample("1D").last().dropna()

        if len(daily) < self.sma_period:
            return None

        sma = daily.iloc[-self.sma_period:].mean()
        return float(sma)

    def _get_current_price(self, symbol: str, oms_client: OMSClient) -> float | None:
        """Get current price for a symbol."""
        base_symbol = symbol.replace("-PERP", "")
        return oms_client.get_current_price(base_symbol, "future")

    def run_strategy(
        self, oms_client: OMSClient, data_manager: HistoricalDataCollector
    ) -> list[dict]:
        """Generate momentum signals."""
        orders = []
        now = oms_client.current_time

        # Only rebalance once per day
        current_day = (now.year, now.month, now.day)
        if self.last_signal_day == current_day:
            return []

        self.last_signal_day = current_day

        for symbol in self.symbols:
            perp_symbol = symbol if symbol.endswith("-PERP") else f"{symbol}-PERP"

            sma = self._get_sma(perp_symbol, oms_client)
            current_price = self._get_current_price(perp_symbol, oms_client)

            if sma is None or current_price is None:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # Momentum signal
            if current_price > sma:
                # Price above SMA -> bullish -> go long
                orders.append({
                    "symbol": perp_symbol,
                    "instrument_type": "future",
                    "side": "LONG",
                })
                logger.info(f"{symbol}: LONG (price {current_price:.2f} > SMA {sma:.2f})")
            else:
                # Price below SMA -> bearish -> go short
                orders.append({
                    "symbol": perp_symbol,
                    "instrument_type": "future",
                    "side": "SHORT",
                })
                logger.info(f"{symbol}: SHORT (price {current_price:.2f} < SMA {sma:.2f})")

        return orders


def main():
    """Run the momentum strategy backtest."""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    print("Strategy: Momentum (SMA Crossover)")
    print()

    # Configure historical data directory
    hist_dir = Path(__file__).parent.parent / "historical_data"

    # Initialize backtester
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Configure strategy
    strategy = MomentumStrategy(
        symbols=["BTC-USDT", "ETH-USDT"],
        lookback_days=30,
        sma_period=20,
    )

    # Configure position manager
    position_manager = PositionManager()

    # Set backtest period
    start_date = datetime.now(UTC) - timedelta(days=60)
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
    backtester.plot_drawdown()

    # Save results
    backtester.save_results(results, "momentum_strategy")

    print()
    print("Results saved to 'momentum_strategy' files")


if __name__ == "__main__":
    main()
