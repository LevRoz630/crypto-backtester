#!/usr/bin/env python3
"""
Long BTC / Short Altcoin Strategy Example

This example demonstrates a market-neutral strategy:
- Long Bitcoin as a hedge
- Short altcoins with negative 24h returns
- Position sizing based on variance-adjusted weights
- Daily rebalancing

Usage:
    python examples/03_long_short_strategy.py
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.hist_data import HistoricalDataCollector
from crypto_backtester_binance.oms_simulation import OMSClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LongShortStrategy:
    """
    Long BTC / Short Altcoin portfolio strategy.

    Strategy logic:
    - Long Bitcoin futures as a hedge (30% of portfolio)
    - Short altcoins with negative 24h returns (70% of portfolio)
    - Position sizing based on variance-adjusted weights
    - Rebalance daily or when BTC/Alt ratio drifts >20%
    """

    def __init__(
        self,
        symbols: list[str],
        lookback_days: int = 30,
        btc_ratio: float = 0.3,
        drift_threshold: float = 0.20,
    ):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.btc_ratio = btc_ratio
        self.drift_threshold = drift_threshold

        # Identify BTC and altcoins
        self.btc_symbol = None
        self.altcoin_symbols = []

        for sym in symbols:
            perp_symbol = sym + "-PERP" if not sym.endswith("-PERP") else sym
            base = perp_symbol.replace("-USDT", "").replace("-PERP", "")
            if base == "BTC":
                self.btc_symbol = perp_symbol
            else:
                self.altcoin_symbols.append(perp_symbol)

        if not self.btc_symbol:
            raise ValueError("BTC-USDT must be included in symbols")
        if not self.altcoin_symbols:
            raise ValueError("At least one altcoin must be included")

        self.last_rebalance_day = None
        self.oms_client: OMSClient | None = None

    def _get_hourly_data(self, base_symbol: str, hours: int = 48) -> pd.DataFrame:
        """Get hourly index price data for a symbol."""
        dm = self.oms_client.data_manager
        window_start = self.oms_client.current_time - pd.Timedelta(hours=hours)
        end_time = self.oms_client.current_time

        df = dm.perpetual_index_ohlcv_data.get(base_symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df[df["timestamp"].between(window_start, end_time, inclusive="left")]
        return df.sort_values("timestamp")

    def _calculate_24h_return(self, base_symbol: str) -> float:
        """Calculate 24h log returns for a symbol."""
        df = self._get_hourly_data(base_symbol, hours=48)
        if df.empty or len(df) < 24:
            return 0.0

        df = df.set_index("timestamp")
        hourly = df["close"].resample("1h").last().dropna()

        if len(hourly) < 24:
            return 0.0

        price_24h_ago = hourly.iloc[-25] if len(hourly) >= 25 else hourly.iloc[0]
        current_price = hourly.iloc[-1]

        return float(np.log(current_price / price_24h_ago))

    def _calculate_variance(self, base_symbol: str) -> float:
        """Calculate variance of returns over lookback period."""
        df = self._get_hourly_data(base_symbol, hours=self.lookback_days * 24)
        if df.empty or len(df) < 48:
            return 1.0

        df = df.set_index("timestamp")
        hourly = df["close"].resample("1H").last().dropna()

        if len(hourly) < 48:
            return 1.0

        returns = np.log(hourly / hourly.shift(24)).dropna()
        if len(returns) < 2:
            return 1.0

        return max(float(returns.var()), 1e-8)

    def _calculate_altcoin_weights(self) -> dict[str, float]:
        """Calculate altcoin portfolio weights based on negative returns and variance."""
        weights = {}

        for symbol in self.altcoin_symbols:
            base = symbol.replace("-PERP", "")
            log_return = self._calculate_24h_return(base)

            # Only short coins with negative returns
            if log_return >= 0:
                continue

            variance = self._calculate_variance(base)
            # Weight: |return| / variance (inverse-variance weighting)
            raw_weight = abs(log_return) / variance
            weights[symbol] = raw_weight

        if not weights:
            return {}

        # Normalize weights
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        # Filter to only keep weights > 10%
        weights = {k: v for k, v in weights.items() if v > 0.10}

        # Renormalize
        if weights:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _should_rebalance(self) -> bool:
        """Check if BTC/Alt ratio has drifted from target."""
        positions = self.oms_client.get_position()

        btc_value = 0.0
        alt_value = 0.0

        for pos in positions:
            symbol = pos["symbol"]
            value = abs(float(pos.get("value", 0.0)))
            if symbol == self.btc_symbol:
                btc_value = value
            elif symbol in self.altcoin_symbols:
                alt_value += value

        total = btc_value + alt_value
        if total < 1.0:
            return True

        current_ratio = btc_value / total
        drift = abs(current_ratio - self.btc_ratio)
        return drift > self.drift_threshold

    def run_strategy(
        self, oms_client: OMSClient, data_manager: HistoricalDataCollector
    ) -> list[dict]:
        """Execute strategy logic."""
        self.oms_client = oms_client
        now = oms_client.current_time

        # Check rebalance conditions
        current_day = (now.year, now.month, now.day)
        should_rebalance_daily = self.last_rebalance_day != current_day
        should_rebalance_ratio = self._should_rebalance()

        if not (should_rebalance_daily or should_rebalance_ratio):
            return []

        logger.info(f"Rebalancing at {now}")

        orders = []
        altcoin_weights = self._calculate_altcoin_weights()

        if not altcoin_weights:
            # No altcoins to short, close all positions
            orders.append(
                {
                    "symbol": self.btc_symbol,
                    "instrument_type": "future",
                    "side": "CLOSE",
                }
            )
            for alt in self.altcoin_symbols:
                orders.append(
                    {
                        "symbol": alt,
                        "instrument_type": "future",
                        "side": "CLOSE",
                    }
                )
            return orders

        # Long BTC
        orders.append(
            {
                "symbol": self.btc_symbol,
                "instrument_type": "future",
                "alloc_frac": self.btc_ratio,
                "side": "LONG",
            }
        )

        # Short altcoins
        for alt_sym, weight in altcoin_weights.items():
            orders.append(
                {
                    "symbol": alt_sym,
                    "instrument_type": "future",
                    "alloc_frac": (1.0 - self.btc_ratio) * weight,
                    "side": "SHORT",
                }
            )

        # Close positions for altcoins not in current portfolio
        for alt_sym in self.altcoin_symbols:
            if alt_sym not in altcoin_weights:
                orders.append(
                    {
                        "symbol": alt_sym,
                        "instrument_type": "future",
                        "side": "CLOSE",
                    }
                )

        self.last_rebalance_day = current_day
        return orders


class LongShortPositionManager:
    """Position manager for the long-short strategy with loss-based exits."""

    def __init__(self, max_position_value: float = 2000.0, loss_threshold: float = 0.05):
        self.max_position_value = max_position_value
        self.loss_threshold = loss_threshold
        self.oms_client: OMSClient | None = None

    def _check_stop_loss(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Close positions that have lost more than threshold."""
        positions = self.oms_client.get_position() or []

        for pos in positions:
            try:
                qty = float(pos.get("quantity", 0.0))
                entry_price = float(pos.get("entry_price", 0.0))
                current_value = float(pos.get("value", 0.0))
            except (TypeError, ValueError):
                continue

            threshold = 0.95 * entry_price * qty
            if current_value < threshold:
                logger.info(f"Stop loss triggered for {pos['symbol']}")
                orders.append(
                    {
                        "symbol": pos["symbol"],
                        "instrument_type": pos["instrument_type"],
                        "side": "CLOSE",
                    }
                )

        return orders

    def _apply_position_sizing(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert alloc_frac to actual position values."""
        sized = []
        for order in orders:
            if order.get("side") == "CLOSE":
                sized.append(order)
                continue

            alloc = order.get("alloc_frac", 0.0)
            order["value"] = self.max_position_value * alloc
            sized.append(order)

        return sized

    def filter_orders(
        self,
        orders: list[dict[str, Any]],
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector,
    ) -> list[dict[str, Any]]:
        """Filter and size orders."""
        self.oms_client = oms_client
        orders = self._check_stop_loss(orders or [])
        orders = self._apply_position_sizing(orders)
        return orders


def main():
    """Run the long-short strategy backtest."""
    print("Crypto Trading Strategy Backtester")
    print("=" * 50)
    print("Strategy: Long BTC / Short Altcoins")
    print()

    # Configure historical data directory
    hist_dir = Path(__file__).parent.parent / "historical_data"

    # Initialize backtester
    backtester = Backtester(historical_data_dir=str(hist_dir))

    # Configure strategy
    strategy = LongShortStrategy(
        symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT"],
        lookback_days=5,
        btc_ratio=0.3,
    )

    # Configure position manager
    position_manager = LongShortPositionManager(max_position_value=2000.0)

    # Set backtest period
    start_date = datetime.now(UTC) - timedelta(days=30)
    end_date = datetime.now(UTC) - timedelta(days=1)

    # Run backtest
    results = backtester.run_backtest(
        strategy=strategy,
        position_manager=position_manager,
        start_date=start_date,
        end_date=end_date,
        time_step=timedelta(hours=1),
        market_type="futures",
    )

    # Display results
    backtester.print_results(results)

    # Generate plots
    backtester.plot_portfolio_value()
    backtester.plot_drawdown()
    backtester.plot_returns()

    # Save results
    backtester.save_results(results, "long_short_strategy")

    print()
    print("Results saved to 'long_short_strategy' files")


if __name__ == "__main__":
    main()
