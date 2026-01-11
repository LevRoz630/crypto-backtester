## Backtest Engine: Structure and Workflow

### Components
- **Backtester (`src/backtester.py`)**: Orchestrates the time loop, calls `strategy.run_strategy(...)`, passes orders through the user's position manager `filter_orders(...)`, and executes via `OMSClient.set_target_position(...)`. Computes returns, drawdown, Sharpe, and aggregates results.
- **OMSClient (`src/oms_simulation.py`)**: Tracks `balance['USDT']`, `positions`, and `trade_history`. Provides `get_current_price`, `set_target_position`, `close_position`, `get_total_portfolio_value`, and reporting helpers.
- **HistoricalDataCollector (`src/hist_data.py`)**: Loads/caches OHLCV and related series via `load_data_period(symbol, timeframe, data_type, start, end)`.
- **Strategy (user-defined)**: Implements `run_strategy(oms_client, data_manager) -> List[order]`.
- **PositionManager (user-defined)**: Implements `filter_orders(orders, oms_client, data_manager)` to risk-screen, size positions, and enforce budget before OMS execution.

### Data flow (per timestep)
1) Backtester revalues portfolio via `OMSClient.get_total_portfolio_value()`; logs positions.
2) Strategy emits raw orders: `run_strategy(oms_client, data_manager) -> List[Dict]`.
3) PositionManager:
   - Processes orders through `filter_orders(orders, oms_client, data_manager)`
   - Returns orders with `value` field set (USDT notional), or None to skip
4) Backtester executes each order via `OMSClient.set_target_position(symbol, instrument_type, value, side)`.
5) Advance `current_time` by `time_step`; repeat until `end_date`.

### Order schema (after PositionManager)
```json
{
  "symbol": "BTC-USDT",                // or base + -PERP-normalized internally for futures
  "instrument_type": "future",
  "side": "LONG" | "SHORT" | "CLOSE",
  "value": 1234.56                      // USDT notional; PositionManager supplies this
}
```
 ### Data storage map
 - `HistoricalDataCollector.spot_ohlcv_data[symbol]`: spot loop and pricing source when `market_type="spot"`.
 - `HistoricalDataCollector.perpetual_index_ohlcv_data[symbol]`: futures loop timing/prices.
 - `HistoricalDataCollector.perpetual_mark_ohlcv_data[symbol]`: futures execution pricing and PM risk inputs.
 - `HistoricalDataCollector.funding_rates_data[symbol]`: available for strategies/PM if needed.
 - `HistoricalDataCollector.open_interest_data[symbol]`: optional risk/signal input.
 - `OMSClient.positions[symbol]`: live in-memory state per symbol (quantity, side, entry_price, value, pnl, instrument_type).
 - `OMSClient.trade_history`: immutable list of executed actions with timestamp and post-trade balance snapshot.

### Data used by market_type
- **spot**: loop prices from `ohlcv_spot` (timeframe from `time_step`).
- **futures**: loop on `index_ohlcv_futures` (derived timeframe) and use `mark_ohlcv_futures` for execution pricing and risk metrics.

Timeframe derivation: `Backtester._time_step_to_timeframe(...)` maps `time_step` to `{'1m','5m','15m','30m','1h'}`; defaults to `15m` if unmatched.

### OMS semantics (key points)
- `set_target_position`: interprets `value` as USDT to deploy at current price; creates/adjusts/close/flip positions.
- Futures do not move principal cash on open/adjust; spot subtracts cash. Portfolio value for futures adds unrealized PnL only; spot adds full notional value.

### Example: Hold strategy + PositionManager
Minimal usage (see `examples/` for full implementations):
```python
from datetime import datetime, timedelta, UTC
from typing import Any

from crypto_backtester_binance.backtester import Backtester
from crypto_backtester_binance.hist_data import HistoricalDataCollector


# User-defined strategy
class HoldStrategy:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.has_bought = False

    def run_strategy(self, oms_client, data_manager):
        if self.has_bought:
            return []
        self.has_bought = True
        return [{"symbol": s, "instrument_type": "future", "side": "LONG"} for s in self.symbols]


# User-defined position manager
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

start_date = datetime.now(UTC) - timedelta(days=30)
end_date = datetime.now(UTC)

results = bt.run_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(hours=1),
    market_type="futures",
)

# For permutation testing:
results = bt.run_permutation_backtest(
    strategy=strategy,
    position_manager=pm,
    start_date=start_date,
    end_date=end_date,
    time_step=timedelta(days=1),
    market_type="futures",
    permutations=3,
)
print("p_value:", results.get("p_value"))
```

See `examples/position_manager.py` for a reference implementation with volatility-based risk screening and inverse-vol position sizing.
