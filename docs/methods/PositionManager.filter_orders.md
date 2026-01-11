## Creating a Position Manager

The position manager is a **user-defined component** that processes orders from your strategy before execution. It handles risk management, position sizing, and budget enforcement.

### Required Interface

Your position manager must implement the `filter_orders` method:

```python
def filter_orders(
    self,
    orders: list[dict],
    oms_client: OMSClient,
    data_manager: HistoricalDataCollector
) -> list[dict] | None
```

### Parameters

- **orders**: List of order dictionaries from your strategy
- **oms_client**: Access to balance, positions, current time via:
  - `oms_client.balance["USDT"]` - current cash balance
  - `oms_client.positions` - current open positions
  - `oms_client.current_time` - current backtest timestamp
- **data_manager**: Access to historical data via `load_data_period()`

### Return Value

- List of order dicts with `value` field set (USDT notional amount)
- Return `None` to skip all orders for this timestep

### Example: Simple Equal-Weight

```python
class SimplePositionManager:
    def filter_orders(self, orders, oms_client, data_manager):
        if not orders:
            return None
        # Equal weight: 10% of balance split across orders
        budget = oms_client.balance["USDT"] * 0.1 / len(orders)
        return [{**order, "value": budget} for order in orders]
```

### Example: Risk-Adjusted Sizing

See `examples/position_manager.py` for a full reference implementation with:

1. **Risk screening**: Uses 4h volatility to filter high-risk orders
2. **Inverse-vol sizing**: Allocates more to lower-volatility assets
3. **Budget enforcement**: Ensures orders don't exceed available balance
4. **CLOSE order handling**: Always allows position exits

### Order Schema

Orders returned from `filter_orders` should have:

```python
{
    "symbol": "BTC-USDT",           # Asset symbol
    "instrument_type": "future",    # "spot" or "future"
    "side": "LONG",                 # "LONG", "SHORT", or "CLOSE"
    "value": 1000.0                 # USDT notional (required!)
}
```
