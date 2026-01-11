import logging
from typing import Any

from hist_data import HistoricalDataCollector
from oms_simulation import OMSClient

logger = logging.getLogger(__name__)


class V1LSPositionManager:
    def __init__(self):
        self.orders = []
        self.oms_client = None
        self.data_manager = None
        self.max_alloc_frac = 2000

    def _set_oms_and_dm(self, oms_client: Any, data_manager: HistoricalDataCollector) -> None:
        self.oms_client = oms_client
        self.data_manager = data_manager

    def _red_button(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Red button to close positions due to n% loss.
        """
        try:
            current_positions = self.oms_client.get_position() or []
            for position in current_positions:
                # Coerce string fields from OMS to floats for safe arithmetic
                try:
                    qty = float(position.get("quantity", 0.0))
                    entry_price = float(position.get("entry_price", 0.0))
                    current_value = float(position.get("value", 0.0))
                except (TypeError, ValueError):
                    continue

                threshold_value = 0.95 * entry_price * qty
                # checks if we lost more than 5% on a position (short or long)
                if current_value < threshold_value:
                    logger.info(
                        f"Closing position {position['symbol']} due to large loss of {position.get('pnl')}"
                    )
                    orders.append(
                        {
                            "symbol": position["symbol"],
                            "instrument_type": position["instrument_type"],
                            "side": "CLOSE",
                        }
                    )
            return orders
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return orders

    def _prioritize_close_orders(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_symbol = {}
        for o in orders:
            sym = o.get("symbol")
            if not sym:
                continue
            if o.get("side") == "CLOSE" or sym not in by_symbol or by_symbol[sym].get("side") != "CLOSE":
                by_symbol[sym] = o
        return list(by_symbol.values())

    def _set_weights(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not orders:
            return []
        sized: list[dict[str, Any]] = []
        for order in orders:
            side = order.get("side")
            # CLOSE orders don't need alloc_frac/value; pass through
            if side == "CLOSE":
                sized.append(order)
                continue
            alloc = order.get("alloc_frac", 0.0)
            try:
                order["value"] = float(self.max_alloc_frac) * float(alloc)
            except Exception:
                order["value"] = 0.0
            sized.append(order)
        return sized

    def filter_orders(
        self,
        orders: list[dict[str, Any]],
        oms_client: OMSClient,
        data_manager: HistoricalDataCollector,
    ) -> list[dict[str, Any]]:
        try:
            self._set_oms_and_dm(oms_client, data_manager)
            incoming = orders or []
            after_rb = self._red_button(incoming)
            after_weights = self._set_weights(after_rb)
            prioritized = self._prioritize_close_orders(after_weights)
            return prioritized or []
        except Exception as e:
            logger.error(f"Error filtering orders: {e}")
            return []
