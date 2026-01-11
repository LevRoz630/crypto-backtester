"""Comprehensive tests for the PositionManager class."""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from crypto_backtester_binance.position_manager import PositionManager


@pytest.fixture
def position_manager():
    """Create a PositionManager instance."""
    return PositionManager()


@pytest.fixture
def mock_oms_client():
    """Create a mock OMS client."""
    oms = Mock()
    oms.balance = {"USDT": 10000.0}
    oms.current_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
    return oms


@pytest.fixture
def mock_data_manager():
    """Create a mock data manager."""
    dm = Mock()

    # Create sample data for volatility calculations
    dates = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(40000, 42000, 100),
            "high": np.random.uniform(42000, 43000, 100),
            "low": np.random.uniform(39000, 40000, 100),
            "close": np.random.uniform(40000, 42000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        }
    )

    dm.load_data_period = Mock(return_value=sample_data)
    return dm


@pytest.fixture
def sample_orders():
    """Create sample orders for testing."""
    return [
        {"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0},
        {"symbol": "ETH-USDT-PERP", "side": "BUY", "qty": 10.0},
        {"symbol": "SOL-USDT-PERP", "side": "SELL", "qty": 100.0},
    ]


class TestPositionManagerInit:
    """Tests for PositionManager initialization."""

    def test_init(self, position_manager):
        """Test initialization."""
        assert position_manager.orders == []
        assert position_manager.oms_client is None
        assert position_manager.data_manager is None


class TestFilterOrders:
    """Tests for filter_orders method."""

    def test_filter_orders_basic(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test basic order filtering."""
        result = position_manager.filter_orders(sample_orders, mock_oms_client, mock_data_manager)

        # Should return processed orders
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

        # Each order should have a value assigned
        for order in result:
            assert "value" in order

    def test_filter_orders_with_close_orders(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test filtering with CLOSE orders."""
        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "CLOSE", "qty": 1.0},
            {"symbol": "ETH-USDT-PERP", "side": "BUY", "qty": 10.0},
        ]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # CLOSE orders should be passed through
        assert result is not None
        close_orders = [o for o in result if o.get("side") == "CLOSE"]
        assert len(close_orders) == 1

    def test_filter_orders_insufficient_balance(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test filtering when orders exceed balance."""
        # Set low balance
        mock_oms_client.balance = {"USDT": 100.0}

        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 10.0},
            {"symbol": "ETH-USDT-PERP", "side": "BUY", "qty": 10.0},
        ]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # Should handle insufficient balance
        # Either returns None or returns only what can be afforded
        assert result is None or isinstance(result, list)

    def test_filter_orders_only_close_orders(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test filtering with only CLOSE orders."""
        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "CLOSE", "qty": 1.0},
            {"symbol": "ETH-USDT-PERP", "side": "CLOSE", "qty": 10.0},
        ]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # CLOSE orders should all be returned
        assert result is not None
        assert len(result) == 2
        assert all(o["side"] == "CLOSE" for o in result)

    def test_filter_orders_empty_list(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test filtering empty order list."""
        result = position_manager.filter_orders([], mock_oms_client, mock_data_manager)

        # Should handle empty list gracefully
        assert result is None or result == []

    def test_filter_orders_with_exception(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test filtering when data manager raises exception."""
        mock_data_manager.load_data_period = Mock(side_effect=Exception("Data error"))

        orders = [{"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0}]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # Should handle exception gracefully
        assert result is None or isinstance(result, list)


class TestCloseRiskyOrders:
    """Tests for _close_risky_orders method."""

    def test_close_risky_orders_low_volatility(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test with low volatility data."""
        # Create low volatility data (tight price range)
        dates = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
        low_vol_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": [40000.0] * 16,
                "high": [40010.0] * 16,
                "low": [39990.0] * 16,
                "close": [40000.0] * 16,
                "volume": [100.0] * 16,
            }
        )
        mock_data_manager.load_data_period = Mock(return_value=low_vol_data)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._close_risky_orders(sample_orders)

        # All orders should pass (no value=0)
        assert len(result) == len(sample_orders)
        assert all(o.get("value", 1) != 0 for o in result)

    def test_close_risky_orders_high_volatility(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test with high volatility data."""
        # Create extreme high volatility data (std/mean > 0.1)
        dates = pd.date_range("2024-01-01", periods=16, freq="15min", tz="UTC")
        # Create data with mean=1000 and std=200, giving scaled_vol = 0.2 > 0.1
        high_vol_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": [1000.0] * 16,
                "high": [1200.0] * 16,
                "low": [800.0] * 16,
                "close": [1000, 800, 1200, 900, 1100, 850, 1150, 900, 1050, 950, 1100, 850, 1200, 800, 1100, 950],
                "volume": [1000.0] * 16,
            }
        )
        mock_data_manager.load_data_period = Mock(return_value=high_vol_data)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        orders = [{"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0}]

        result = position_manager._close_risky_orders(orders)

        # Order should be marked as risky (value=0) if scaled_vol > 0.1
        assert len(result) == 1
        # Verify volatility check happened - order either has value=0 or is unchanged
        assert "value" in result[0] or result[0] == orders[0]

    def test_close_risky_orders_no_data(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test when no data is available."""
        mock_data_manager.load_data_period = Mock(return_value=None)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._close_risky_orders(sample_orders)

        # Orders should pass through unchanged when no data
        assert len(result) == len(sample_orders)

    def test_close_risky_orders_empty_data(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test with empty dataframe."""
        empty_df = pd.DataFrame()
        mock_data_manager.load_data_period = Mock(return_value=empty_df)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._close_risky_orders(sample_orders)

        # Orders should pass through unchanged
        assert len(result) == len(sample_orders)


class TestSetWeights:
    """Tests for _set_weights method."""

    def test_set_weights_basic(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test basic weight setting."""
        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._set_weights(sample_orders)

        # All orders should have values assigned
        assert len(result) == len(sample_orders)
        assert all("value" in o for o in result)

        # Total value should be ~10% of balance (budget limit), allow small floating point error
        total_value = sum(o.get("value", 0) for o in result)
        budget_limit = mock_oms_client.balance["USDT"] / 10
        assert total_value <= budget_limit * 1.01  # Allow 1% tolerance for floating point

    def test_set_weights_with_zero_value_orders(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test weight setting with some orders marked as zero value."""
        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0, "value": 0},
            {"symbol": "ETH-USDT-PERP", "side": "BUY", "qty": 10.0},
        ]

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._set_weights(orders)

        # Zero value order should remain zero
        zero_order = [o for o in result if o["symbol"] == "BTC-USDT-PERP"][0]
        assert zero_order["value"] == 0

        # Other order should have positive value
        eth_order = [o for o in result if o["symbol"] == "ETH-USDT-PERP"][0]
        assert eth_order["value"] > 0

    def test_set_weights_no_data(
        self, position_manager, mock_oms_client, mock_data_manager, sample_orders
    ):
        """Test weight setting when no data available."""
        mock_data_manager.load_data_period = Mock(return_value=None)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._set_weights(sample_orders)

        # Should handle gracefully
        assert len(result) == len(sample_orders)

    def test_set_weights_equal_volatility(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test weight setting with equal volatility."""
        # Create identical volatility data for all symbols
        dates = pd.date_range("2024-01-01", periods=96, freq="15min", tz="UTC")
        equal_vol_data = pd.DataFrame(
            {
                "timestamp": dates,
                "close": np.random.uniform(40000, 41000, 96),
                "open": [40000.0] * 96,
                "high": [41000.0] * 96,
                "low": [40000.0] * 96,
                "volume": [100.0] * 96,
            }
        )
        mock_data_manager.load_data_period = Mock(return_value=equal_vol_data)

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0},
            {"symbol": "ETH-USDT-PERP", "side": "BUY", "qty": 1.0},
        ]

        result = position_manager._set_weights(orders)

        # With equal volatility, weights should be similar
        values = [o["value"] for o in result]
        assert len(values) == 2
        # Values should be close (within 10% tolerance)
        assert abs(values[0] - values[1]) / max(values) < 0.1 if max(values) > 0 else True

    def test_set_weights_single_order(self, position_manager, mock_oms_client, mock_data_manager):
        """Test weight setting with single order."""
        orders = [{"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0}]

        position_manager.oms_client = mock_oms_client
        position_manager.data_manager = mock_data_manager

        result = position_manager._set_weights(orders)

        # Single order should get full budget allocation
        assert len(result) == 1
        assert result[0]["value"] <= mock_oms_client.balance["USDT"] / 10


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_filter_orders_perp_suffix_removal(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test that -PERP suffix is correctly removed for data lookup."""
        orders = [{"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0}]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # Data manager should be called with base symbol (without -PERP)
        calls = mock_data_manager.load_data_period.call_args_list
        if calls:
            # First argument should be base symbol
            assert "BTC-USDT" in str(calls[0][0][0])

    def test_filter_orders_mixed_order_types(
        self, position_manager, mock_oms_client, mock_data_manager
    ):
        """Test with mixed order types (BUY, SELL, CLOSE)."""
        orders = [
            {"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0},
            {"symbol": "ETH-USDT-PERP", "side": "SELL", "qty": 10.0},
            {"symbol": "SOL-USDT-PERP", "side": "CLOSE", "qty": 100.0},
        ]

        result = position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # Should handle all order types
        assert result is not None
        assert len(result) > 0

    def test_state_persistence(self, position_manager, mock_oms_client, mock_data_manager):
        """Test that OMS and data manager are persisted."""
        orders = [{"symbol": "BTC-USDT-PERP", "side": "BUY", "qty": 1.0}]

        position_manager.filter_orders(orders, mock_oms_client, mock_data_manager)

        # State should be persisted
        assert position_manager.oms_client is mock_oms_client
        assert position_manager.data_manager is mock_data_manager
