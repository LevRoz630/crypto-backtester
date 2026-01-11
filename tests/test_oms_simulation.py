"""Comprehensive tests for OMS simulation."""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_backtester_binance.oms_simulation import OMSClient


@pytest.fixture
def oms_client(tmp_path):
    """Create OMS client instance."""
    return OMSClient(historical_data_dir=str(tmp_path))


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    dm = Mock()
    # Default: return sample price data
    dates = pd.date_range("2024-01-01", periods=1, freq="15min", tz="UTC")
    sample_data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [45000.0],
            "high": [45100.0],
            "low": [44900.0],
            "close": [45000.0],
            "volume": [100.0],
        }
    )
    dm.load_data_period = Mock(return_value=sample_data)
    return dm


class TestOMSInitialization:
    """Tests for OMS initialization."""

    def test_init_default(self, oms_client):
        """Test default initialization."""
        assert oms_client.positions == {}
        assert oms_client.balance == {"USDT": 10000.0}
        assert oms_client.trade_history == []
        assert oms_client.current_time is None
        assert oms_client.data_manager is None

    def test_init_custom_dir(self, tmp_path):
        """Test initialization with custom directory."""
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        oms = OMSClient(historical_data_dir=str(custom_dir))
        assert oms.historical_data_dir == custom_dir

    def test_set_current_time(self, oms_client):
        """Test setting current time."""
        test_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        oms_client.set_current_time(test_time)
        assert oms_client.current_time == test_time

    def test_set_data_manager(self, oms_client, mock_data_manager):
        """Test setting data manager."""
        oms_client.set_data_manager(mock_data_manager)
        assert oms_client.data_manager is mock_data_manager


class TestNormalizeSymbol:
    """Tests for symbol normalization."""

    def test_normalize_futures_symbol(self, oms_client):
        """Test futures symbol normalization removes -PERP."""
        assert oms_client._normalize_symbol("BTC-USDT-PERP", "future") == "BTC-USDT"
        assert oms_client._normalize_symbol("ETH-USDT-PERP", "future") == "ETH-USDT"

    def test_normalize_spot_symbol(self, oms_client):
        """Test spot symbol normalization unchanged."""
        assert oms_client._normalize_symbol("BTC-USDT", "spot") == "BTC-USDT"
        assert oms_client._normalize_symbol("ETH-USDT", "spot") == "ETH-USDT"


class TestGetCurrentPrice:
    """Tests for get_current_price method."""

    def test_get_price_spot(self, oms_client, mock_data_manager):
        """Test getting spot price."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        price = oms_client.get_current_price("BTC-USDT", "spot")
        assert price == 45000.0

    def test_get_price_futures(self, oms_client, mock_data_manager):
        """Test getting futures price."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        price = oms_client.get_current_price("BTC-USDT-PERP", "future")
        assert price == 45000.0

    def test_get_price_no_data_manager(self, oms_client):
        """Test price returns None without data manager."""
        price = oms_client.get_current_price("BTC-USDT", "spot")
        assert price is None

    def test_get_price_no_current_time(self, oms_client, mock_data_manager):
        """Test price returns None without current time."""
        oms_client.set_data_manager(mock_data_manager)
        price = oms_client.get_current_price("BTC-USDT", "spot")
        assert price is None

    def test_get_price_no_data_available(self, oms_client, mock_data_manager):
        """Test price returns None when no data available."""
        mock_data_manager.load_data_period = Mock(return_value=pd.DataFrame())
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        price = oms_client.get_current_price("BTC-USDT", "spot")
        assert price is None


class TestSetTargetPosition:
    """Tests for position setting."""

    def test_open_long_position(self, oms_client, mock_data_manager):
        """Test opening a long position."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        result = oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        assert result["status"] == "success"
        assert "BTC-USDT-PERP" in oms_client.positions
        assert oms_client.positions["BTC-USDT-PERP"]["side"] == "LONG"
        assert oms_client.positions["BTC-USDT-PERP"]["quantity"] > 0
        assert oms_client.balance["USDT"] == 9000.0  # 10000 - 1000
        assert len(oms_client.trade_history) == 1

    def test_open_short_position(self, oms_client, mock_data_manager):
        """Test opening a short position."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        result = oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "SHORT")

        assert result["status"] == "success"
        assert oms_client.positions["BTC-USDT-PERP"]["side"] == "SHORT"
        assert oms_client.balance["USDT"] == 9000.0

    def test_add_to_long_position(self, oms_client, mock_data_manager):
        """Test adding to existing long position."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open initial position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")
        initial_qty = oms_client.positions["BTC-USDT-PERP"]["quantity"]

        # Add to position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 500.0, "LONG")
        new_qty = oms_client.positions["BTC-USDT-PERP"]["quantity"]

        assert new_qty > initial_qty
        assert oms_client.balance["USDT"] == 8500.0  # 10000 - 1000 - 500

    def test_insufficient_balance(self, oms_client, mock_data_manager):
        """Test insufficient balance is handled gracefully."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Should return None when balance is insufficient (error is logged)
        result = oms_client.set_target_position("BTC-USDT-PERP", "future", 15000.0, "LONG")
        assert result is None
        assert oms_client.balance["USDT"] == 10000.0  # Balance unchanged

    def test_close_position(self, oms_client, mock_data_manager):
        """Test closing a position."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Close position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 0.0, "CLOSE")

        assert oms_client.positions["BTC-USDT-PERP"]["quantity"] == 0.0
        assert oms_client.positions["BTC-USDT-PERP"]["side"] == "CLOSE"
        # Balance should be restored (minus fees)
        assert oms_client.balance["USDT"] <= 10000.0

    def test_flip_position_long_to_short(self, oms_client, mock_data_manager):
        """Test flipping position from LONG to SHORT."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open LONG
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Flip to SHORT
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "SHORT")

        assert oms_client.positions["BTC-USDT-PERP"]["side"] == "SHORT"
        # Should have 2 trades (open long, flip to short)
        assert len(oms_client.trade_history) == 2

    def test_unsupported_instrument_type(self, oms_client, mock_data_manager):
        """Test unsupported instrument type."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        result = oms_client.set_target_position("BTC-USDT", "options", 1000.0, "LONG")
        # Should handle gracefully
        assert result is None


class TestClosePosition:
    """Tests for close_position method."""

    def test_close_long_position_profit(self, oms_client, mock_data_manager):
        """Test closing long position with profit."""
        # Setup: entry at 45000, close at 46000
        entry_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [45000.0],
            }
        )
        exit_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [46000.0],
            }
        )

        mock_data_manager.load_data_period = Mock(side_effect=[entry_data, exit_data])
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Advance time
        oms_client.set_current_time(datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Close
        pnl, principal = oms_client.close_position("BTC-USDT-PERP", "future")

        # Should have profit (price went up for LONG)
        assert pnl > 0
        assert principal > 0

    def test_close_long_position_loss(self, oms_client, mock_data_manager):
        """Test closing long position with loss."""
        # Entry at 45000, close at 44000
        entry_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [45000.0],
            }
        )
        exit_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [44000.0],
            }
        )

        mock_data_manager.load_data_period = Mock(side_effect=[entry_data, exit_data])
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Advance time
        oms_client.set_current_time(datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Close
        pnl, principal = oms_client.close_position("BTC-USDT-PERP", "future")

        # Should have loss (price went down for LONG)
        assert pnl < 0
        assert principal > 0

    def test_close_short_position_profit(self, oms_client, mock_data_manager):
        """Test closing short position with profit."""
        # Entry at 45000, close at 44000 (price down = profit for SHORT)
        entry_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [45000.0],
            }
        )
        exit_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [44000.0],
            }
        )

        # Need 3 calls: first for checking price, second for SHORT position itself, third for close
        # (Opening SHORT when default is LONG triggers a position flip which calls close internally)
        mock_data_manager.load_data_period = Mock(side_effect=[entry_data, entry_data, exit_data])
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open SHORT position (this actually does NOT trigger a flip since quantity is 0)
        # Just use return_value instead
        mock_data_manager.load_data_period = Mock(return_value=entry_data)
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "SHORT")

        # Advance time
        oms_client.set_current_time(datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))
        mock_data_manager.load_data_period = Mock(return_value=exit_data)

        # Close
        pnl, principal = oms_client.close_position("BTC-USDT-PERP", "future")

        # Should have profit (price went down for SHORT)
        assert pnl > 0


class TestGetPosition:
    """Tests for get_position method."""

    def test_get_position_empty(self, oms_client):
        """Test getting positions when none exist."""
        positions = oms_client.get_position()
        assert positions == []

    def test_get_position_with_open_position(self, oms_client, mock_data_manager):
        """Test getting position with open position."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        positions = oms_client.get_position()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "BTC-USDT-PERP"
        assert positions[0]["position_side"] == "LONG"
        assert "pnl" in positions[0]

    def test_get_position_excludes_zero_positions(self, oms_client, mock_data_manager):
        """Test that closed positions are excluded."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open and close
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")
        oms_client.set_target_position("BTC-USDT-PERP", "future", 0.0, "CLOSE")

        positions = oms_client.get_position()
        assert len(positions) == 0


class TestUpdatePortfolioValue:
    """Tests for portfolio value calculation."""

    def test_portfolio_value_no_positions(self, oms_client):
        """Test portfolio value with no positions."""
        value = oms_client.update_portfolio_value()
        assert value == 10000.0

    def test_portfolio_value_with_long_profit(self, oms_client, mock_data_manager):
        """Test portfolio value with profitable long position."""
        # Entry at 45000, current at 46000
        entry_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [45000.0],
            }
        )
        current_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [46000.0],
            }
        )

        mock_data_manager.load_data_period = Mock(side_effect=[entry_data, current_data])
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Advance time
        oms_client.set_current_time(datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Get value
        value = oms_client.update_portfolio_value()

        # Should be greater than initial (profit from price increase)
        assert value > 10000.0

    def test_portfolio_value_with_long_loss(self, oms_client, mock_data_manager):
        """Test portfolio value with losing long position."""
        # Entry at 45000, current at 44000
        entry_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [45000.0],
            }
        )
        current_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)],
                "close": [44000.0],
            }
        )

        mock_data_manager.load_data_period = Mock(side_effect=[entry_data, current_data])
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        # Advance time
        oms_client.set_current_time(datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Get value
        value = oms_client.update_portfolio_value()

        # Should be less than initial (loss from price decrease)
        assert value < 10000.0


class TestGetPositionSummary:
    """Tests for position summary."""

    def test_summary_no_positions(self, oms_client):
        """Test summary with no positions."""
        summary = oms_client.get_position_summary()

        assert "balances" in summary
        assert summary["balances"]["USDT"] == 10000.0
        assert summary["positions"] == {}
        assert summary["total_trades"] == 0

    def test_summary_with_positions(self, oms_client, mock_data_manager):
        """Test summary with open positions."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Open position
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        summary = oms_client.get_position_summary()

        assert "BTC-USDT-PERP" in summary["positions"]
        assert summary["total_trades"] == 1
        assert summary["total_portfolio_value"] > 0


class TestTradeHistory:
    """Tests for trade history tracking."""

    def test_trade_history_empty(self, oms_client):
        """Test empty trade history."""
        assert oms_client.trade_history == []

    def test_trade_history_records_trades(self, oms_client, mock_data_manager):
        """Test that trades are recorded."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Execute trades
        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")
        oms_client.set_target_position("ETH-USDT-PERP", "future", 500.0, "LONG")

        assert len(oms_client.trade_history) == 2
        assert oms_client.trade_history[0]["symbol"] == "BTC-USDT-PERP"
        assert oms_client.trade_history[1]["symbol"] == "ETH-USDT-PERP"

    def test_trade_history_contains_details(self, oms_client, mock_data_manager):
        """Test trade history contains expected fields."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")

        trade = oms_client.trade_history[0]
        assert "timestamp" in trade
        assert "symbol" in trade
        assert "type" in trade
        assert "side" in trade
        assert "quantity" in trade
        assert "value" in trade
        assert "price" in trade
        assert "balance_after" in trade


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_position_with_zero_quantity(self, oms_client, mock_data_manager):
        """Test handling position with zero quantity."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Manually set position with zero quantity
        oms_client.positions["BTC-USDT-PERP"] = {
            "quantity": 0.0,
            "value": 0.0,
            "side": "LONG",
            "entry_price": 45000.0,
            "pnl": 0.0,
            "instrument_type": "future",
        }

        positions = oms_client.get_position()
        assert len(positions) == 0  # Zero positions should be filtered

    def test_multiple_positions(self, oms_client, mock_data_manager):
        """Test managing multiple positions simultaneously."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        oms_client.set_target_position("BTC-USDT-PERP", "future", 1000.0, "LONG")
        oms_client.set_target_position("ETH-USDT-PERP", "future", 1000.0, "SHORT")
        oms_client.set_target_position("SOL-USDT-PERP", "future", 1000.0, "LONG")

        assert len(oms_client.positions) == 3
        assert oms_client.balance["USDT"] == 7000.0

    def test_balance_never_negative(self, oms_client, mock_data_manager):
        """Test that balance checks prevent negative balance."""
        oms_client.set_data_manager(mock_data_manager)
        oms_client.set_current_time(datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz))

        # Try to spend more than balance (error is logged, returns None)
        result = oms_client.set_target_position("BTC-USDT-PERP", "future", 20000.0, "LONG")

        # Should return None when error occurs
        assert result is None
        # Balance should remain unchanged
        assert oms_client.balance["USDT"] == 10000.0
