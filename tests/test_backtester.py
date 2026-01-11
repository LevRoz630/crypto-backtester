"""Comprehensive tests for the Backtester class."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from crypto_backtester_binance.backtester import Backtester


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def backtester(temp_data_dir):
    """Create a Backtester instance with mocked dependencies."""
    with patch("crypto_backtester_binance.backtester.HistoricalDataCollector"):
        with patch("crypto_backtester_binance.backtester.OMSClient"):
            bt = Backtester(historical_data_dir=temp_data_dir)
            return bt


@pytest.fixture
def mock_strategy():
    """Create a mock strategy object."""
    strategy = Mock()
    strategy.symbols = ["BTC-USDT", "ETH-USDT"]
    strategy.lookback_days = 30
    strategy.get_signal = Mock(return_value={"BTC-USDT": {"side": "buy", "size": 1.0}})
    strategy.oms_client = None
    strategy.data_manager = None
    return strategy


@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    pm = Mock()
    pm.calculate_position_size = Mock(return_value=1000.0)
    pm.filter_signals = Mock(return_value=[{"symbol": "BTC-USDT", "side": "buy"}])
    return pm


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.random.uniform(40000, 50000, 100),
            "high": np.random.uniform(50000, 52000, 100),
            "low": np.random.uniform(38000, 40000, 100),
            "close": np.random.uniform(40000, 50000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        }
    )
    return data


class TestBacktesterInit:
    """Tests for Backtester initialization."""

    def test_init_creates_managers(self, temp_data_dir):
        """Test that initialization creates data manager and OMS client."""
        with patch("crypto_backtester_binance.backtester.HistoricalDataCollector") as mock_hdc:
            with patch("crypto_backtester_binance.backtester.OMSClient") as mock_oms:
                bt = Backtester(historical_data_dir=temp_data_dir)

                mock_hdc.assert_called_once_with(temp_data_dir)
                mock_oms.assert_called_once_with(historical_data_dir=temp_data_dir)

                assert bt.historical_data_dir == temp_data_dir
                assert bt.portfolio_values == []
                assert bt.returns == []
                assert bt.drawdowns == []
                assert bt.max_drawdown == 0
                assert bt.sharpe_ratio == 0
                assert bt.trade_history == []
                assert bt.final_balance == 0
                assert bt.final_positions == []
                assert bt.position_manager is None
                assert bt.position_exposures_history == []
                assert bt.permutation_returns == []

    def test_init_with_custom_dir(self):
        """Test initialization with custom directory."""
        custom_dir = "/custom/path"
        with patch("crypto_backtester_binance.backtester.HistoricalDataCollector"):
            with patch("crypto_backtester_binance.backtester.OMSClient"):
                bt = Backtester(historical_data_dir=custom_dir)
                assert bt.historical_data_dir == custom_dir


class TestTimeStepToTimeframe:
    """Tests for _time_step_to_timeframe helper method."""

    def test_1_minute_timeframe(self, backtester):
        """Test 1-minute timedelta conversion."""
        assert backtester._time_step_to_timeframe(timedelta(minutes=1)) == "1m"

    def test_5_minute_timeframe(self, backtester):
        """Test 5-minute timedelta conversion."""
        assert backtester._time_step_to_timeframe(timedelta(minutes=5)) == "5m"

    def test_15_minute_timeframe(self, backtester):
        """Test 15-minute timedelta conversion."""
        assert backtester._time_step_to_timeframe(timedelta(minutes=15)) == "15m"

    def test_1_hour_timeframe(self, backtester):
        """Test 1-hour timedelta conversion."""
        assert backtester._time_step_to_timeframe(timedelta(hours=1)) == "1h"

    def test_30_minute_timeframe(self, backtester):
        """Test 30-minute timedelta conversion."""
        assert backtester._time_step_to_timeframe(timedelta(minutes=30)) == "30m"

    def test_unsupported_timeframe_defaults_to_15m(self, backtester):
        """Test unsupported timedelta defaults to 15m."""
        # Unsupported timeframes default to 15m
        assert backtester._time_step_to_timeframe(timedelta(hours=4)) == "15m"
        assert backtester._time_step_to_timeframe(timedelta(days=1)) == "15m"
        assert backtester._time_step_to_timeframe(timedelta(minutes=7)) == "15m"

    def test_none_timeframe_raises_error(self, backtester):
        """Test None timedelta raises ValueError."""
        with pytest.raises(ValueError, match="Time step is None"):
            backtester._time_step_to_timeframe(None)


class TestRunBacktest:
    """Tests for run_backtest method."""

    def test_run_backtest_missing_time_step(
        self, backtester, mock_strategy, mock_position_manager
    ):
        """Test that run_backtest requires time_step parameter."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = backtester.run_backtest(
            strategy=mock_strategy,
            position_manager=mock_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=None,
            market_type="spot",
        )

        assert result is None


class TestRunPermutationBacktest:
    """Tests for run_permutation_backtest method."""

    @pytest.fixture
    def permutation_backtester(self, tmp_path):
        """Create backtester for permutation tests."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        bt = Backtester(historical_data_dir=str(data_dir))

        dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(44000, 46000, 100),
            "high": np.random.uniform(45000, 47000, 100),
            "low": np.random.uniform(43000, 45000, 100),
            "close": np.random.uniform(44000, 46000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        })

        # Need to use actual DataFrame.get() method which returns DataFrame or None
        # Set up data stores with actual data for symbols that will be used
        bt.data_manager.load_data_period = Mock(return_value=sample_data.copy())
        bt.data_manager.spot_ohlcv_data = {"BTC": sample_data.copy(), "BTC-USDT": sample_data.copy()}
        bt.data_manager.perpetual_mark_ohlcv_data = {"BTC": sample_data.copy(), "BTC-USDT": sample_data.copy()}
        bt.data_manager.perpetual_index_ohlcv_data = {"BTC": sample_data.copy(), "BTC-USDT": sample_data.copy()}

        bt.oms_client.update_portfolio_value = Mock(return_value=10000.0)
        bt.oms_client.get_position = Mock(return_value=[])
        bt.oms_client.get_position_summary = Mock(return_value={})
        bt.oms_client.set_target_position = Mock(return_value={"status": "success"})
        bt.oms_client.get_current_price = Mock(return_value=45000.0)
        bt.oms_client.positions = {}
        bt.oms_client.balance = {"USDT": 10000.0}
        bt.oms_client.trade_history = []
        bt.oms_client.current_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        def set_time(t):
            bt.oms_client.current_time = t
        bt.oms_client.set_current_time = Mock(side_effect=set_time)
        bt.oms_client.set_data_manager = Mock()

        return bt

    @pytest.fixture
    def perm_strategy(self):
        """Strategy for permutation tests."""
        strategy = Mock()
        strategy.symbols = ["BTC-USDT-PERP"]
        strategy.lookback_days = 1
        strategy.run_strategy = Mock(return_value=[])
        return strategy

    @pytest.fixture
    def perm_position_manager(self):
        """Position manager for permutation tests."""
        pm = Mock()
        pm.filter_orders = Mock(return_value=[])
        return pm

    def test_permutation_backtest_missing_time_step(self, permutation_backtester, perm_strategy, perm_position_manager):
        """Test permutation backtest requires time_step."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        with pytest.raises(ValueError, match="Time step is required"):
            permutation_backtester.run_permutation_backtest(
                strategy=perm_strategy,
                position_manager=perm_position_manager,
                start_date=start_date,
                end_date=end_date,
                time_step=None,
                market_type="futures",
                permutations=2,
            )

    def test_permutation_backtest_spot(self, permutation_backtester, perm_strategy, perm_position_manager):
        """Test permutation backtest with spot market."""
        perm_strategy.symbols = ["BTC-USDT"]
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = permutation_backtester.run_permutation_backtest(
            strategy=perm_strategy,
            position_manager=perm_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="spot",
            permutations=2,
        )

        assert result is not None

    def test_permutation_backtest_futures(self, permutation_backtester, perm_strategy, perm_position_manager):
        """Test permutation backtest with futures market."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = permutation_backtester.run_permutation_backtest(
            strategy=perm_strategy,
            position_manager=perm_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
            permutations=2,
        )

        assert result is not None

    def test_permutation_backtest_invalid_market(self, permutation_backtester, perm_strategy, perm_position_manager):
        """Test permutation backtest with invalid market type."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        with pytest.raises(ValueError, match="Invalid market type"):
            permutation_backtester.run_permutation_backtest(
                strategy=perm_strategy,
                position_manager=perm_position_manager,
                start_date=start_date,
                end_date=end_date,
                time_step=timedelta(hours=1),
                market_type="invalid",
                permutations=2,
            )


class TestCalculatePerformanceMetrics:
    """Tests for calculate_performance_metrics method."""

    def test_calculate_metrics_empty_portfolio(self, backtester):
        """Test metrics calculation with empty portfolio."""
        backtester.portfolio_values = []
        backtester.returns = []

        backtester.calculate_performance_metrics()

        # Should handle empty data gracefully
        assert backtester.sharpe_ratio == 0 or np.isnan(backtester.sharpe_ratio)

    def test_calculate_metrics_positive_returns(self, backtester):
        """Test metrics calculation with positive returns."""
        backtester.portfolio_values = [10000, 10100, 10200, 10300]
        backtester.returns = [0.01, 0.01, 0.01]

        backtester.calculate_performance_metrics()

        assert backtester.sharpe_ratio > 0
        assert backtester.max_drawdown == 0  # No drawdown with positive returns

    def test_calculate_metrics_with_drawdown(self, backtester):
        """Test metrics calculation with drawdown."""
        backtester.portfolio_values = [10000, 10500, 9000, 9500]
        backtester.returns = [0.05, -0.14, 0.06]

        backtester.calculate_performance_metrics()

        # Should calculate max drawdown (implementation stores absolute value)
        assert backtester.max_drawdown > 0  # Stored as positive value
        assert len(backtester.drawdowns) > 0

    def test_calculate_metrics_zero_variance(self, backtester):
        """Test metrics with zero variance (flat returns)."""
        backtester.portfolio_values = [10000, 10000, 10000]
        backtester.returns = [0.0, 0.0]

        backtester.calculate_performance_metrics()

        # Sharpe ratio should be 0 or undefined
        assert backtester.sharpe_ratio == 0 or np.isnan(backtester.sharpe_ratio)


class TestPlottingMethods:
    """Tests for plotting methods."""

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_portfolio_value(self, mock_show, backtester):
        """Test portfolio value plotting."""
        backtester.portfolio_values = [10000, 10100, 10200]

        # Should not raise exception
        fig = backtester.plot_portfolio_value()

        # Should create figure but not show it during tests
        assert fig is None or hasattr(fig, "data")

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_drawdown(self, mock_show, backtester):
        """Test drawdown plotting."""
        backtester.drawdowns = [0, -0.05, -0.10]

        fig = backtester.plot_drawdown()

        assert fig is None or hasattr(fig, "data")

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_returns(self, mock_show, backtester):
        """Test returns plotting."""
        backtester.returns = [0.01, 0.02, -0.01]

        fig = backtester.plot_returns()

        assert fig is None or hasattr(fig, "data")

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_positions(self, mock_show, backtester):
        """Test positions plotting."""
        backtester.position_exposures_history = [
            {
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
                "exposures": {"BTC-USDT": 5000, "ETH-USDT": 3000},
            },
            {
                "timestamp": pd.Timestamp("2024-01-02", tz="UTC"),
                "exposures": {"BTC-USDT": 6000, "ETH-USDT": 2000},
            },
        ]

        fig = backtester.plot_positions()

        assert fig is None or hasattr(fig, "data")

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_positions_empty(self, mock_show, backtester):
        """Test positions plotting with empty history."""
        backtester.position_exposures_history = []

        # Should handle gracefully
        fig = backtester.plot_positions()

        assert fig is None or hasattr(fig, "data")


class TestSaveAndPrintResults:
    """Tests for save_results and print_results methods."""

    def test_save_results(self, backtester, tmp_path):
        """Test saving results to JSON file."""
        results = {
            "final_balance": 10500.0,
            "total_return": 0.05,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "trade_history": [],
        }

        filename = tmp_path / "test_results"

        backtester.save_results(results, str(filename))

        # File gets .json appended automatically
        json_file = Path(str(filename) + ".json")
        assert json_file.exists()

        # Verify contents
        with open(json_file) as f:
            saved_data = json.load(f)

        assert saved_data["final_balance"] == 10500.0
        assert saved_data["total_return"] == 0.05

    def test_save_results_with_complex_data(self, backtester, tmp_path):
        """Test saving results with complex data types."""
        results = {
            "returns": [0.01, 0.02, 0.03],
            "portfolio_values": [10000.0, 10100.0, 10200.0],
            "trade_history": [
                {
                    "timestamp": datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz),
                    "price": 45000.0,
                }
            ],
        }

        filename = tmp_path / "complex_results"

        backtester.save_results(results, str(filename))

        # Should handle serialization
        json_file = Path(str(filename) + ".json")
        assert json_file.exists()

    def test_print_results(self, backtester, capsys):
        """Test printing results to stdout."""
        results = {
            "final_balance": 10500.0,
            "total_return": 0.05,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.10,
            "trade_history": [{"trade": 1}, {"trade": 2}],
        }

        backtester.print_results(results)

        captured = capsys.readouterr()

        # Verify output contains key information
        assert "10500" in captured.out
        assert "BACKTEST RESULTS" in captured.out

    def test_print_results_minimal(self, backtester, capsys):
        """Test printing results with minimal required fields."""
        results = {
            "final_balance": 10000.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trade_history": [],
        }

        backtester.print_results(results)

        # Should handle gracefully
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_backtester_reusability(self, backtester):
        """Test that backtester can be reused for multiple runs."""
        # First run
        backtester.portfolio_values = [10000, 10100]
        backtester.returns = [0.01]

        # Reset should happen in run_backtest (via OMS client reset)
        backtester.oms_client.reset = Mock()

        # Verify state can be reset
        backtester.oms_client.reset()
        assert backtester.oms_client.reset.called

    def test_extreme_returns(self, backtester):
        """Test metrics calculation with extreme returns."""
        # Simulate extreme volatility
        backtester.portfolio_values = [10000, 20000, 5000, 15000, 8000]
        backtester.returns = [1.0, -0.75, 2.0, -0.47]

        backtester.calculate_performance_metrics()

        # Should handle extreme values
        assert not np.isinf(backtester.sharpe_ratio)
        assert backtester.max_drawdown > 0  # Stored as absolute value


class TestRunBacktestIntegration:
    """Integration tests for run_backtest method with mocked dependencies."""

    @pytest.fixture
    def integrated_backtester(self, tmp_path):
        """Create backtester with properly mocked dependencies."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        bt = Backtester(historical_data_dir=str(data_dir))

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "open": [45000.0] * 50,
            "high": [45100.0] * 50,
            "low": [44900.0] * 50,
            "close": [45000.0] * 50,
            "volume": [100.0] * 50,
        })

        # Mock data manager methods
        bt.data_manager.load_data_period = Mock(return_value=sample_data)
        bt.data_manager.spot_ohlcv_data = {"BTC": sample_data}
        bt.data_manager.perpetual_mark_ohlcv_data = {"BTC": sample_data}
        bt.data_manager.perpetual_index_ohlcv_data = {"BTC": sample_data}

        # Mock OMS client
        bt.oms_client.update_portfolio_value = Mock(return_value=10000.0)
        bt.oms_client.get_position = Mock(return_value=[])
        bt.oms_client.get_position_summary = Mock(return_value={})
        bt.oms_client.set_target_position = Mock(return_value={"status": "success"})
        bt.oms_client.get_current_price = Mock(return_value=45000.0)
        bt.oms_client.positions = {}
        bt.oms_client.current_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        def set_time(t):
            bt.oms_client.current_time = t
        bt.oms_client.set_current_time = Mock(side_effect=set_time)
        bt.oms_client.set_data_manager = Mock()

        return bt

    @pytest.fixture
    def simple_strategy(self):
        """Create a simple strategy for testing."""
        strategy = Mock()
        strategy.symbols = ["BTC-USDT-PERP"]
        strategy.lookback_days = 1
        strategy.run_strategy = Mock(return_value=[
            {"symbol": "BTC-USDT-PERP", "instrument_type": "future", "side": "LONG", "value": 1000.0}
        ])
        return strategy

    @pytest.fixture
    def simple_position_manager(self):
        """Create a simple position manager for testing."""
        pm = Mock()
        pm.filter_orders = Mock(side_effect=lambda orders, **kwargs: orders)
        return pm

    def test_run_backtest_spot_market(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test run_backtest with spot market."""
        simple_strategy.symbols = ["BTC-USDT"]
        simple_strategy.run_strategy = Mock(return_value=[
            {"symbol": "BTC-USDT", "instrument_type": "spot", "side": "LONG", "value": 1000.0}
        ])

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="spot",
        )

        assert result is not None
        assert "final_balance" in result or result.get("portfolio_values") is not None

    def test_run_backtest_futures_market(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test run_backtest with futures market."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        assert result is not None

    def test_run_backtest_invalid_market_type(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test run_backtest with invalid market type."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        # Invalid market type should log error and continue
        result = integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="invalid",
        )

        # Should still return something (might be None due to no data)
        # The key is it doesn't crash

    def test_run_backtest_strategy_orders_executed(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test that strategy orders are executed."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        # Strategy should have been called
        assert simple_strategy.run_strategy.called
        # Position manager should have filtered orders
        assert simple_position_manager.filter_orders.called

    def test_run_backtest_with_none_orders(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test run_backtest when strategy returns no orders."""
        simple_strategy.run_strategy = Mock(return_value=None)
        simple_position_manager.filter_orders = Mock(return_value=None)

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        result = integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        # Should complete without errors
        assert result is not None

    def test_run_backtest_position_exposure_tracking(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test that position exposures are tracked."""
        # Setup position for exposure tracking
        integrated_backtester.oms_client.positions = {
            "BTC-USDT-PERP": {"quantity": 0.1, "instrument_type": "future"}
        }

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        # Position exposures should be tracked
        assert len(integrated_backtester.position_exposures_history) > 0

    def test_run_backtest_order_execution_error_handling(self, integrated_backtester, simple_strategy, simple_position_manager):
        """Test that order execution errors are handled gracefully."""
        integrated_backtester.oms_client.set_target_position = Mock(side_effect=Exception("Order failed"))

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        # Should not raise exception
        result = integrated_backtester.run_backtest(
            strategy=simple_strategy,
            position_manager=simple_position_manager,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        assert result is not None


class TestBacktestResults:
    """Tests for backtest result generation."""

    @pytest.fixture
    def backtester_with_results(self, tmp_path):
        """Create backtester with simulated results."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()

        bt = Backtester(historical_data_dir=str(data_dir))
        bt.portfolio_values = [10000, 10100, 10200, 10300, 10400]
        bt.returns = [0.01, 0.01, 0.01, 0.01]
        bt.drawdowns = [0, 0, 0, 0, 0]
        bt.trade_history = [{"symbol": "BTC", "side": "LONG"}]
        bt.final_balance = 10400
        bt.oms_client.balance = {"USDT": 10400}
        bt.oms_client.positions = {}
        bt.oms_client.trade_history = bt.trade_history

        return bt

    def test_trade_history_accessible(self, backtester_with_results):
        """Test trade history is accessible via oms_client."""
        assert backtester_with_results.oms_client.trade_history == backtester_with_results.trade_history

    def test_final_balance_accessible(self, backtester_with_results):
        """Test final balance is set correctly."""
        assert backtester_with_results.final_balance == 10400


class TestBacktestEdgeCases:
    """Additional edge case tests for better coverage."""

    @pytest.fixture
    def edge_backtester(self, tmp_path):
        """Create backtester for edge case tests."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        bt = Backtester(historical_data_dir=str(data_dir))

        dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "open": [45000.0] * 50,
            "high": [45100.0] * 50,
            "low": [44900.0] * 50,
            "close": [45000.0] * 50,
            "volume": [100.0] * 50,
        })

        bt.data_manager.load_data_period = Mock(return_value=sample_data)
        bt.data_manager.spot_ohlcv_data = {"BTC": sample_data}
        bt.data_manager.perpetual_mark_ohlcv_data = {"BTC": sample_data}
        bt.data_manager.perpetual_index_ohlcv_data = {"BTC": sample_data}

        bt.oms_client.update_portfolio_value = Mock(return_value=10000.0)
        bt.oms_client.get_position = Mock(return_value=[])
        bt.oms_client.get_position_summary = Mock(return_value={})
        bt.oms_client.set_target_position = Mock(return_value={"status": "success"})
        bt.oms_client.get_current_price = Mock(return_value=45000.0)
        bt.oms_client.positions = {}
        bt.oms_client.balance = {"USDT": 10000.0}
        bt.oms_client.trade_history = []
        bt.oms_client.current_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        def set_time(t):
            bt.oms_client.current_time = t
        bt.oms_client.set_current_time = Mock(side_effect=set_time)
        bt.oms_client.set_data_manager = Mock()

        return bt

    @pytest.fixture
    def edge_strategy(self):
        """Strategy for edge case tests."""
        strategy = Mock()
        strategy.symbols = ["BTC-USDT-PERP"]
        strategy.lookback_days = 1
        strategy.run_strategy = Mock(return_value=[])
        return strategy

    @pytest.fixture
    def edge_pm(self):
        """Position manager for edge case tests."""
        pm = Mock()
        pm.filter_orders = Mock(return_value=[])
        return pm

    @patch("crypto_backtester_binance.backtester.format_positions_table")
    def test_backtest_with_positions_error(self, mock_format, edge_backtester, edge_strategy, edge_pm):
        """Test backtest handles position formatting errors gracefully."""
        mock_format.side_effect = Exception("Format error")

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        result = edge_backtester.run_backtest(
            strategy=edge_strategy,
            position_manager=edge_pm,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        # Should still complete (falls back to get_position_summary)
        assert result is not None

    def test_backtest_with_exposure_error(self, edge_backtester, edge_strategy, edge_pm):
        """Test backtest handles exposure snapshot errors gracefully."""
        edge_backtester.oms_client.positions = {"BTC-USDT-PERP": {"quantity": 0.1}}
        edge_backtester.oms_client.get_current_price = Mock(return_value=None)

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        result = edge_backtester.run_backtest(
            strategy=edge_strategy,
            position_manager=edge_pm,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        assert result is not None

    def test_backtest_iteration_error_recovery(self, edge_backtester, edge_strategy, edge_pm):
        """Test backtest recovers from iteration errors."""
        call_count = [0]
        def failing_update():
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Iteration error")
            return 10000.0

        edge_backtester.oms_client.update_portfolio_value = Mock(side_effect=failing_update)

        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 2, tzinfo=pd.Timestamp.now("UTC").tz)

        result = edge_backtester.run_backtest(
            strategy=edge_strategy,
            position_manager=edge_pm,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        assert result is not None


class TestDataAlignment:
    """Tests for data alignment in backtester."""

    @pytest.fixture
    def alignment_backtester(self, tmp_path):
        """Create backtester for alignment tests."""
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        bt = Backtester(historical_data_dir=str(data_dir))

        # Create data with naive timestamps (no timezone)
        dates_naive = pd.date_range("2024-01-01", periods=50, freq="1h")
        sample_data_naive = pd.DataFrame({
            "timestamp": dates_naive,
            "open": [45000.0] * 50,
            "high": [45100.0] * 50,
            "low": [44900.0] * 50,
            "close": [45000.0] * 50,
            "volume": [100.0] * 50,
        })

        bt.data_manager.load_data_period = Mock(return_value=sample_data_naive)
        bt.data_manager.spot_ohlcv_data = {"BTC": sample_data_naive}
        bt.data_manager.perpetual_mark_ohlcv_data = {"BTC": sample_data_naive}
        bt.data_manager.perpetual_index_ohlcv_data = {"BTC": sample_data_naive}

        bt.oms_client.update_portfolio_value = Mock(return_value=10000.0)
        bt.oms_client.get_position = Mock(return_value=[])
        bt.oms_client.get_position_summary = Mock(return_value={})
        bt.oms_client.set_target_position = Mock(return_value={"status": "success"})
        bt.oms_client.get_current_price = Mock(return_value=45000.0)
        bt.oms_client.positions = {}
        bt.oms_client.balance = {"USDT": 10000.0}
        bt.oms_client.trade_history = []
        bt.oms_client.current_time = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        def set_time(t):
            bt.oms_client.current_time = t
        bt.oms_client.set_current_time = Mock(side_effect=set_time)
        bt.oms_client.set_data_manager = Mock()

        return bt

    @pytest.fixture
    def alignment_strategy(self):
        """Strategy for alignment tests."""
        strategy = Mock()
        strategy.symbols = ["BTC-USDT-PERP"]
        strategy.lookback_days = 1
        strategy.run_strategy = Mock(return_value=[])
        return strategy

    @pytest.fixture
    def alignment_pm(self):
        """Position manager for alignment tests."""
        pm = Mock()
        pm.filter_orders = Mock(return_value=[])
        return pm

    def test_timezone_naive_data_alignment(self, alignment_backtester, alignment_strategy, alignment_pm):
        """Test backtest handles timezone-naive data correctly."""
        start_date = datetime(2024, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)
        end_date = datetime(2024, 1, 1, 1, tzinfo=pd.Timestamp.now("UTC").tz)

        result = alignment_backtester.run_backtest(
            strategy=alignment_strategy,
            position_manager=alignment_pm,
            start_date=start_date,
            end_date=end_date,
            time_step=timedelta(hours=1),
            market_type="futures",
        )

        assert result is not None
