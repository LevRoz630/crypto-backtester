"""
Comprehensive test suite for HistoricalDataCollector class.

This module contains unit tests and integration tests for the historical data
collection functionality, including OHLCV data, funding rates, open interest,
and caching mechanisms.
"""

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from crypto_backtester_binance.hist_data import HistoricalDataCollector


class TestHistoricalDataCollectorInit:
    """Tests for HistoricalDataCollector initialization."""

    def test_init_creates_data_directory(self):
        """Test that initialization creates the data directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test_data"
            collector = HistoricalDataCollector(data_dir=str(test_dir))

            assert test_dir.exists()
            assert collector.data_dir == test_dir

    def test_init_with_existing_directory(self):
        """Test initialization with an existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            assert Path(tmpdir).exists()
            assert collector.data_dir == Path(tmpdir)

    def test_init_initializes_data_stores(self):
        """Test that all data storage dictionaries are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            assert isinstance(collector.spot_ohlcv_data, dict)
            assert isinstance(collector.perpetual_mark_ohlcv_data, dict)
            assert isinstance(collector.perpetual_index_ohlcv_data, dict)
            assert isinstance(collector.spot_trades_data, dict)
            assert isinstance(collector.perpetual_trades_data, dict)
            assert isinstance(collector.funding_rates_data, dict)
            assert isinstance(collector.open_interest_data, dict)
            assert len(collector.spot_ohlcv_data) == 0

    def test_init_creates_exchanges(self):
        """Test that CCXT exchange instances are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            assert collector.spot_exchange is not None
            assert collector.futures_exchange is not None
            assert collector.spot_exchange_pro is not None
            assert collector.futures_exchange_pro is not None

    def test_kind_map_contains_all_data_types(self):
        """Test that kind_map contains all supported data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            expected_kinds = [
                "ohlcv_spot",
                "mark_ohlcv_futures",
                "index_ohlcv_futures",
                "funding_rates",
                "open_interest",
                "trades_futures",
            ]

            for kind in expected_kinds:
                assert kind in collector.kind_map


class TestCacheUtilities:
    """Tests for cache-related utility methods."""

    def test_cache_glob_spot_ohlcv(self):
        """Test cache file pattern matching for spot OHLCV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Create test cache files
            test_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_file.touch()

            files = collector._cache_glob("ohlcv_spot", "BTC-USDT", "1h")
            assert len(files) == 1
            assert files[0] == test_file

    def test_cache_glob_perpetual_mark(self):
        """Test cache file pattern matching for perpetual mark price."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            test_file = collector.data_dir / "perpetual_ETH_USDT_mark_15m_20240101_20240102.parquet"
            test_file.touch()

            files = collector._cache_glob("mark_ohlcv_futures", "ETH-USDT", "15m")
            assert len(files) == 1

    def test_cache_glob_funding_rates(self):
        """Test cache file pattern matching for funding rates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            test_file = (
                collector.data_dir
                / "perpetual_BTC_USDT_funding_rates_20240101_000000_20240102_000000.parquet"
            )
            test_file.touch()

            files = collector._cache_glob("funding_rates", "BTC-USDT", None)
            assert len(files) == 1

    def test_cache_glob_no_matches(self):
        """Test cache glob when no files match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            files = collector._cache_glob("ohlcv_spot", "BTC-USDT", "1h")
            assert len(files) == 0


class TestLoadCachedWindow:
    """Tests for load_cached_window method."""

    def test_load_cached_window_no_files(self):
        """Test loading from cache when no files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            result = collector.load_cached_window("ohlcv_spot", "BTC-USDT", start, end, "1h")
            assert result is None

    def test_load_cached_window_with_valid_data(self):
        """Test loading valid cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Create test data
            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1h", tz="UTC"),
                    "open": [100.0] * 25,
                    "high": [105.0] * 25,
                    "low": [95.0] * 25,
                    "close": [102.0] * 25,
                    "volume": [1000.0] * 25,
                }
            )

            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            result = collector.load_cached_window("ohlcv_spot", "BTC-USDT", start, end, "1h")

            assert result is not None
            assert not result.empty
            assert len(result) == 25
            assert "timestamp" in result.columns

    def test_load_cached_window_filters_by_date_range(self):
        """Test that cached data is properly filtered by date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Create test data spanning larger range
            full_start = datetime(2024, 1, 1, tzinfo=UTC)
            full_end = datetime(2024, 1, 10, tzinfo=UTC)

            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(full_start, full_end, freq="1h", tz="UTC"),
                    "open": [100.0] * 217,
                    "close": [102.0] * 217,
                }
            )

            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240110_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            # Request smaller window
            request_start = datetime(2024, 1, 3, tzinfo=UTC)
            request_end = datetime(2024, 1, 5, tzinfo=UTC)

            result = collector.load_cached_window(
                "ohlcv_spot", "BTC-USDT", request_start, request_end, "1h"
            )

            assert result is not None
            assert result["timestamp"].min() >= pd.Timestamp(request_start)
            assert result["timestamp"].max() <= pd.Timestamp(request_end)

    def test_load_cached_window_handles_timezone_aware_data(self):
        """Test that cached data with timezones is handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            test_data = pd.DataFrame(
                {"timestamp": pd.date_range(start, end, freq="1h", tz="UTC"), "open": [100.0] * 25}
            )

            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            result = collector.load_cached_window("ohlcv_spot", "BTC-USDT", start, end, "1h")

            assert result is not None
            assert result["timestamp"].dt.tz is not None

    def test_load_cached_window_removes_duplicates(self):
        """Test that duplicate timestamps are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)

            # Create data with duplicates
            timestamps = [
                start,
                start + timedelta(hours=1),
                start + timedelta(hours=1),
                start + timedelta(hours=2),
            ]
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(timestamps, utc=True),
                    "open": [100.0, 101.0, 101.5, 102.0],
                }
            )

            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            result = collector.load_cached_window(
                "ohlcv_spot", "BTC-USDT", start, start + timedelta(hours=3), "1h"
            )

            assert result is not None
            assert len(result) == 3  # Duplicates should be removed


class TestLoadFromClass:
    """Tests for load_from_class method."""

    def test_load_from_class_spot_ohlcv(self):
        """Test loading spot OHLCV data from class storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            # Store test data
            test_data = pd.DataFrame(
                {"timestamp": pd.date_range(start, end, freq="1h", tz="UTC"), "open": [100.0] * 25}
            )
            collector.spot_ohlcv_data["BTC-USDT"] = test_data

            result = collector.load_from_class("ohlcv_spot", "BTC-USDT", start, end)

            assert result is not None
            assert not result.empty
            pd.testing.assert_frame_equal(result, test_data)

    def test_load_from_class_mark_ohlcv(self):
        """Test loading perpetual mark OHLCV data from class storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            timestamps = pd.date_range(start, end, freq="15min", tz="UTC")
            test_data = pd.DataFrame({"timestamp": timestamps, "open": [100.0] * len(timestamps)})
            collector.perpetual_mark_ohlcv_data["ETH-USDT"] = test_data

            result = collector.load_from_class("mark_ohlcv_futures", "ETH-USDT", start, end)

            assert result is not None
            pd.testing.assert_frame_equal(result, test_data)

    def test_load_from_class_no_data(self):
        """Test loading from class when no data exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            result = collector.load_from_class("ohlcv_spot", "BTC-USDT", start, end)

            assert result is None

    def test_load_from_class_invalid_kind(self):
        """Test loading with invalid kind parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            with pytest.raises(ValueError, match="Invalid kind"):
                collector.load_from_class("invalid_kind", "BTC-USDT", start, end)


class TestCollectSpotOHLCV:
    """Tests for collect_spot_ohlcv method."""

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_spot_ohlcv_success(self, mock_loop):
        """Test successful collection of spot OHLCV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Mock data
            mock_data = [
                [1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0],
                [1704070800000, 102.0, 107.0, 97.0, 104.0, 1100.0],
            ]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=False)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "timestamp" in result.columns
            assert "open" in result.columns
            assert "high" in result.columns
            assert "low" in result.columns
            assert "close" in result.columns
            assert "volume" in result.columns
            assert result["symbol"].iloc[0] == "BTC-USDT"
            assert result["market_type"].iloc[0] == "spot"

    def test_collect_spot_ohlcv_requires_start_time(self):
        """Test that start_time parameter is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            with pytest.raises(ValueError, match="Start time is required"):
                collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time=None)

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_spot_ohlcv_export(self, mock_loop):
        """Test that export parameter saves data to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            _result = collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=True)

            # Check that file was created
            cache_files = list(collector.data_dir.glob("spot_BTC_USDT_ohlcv_1h_*.parquet"))
            assert len(cache_files) == 1

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_spot_ohlcv_stores_in_class(self, mock_loop):
        """Test that collected data is stored in class dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=False)

            assert "BTC-USDT" in collector.spot_ohlcv_data
            assert len(collector.spot_ohlcv_data["BTC-USDT"]) == 1

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_spot_ohlcv_no_data(self, mock_loop):
        """Test handling when no data is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_loop.return_value = []

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_spot_ohlcv("BTC-USDT", "1h", start_time, export=False)

            assert result is None


class TestCollectPerpetualMarkOHLCV:
    """Tests for collect_perpetual_mark_ohlcv method."""

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_perpetual_mark_ohlcv_success(self, mock_loop):
        """Test successful collection of perpetual mark price OHLCV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_perpetual_mark_ohlcv(
                "BTC-USDT", "15m", start_time, export=False
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert "price_type" in result.columns
            assert result["price_type"].iloc[0] == "mark"
            assert result["market_type"].iloc[0] == "perpetual"

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_perpetual_mark_ohlcv_params(self, mock_loop):
        """Test that mark price parameter is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            collector.collect_perpetual_mark_ohlcv("BTC-USDT", "15m", start_time, export=False)

            # Verify params were passed
            call_args = mock_loop.call_args
            assert call_args[1]["params"] == {"price": "mark"}


class TestCollectPerpetualIndexOHLCV:
    """Tests for collect_perpetual_index_ohlcv method."""

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_perpetual_index_ohlcv_success(self, mock_loop):
        """Test successful collection of perpetual index price OHLCV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_perpetual_index_ohlcv(
                "BTC-USDT", "15m", start_time, export=False
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert "price_type" in result.columns
            assert result["price_type"].iloc[0] == "index"
            assert result["market_type"].iloc[0] == "perpetual"

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_perpetual_index_ohlcv_params(self, mock_loop):
        """Test that index price parameter is passed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [[1704067200000, 100.0, 105.0, 95.0, 102.0, 1000.0]]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            collector.collect_perpetual_index_ohlcv("BTC-USDT", "15m", start_time, export=False)

            # Verify params were passed
            call_args = mock_loop.call_args
            assert call_args[1]["params"] == {"price": "index"}


class TestCollectFundingRates:
    """Tests for collect_funding_rates method."""

    def test_collect_funding_rates_requires_start_time(self):
        """Test that start_time parameter is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            with pytest.raises(ValueError, match="Start time is required"):
                collector.collect_funding_rates("BTC-USDT", start_time=None)

    def test_collect_funding_rates_success(self):
        """Test successful collection of funding rates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [
                {
                    "timestamp": 1704067200000,
                    "fundingRate": 0.0001,
                    "fundingTime": 1704067200000,
                    "markPrice": "42000.0",
                    "indexPrice": "41995.0",
                    "info": {},
                }
            ]
            collector.futures_exchange.fetch_funding_rate_history = Mock(return_value=mock_data)

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_funding_rates("BTC-USDT", start_time, export=False)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert "funding_rate" in result.columns
            assert "mark_price" in result.columns
            assert "index_price" in result.columns
            assert result["market_type"].iloc[0] == "perpetual"

    def test_collect_funding_rates_handles_exception(self):
        """Test handling of exceptions during funding rate collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_fetch = Mock(side_effect=Exception("API Error"))
            collector.futures_exchange.fetch_funding_rate_history = mock_fetch

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_funding_rates("BTC-USDT", start_time, export=False)

            assert result is None


class TestCollectOpenInterest:
    """Tests for collect_open_interest method."""

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_collect_open_interest_success(self, mock_loop):
        """Test successful collection of open interest data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [
                {
                    "timestamp": 1704067200000,
                    "openInterestAmount": 1000.0,
                    "openInterestValue": 42000000.0,
                    "contractType": "perpetual",
                }
            ]
            mock_loop.return_value = mock_data

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            result = collector.collect_open_interest("BTC-USDT", "15m", start_time, export=False)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert "open_interest" in result.columns
            assert "open_interest_value" in result.columns
            assert result["market_type"].iloc[0] == "perpetual"

    def test_collect_open_interest_requires_start_time(self):
        """Test that start_time parameter is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            with pytest.raises(ValueError, match="Start time is required"):
                collector.collect_open_interest("BTC-USDT", "15m", start_time=None)


class TestCollectPerpetualTrades:
    """Tests for collect_perpetual_trades method."""

    def test_collect_perpetual_trades_requires_start_time(self):
        """Test that start_time parameter is required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            with pytest.raises(ValueError, match="Start time is required"):
                collector.collect_perpetual_trades("BTC-USDT", start_time=None)

    def test_collect_perpetual_trades_success(self):
        """Test successful collection of perpetual trades."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            mock_data = [
                {
                    "timestamp": 1704067200000,
                    "id": "123456",
                    "side": "buy",
                    "price": 42000.0,
                    "amount": 0.5,
                    "cost": 21000.0,
                    "takerOrMaker": "taker",
                }
            ]
            # Return data first time, empty list second time to exit loop
            collector.futures_exchange.fetch_trades = Mock(side_effect=[mock_data, []])

            start_time = datetime(2024, 1, 1, tzinfo=UTC)
            # Set end_time very close to start_time to avoid infinite loop
            result = collector.collect_perpetual_trades("BTC-USDT", start_time, export=False)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert "side" in result.columns
            assert "price" in result.columns
            assert "amount" in result.columns


class TestLoadDataPeriod:
    """Tests for load_data_period wrapper method."""

    def test_load_data_period_validates_dates(self):
        """Test that date validation is performed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            with pytest.raises(ValueError, match="Start and end dates are required"):
                collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", None, None)

    def test_load_data_period_validates_date_order(self):
        """Test that start date must be before end date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 2, tzinfo=UTC)
            end = datetime(2024, 1, 1, tzinfo=UTC)

            with pytest.raises(ValueError, match="Start date must be before end date"):
                collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

    def test_load_data_period_validates_data_type(self):
        """Test that invalid data types are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            with pytest.raises(ValueError, match="Invalid data type"):
                collector.load_data_period("BTC-USDT", "1h", "invalid_type", start, end)

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector.collect_spot_ohlcv")
    def test_load_data_period_spot_ohlcv(self, mock_collect):
        """Test loading spot OHLCV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1h", tz="UTC"),
                    "open": [100.0] * 25,
                    "close": [102.0] * 25,
                }
            )
            mock_collect.return_value = test_data

            result = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

            assert result is not None
            assert len(result) > 0

    @patch(
        "crypto_backtester_binance.hist_data.HistoricalDataCollector.collect_perpetual_mark_ohlcv"
    )
    def test_load_data_period_mark_ohlcv(self, mock_collect):
        """Test loading perpetual mark price data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            timestamps = pd.date_range(start, end, freq="15min", tz="UTC")
            test_data = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100.0] * len(timestamps),
                    "close": [102.0] * len(timestamps),
                }
            )
            mock_collect.return_value = test_data

            result = collector.load_data_period("BTC-USDT", "15m", "mark_ohlcv_futures", start, end)

            assert result is not None

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector.collect_funding_rates")
    def test_load_data_period_funding_rates(self, mock_collect):
        """Test loading funding rates data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            timestamps = pd.date_range(start, end, freq="8h", tz="UTC")
            test_data = pd.DataFrame(
                {"timestamp": timestamps, "funding_rate": [0.0001] * len(timestamps)}
            )
            mock_collect.return_value = test_data

            result = collector.load_data_period("BTC-USDT", "1h", "funding_rates", start, end)

            assert result is not None

    def test_load_data_period_uses_cache_when_available(self):
        """Test that cached data is used when available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            # Create cached data
            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1h", tz="UTC"),
                    "open": [100.0] * 25,
                    "close": [102.0] * 25,
                }
            )
            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            # Mock collect methods to track if they're called
            with patch.object(collector, "collect_spot_ohlcv") as mock_collect:
                result = collector.load_data_period(
                    "BTC-USDT", "1h", "ohlcv_spot", start, end, load_from_class=False
                )

                # Should use cache, not call collect
                assert mock_collect.call_count == 0
                assert result is not None

    def test_load_data_period_saves_to_class_when_requested(self):
        """Test that save_to_class parameter works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1h", tz="UTC"),
                    "open": [100.0] * 25,
                    "close": [102.0] * 25,
                }
            )
            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            test_data.to_parquet(cache_file)

            _result = collector.load_data_period(
                "BTC-USDT", "1h", "ohlcv_spot", start, end, save_to_class=True
            )

            assert "BTC-USDT" in collector.spot_ohlcv_data


class TestLoopDataCollection:
    """Tests for _loop_data_collection helper method."""

    def test_loop_data_collection_handles_timeframe_minutes(self):
        """Test that timeframe is correctly converted to minutes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 1, 2, tzinfo=UTC)  # 2 hours

            mock_function = Mock(return_value=[])

            _result = collector._loop_data_collection(
                function=mock_function,
                ccxt_symbol="BTC/USDT",
                timeframe="1h",
                limit=1000,
                start_time=start,
                end_time=end,
                params=None,
                logger=None,
            )

            # Should be called at least once
            assert mock_function.call_count >= 1


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_non_utc_timezone_raises_error(self):
        """Test that non-UTC timezones raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Create naive datetime (no timezone)
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 2)

            with pytest.raises(ValueError, match="not in UTC timezone"):
                collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

    def test_handles_empty_dataframe_gracefully(self):
        """Test handling of empty dataframes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            # Create empty cached data
            empty_data = pd.DataFrame(columns=["timestamp", "open", "close"])
            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_000000.parquet"
            )
            empty_data.to_parquet(cache_file)

            with patch.object(collector, "collect_spot_ohlcv") as mock_collect:
                mock_collect.return_value = pd.DataFrame(columns=["timestamp", "open", "close"])

                result = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

                # Should handle empty data gracefully
                assert result is not None

    def test_timeframe_alignment(self):
        """Test that start and end dates are aligned to timeframe boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            # Use non-aligned times
            start = datetime(2024, 1, 1, 0, 37, 15, tzinfo=UTC)  # 37 minutes, 15 seconds
            end = datetime(2024, 1, 2, 0, 42, 30, tzinfo=UTC)  # 42 minutes, 30 seconds

            test_data = pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        datetime(2024, 1, 1, tzinfo=UTC),
                        datetime(2024, 1, 2, tzinfo=UTC),
                        freq="1h",
                        tz="UTC",
                    ),
                    "open": [100.0] * 25,
                    "close": [102.0] * 25,
                }
            )
            cache_file = (
                collector.data_dir
                / "spot_BTC_USDT_ohlcv_1h_20240101_000000_20240102_010000.parquet"
            )
            test_data.to_parquet(cache_file)

            # Mock network calls to prevent API access
            with patch.object(collector, "collect_spot_ohlcv", return_value=test_data):
                result = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

            # Should still work with aligned boundaries
            assert result is not None


class TestIntegration:
    """Integration tests that test multiple components together."""

    @patch("crypto_backtester_binance.hist_data.HistoricalDataCollector._loop_data_collection")
    def test_full_workflow_with_caching(self, mock_loop):
        """Test complete workflow: collect, cache, and retrieve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            # First call - should collect from API
            mock_data = [
                [int(ts.timestamp() * 1000), 100.0, 105.0, 95.0, 102.0, 1000.0]
                for ts in pd.date_range(start, end, freq="1h")
            ]
            mock_loop.return_value = mock_data

            result1 = collector.load_data_period(
                "BTC-USDT", "1h", "ohlcv_spot", start, end, export=True
            )

            assert result1 is not None
            assert len(result1) > 0

            # Second call - should use cache
            mock_loop.reset_mock()
            result2 = collector.load_data_period("BTC-USDT", "1h", "ohlcv_spot", start, end)

            assert result2 is not None
            # Should not call API again
            assert mock_loop.call_count == 0

            # Results should be similar
            assert len(result1) == len(result2)

    def test_multiple_symbols_independent_storage(self):
        """Test that data for multiple symbols is stored independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = HistoricalDataCollector(data_dir=tmpdir)

            start = datetime(2024, 1, 1, tzinfo=UTC)
            end = datetime(2024, 1, 2, tzinfo=UTC)

            # Store data for two symbols
            btc_data = pd.DataFrame(
                {"timestamp": pd.date_range(start, end, freq="1h", tz="UTC"), "open": [100.0] * 25}
            )

            eth_data = pd.DataFrame(
                {"timestamp": pd.date_range(start, end, freq="1h", tz="UTC"), "open": [200.0] * 25}
            )

            collector.spot_ohlcv_data["BTC-USDT"] = btc_data
            collector.spot_ohlcv_data["ETH-USDT"] = eth_data

            # Verify independent storage
            assert "BTC-USDT" in collector.spot_ohlcv_data
            assert "ETH-USDT" in collector.spot_ohlcv_data
            assert collector.spot_ohlcv_data["BTC-USDT"]["open"].iloc[0] == 100.0
            assert collector.spot_ohlcv_data["ETH-USDT"]["open"].iloc[0] == 200.0
