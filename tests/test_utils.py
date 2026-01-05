"""
Test suite for utility functions.

This module contains tests for utility functions used throughout the project,
including symbol conversion, timezone validation, and timeframe calculations.
"""

import pytest
from datetime import datetime, timezone, timedelta

from crypto_backtester_binance.utils import (
    _convert_symbol_to_ccxt,
    _normalize_symbol_pair,
    _is_utc,
    _get_number_of_periods,
    _get_timeframe_to_minutes
)


class TestConvertSymbolToCCXT:
    """Tests for _convert_symbol_to_ccxt function."""

    def test_convert_spot_symbol_with_hyphen(self):
        """Test conversion of spot symbol with hyphen."""
        result = _convert_symbol_to_ccxt("BTC-USDT", "spot")
        assert result == "BTC/USDT"

    def test_convert_spot_symbol_already_formatted(self):
        """Test conversion of spot symbol already in CCXT format."""
        result = _convert_symbol_to_ccxt("BTC/USDT", "spot")
        assert result == "BTC/USDT"

    def test_convert_future_symbol_simple(self):
        """Test conversion of futures symbol."""
        result = _convert_symbol_to_ccxt("BTC-USDT", "future")
        assert result == "BTC/USDT:USDT"

    def test_convert_future_symbol_with_perp(self):
        """Test conversion of futures symbol with -PERP suffix."""
        result = _convert_symbol_to_ccxt("BTC-USDT-PERP", "future")
        assert result == "BTC/USDT:USDT"

    def test_convert_eth_spot(self):
        """Test conversion of ETH spot symbol."""
        result = _convert_symbol_to_ccxt("ETH-USDT", "spot")
        assert result == "ETH/USDT"

    def test_convert_eth_future(self):
        """Test conversion of ETH futures symbol."""
        result = _convert_symbol_to_ccxt("ETH-USDT", "future")
        assert result == "ETH/USDT:USDT"

    def test_convert_symbol_handles_exception(self):
        """Test that function handles exceptions gracefully."""
        result = _convert_symbol_to_ccxt(None, "spot")
        assert result is None


class TestNormalizeSymbolPair:
    """Tests for _normalize_symbol_pair function."""

    def test_normalize_perp_symbol(self):
        """Test normalization of perpetual symbol."""
        result = _normalize_symbol_pair("BTC-USDT-PERP")
        assert result == "BTC/USDT"

    def test_normalize_spot_symbol(self):
        """Test normalization of spot symbol."""
        result = _normalize_symbol_pair("BTC-USDT")
        assert result == "BTC/USDT"

    def test_normalize_already_formatted(self):
        """Test normalization of already formatted symbol."""
        result = _normalize_symbol_pair("BTC/USDT")
        assert result == "BTC/USDT"

    def test_normalize_lowercase(self):
        """Test normalization converts to uppercase."""
        result = _normalize_symbol_pair("btc-usdt")
        assert result == "BTC/USDT"

    def test_normalize_mixed_case(self):
        """Test normalization with mixed case."""
        result = _normalize_symbol_pair("Btc-UsDt")
        assert result == "BTC/USDT"

    def test_normalize_single_symbol(self):
        """Test normalization with single symbol (assumes USDT)."""
        result = _normalize_symbol_pair("BTC")
        assert result == "BTC/USDT"

    def test_normalize_empty_string(self):
        """Test normalization with empty string."""
        result = _normalize_symbol_pair("")
        assert result is None

    def test_normalize_none(self):
        """Test normalization with None."""
        result = _normalize_symbol_pair(None)
        assert result is None

    def test_normalize_with_spaces(self):
        """Test normalization handles whitespace."""
        result = _normalize_symbol_pair("  BTC-USDT  ")
        assert result == "BTC/USDT"

    def test_normalize_eth_perp(self):
        """Test normalization of ETH perpetual."""
        result = _normalize_symbol_pair("ETH-USDT-PERP")
        assert result == "ETH/USDT"


class TestIsUTC:
    """Tests for _is_utc function."""

    def test_valid_utc_timezone(self):
        """Test that UTC timezone is accepted."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Should not raise an exception
        _is_utc(dt)

    def test_naive_datetime_raises_error(self):
        """Test that naive datetime (no timezone) raises error."""
        dt = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="not in UTC timezone"):
            _is_utc(dt)

    def test_non_utc_timezone_raises_error(self):
        """Test that non-UTC timezone raises error."""
        from datetime import timedelta as td
        # Create a timezone offset of +5 hours
        dt = datetime(2024, 1, 1, tzinfo=timezone(td(hours=5)))
        with pytest.raises(ValueError, match="not in UTC timezone"):
            _is_utc(dt)

    def test_utc_now(self):
        """Test with datetime.now(timezone.utc)."""
        dt = datetime.now(timezone.utc)
        # Should not raise an exception
        _is_utc(dt)

    def test_utc_timestamp(self):
        """Test with datetime.fromtimestamp with UTC."""
        dt = datetime.fromtimestamp(1704067200, tz=timezone.utc)
        # Should not raise an exception
        _is_utc(dt)


class TestGetNumberOfPeriods:
    """Tests for _get_number_of_periods function."""

    def test_one_day_hourly_periods(self):
        """Test calculating hourly periods over one day."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = _get_number_of_periods('1h', start, end)
        assert result == 24

    def test_one_day_15min_periods(self):
        """Test calculating 15-minute periods over one day."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = _get_number_of_periods('15m', start, end)
        assert result == 96  # 24 hours * 4 periods per hour

    def test_one_week_daily_periods(self):
        """Test calculating daily periods over one week."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)

        result = _get_number_of_periods('1d', start, end)
        assert result == 7

    def test_one_hour_5min_periods(self):
        """Test calculating 5-minute periods over one hour."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)

        result = _get_number_of_periods('5m', start, end)
        assert result == 12

    def test_partial_period(self):
        """Test calculating periods with partial period at end."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 30, tzinfo=timezone.utc)

        result = _get_number_of_periods('1h', start, end)
        assert result == 1  # Should truncate partial period

    def test_zero_duration(self):
        """Test with start and end at same time."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)

        result = _get_number_of_periods('1h', start, end)
        assert result == 0

    def test_30min_timeframe(self):
        """Test with 30-minute timeframe."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        result = _get_number_of_periods('30m', start, end)
        assert result == 48

    def test_1min_timeframe(self):
        """Test with 1-minute timeframe."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)

        result = _get_number_of_periods('1m', start, end)
        assert result == 60


class TestGetTimeframeToMinutes:
    """Tests for _get_timeframe_to_minutes function."""

    def test_1min_timeframe(self):
        """Test 1-minute timeframe."""
        result = _get_timeframe_to_minutes('1m')
        assert result == 1

    def test_5min_timeframe(self):
        """Test 5-minute timeframe."""
        result = _get_timeframe_to_minutes('5m')
        assert result == 5

    def test_15min_timeframe(self):
        """Test 15-minute timeframe."""
        result = _get_timeframe_to_minutes('15m')
        assert result == 15

    def test_30min_timeframe(self):
        """Test 30-minute timeframe."""
        result = _get_timeframe_to_minutes('30m')
        assert result == 30

    def test_1hour_timeframe(self):
        """Test 1-hour timeframe."""
        result = _get_timeframe_to_minutes('1h')
        assert result == 60

    def test_2hour_timeframe(self):
        """Test 2-hour timeframe."""
        result = _get_timeframe_to_minutes('2h')
        assert result == 120

    def test_4hour_timeframe(self):
        """Test 4-hour timeframe."""
        result = _get_timeframe_to_minutes('4h')
        assert result == 240

    def test_6hour_timeframe(self):
        """Test 6-hour timeframe."""
        result = _get_timeframe_to_minutes('6h')
        assert result == 360

    def test_12hour_timeframe(self):
        """Test 12-hour timeframe."""
        result = _get_timeframe_to_minutes('12h')
        assert result == 720

    def test_1day_timeframe(self):
        """Test 1-day timeframe."""
        result = _get_timeframe_to_minutes('1d')
        assert result == 1440

    def test_unknown_timeframe_defaults_to_15min(self):
        """Test that unknown timeframe defaults to 15 minutes."""
        result = _get_timeframe_to_minutes('unknown')
        assert result == 15

    def test_empty_string_defaults_to_15min(self):
        """Test that empty string defaults to 15 minutes."""
        result = _get_timeframe_to_minutes('')
        assert result == 15


class TestTimeframeIntegration:
    """Integration tests combining timeframe functions."""

    def test_consistent_period_calculation(self):
        """Test that period calculation is consistent with timeframe conversion."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        timeframe = '1h'
        minutes = _get_timeframe_to_minutes(timeframe)
        periods = _get_number_of_periods(timeframe, start, end)

        # Total minutes / minutes per period should equal number of periods
        total_minutes = (end - start).total_seconds() / 60
        expected_periods = int(total_minutes // minutes)

        assert periods == expected_periods

    def test_all_timeframes_produce_valid_periods(self):
        """Test that all supported timeframes produce valid period counts."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)  # 1 week

        timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']

        for tf in timeframes:
            periods = _get_number_of_periods(tf, start, end)
            assert periods > 0, f"Timeframe {tf} should produce positive periods"
            assert isinstance(periods, int), f"Timeframe {tf} should produce integer periods"

    def test_period_calculation_with_various_durations(self):
        """Test period calculation with different durations."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

        test_cases = [
            (timedelta(hours=1), '1m', 60),
            (timedelta(hours=1), '5m', 12),
            (timedelta(hours=6), '1h', 6),
            (timedelta(days=1), '1h', 24),
            (timedelta(days=7), '1d', 7),
        ]

        for duration, timeframe, expected in test_cases:
            end = start + duration
            result = _get_number_of_periods(timeframe, start, end)
            assert result == expected, f"Failed for {duration} with {timeframe}"
