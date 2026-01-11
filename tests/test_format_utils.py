"""Tests for format_utils module."""

import pytest

from crypto_backtester_binance.format_utils import (
    format_balances_table,
    format_batch_result,
    format_positions_table,
    format_table,
    format_target_elements,
)


class TestFormatTable:
    """Tests for format_table function."""

    def test_empty_rows(self):
        """Test with empty rows."""
        result = format_table(["Col1", "Col2"], [])
        assert result == "(empty)"

    def test_single_row(self):
        """Test with single row."""
        result = format_table(["Name", "Value"], [["test", "123"]])
        assert "Name" in result
        assert "Value" in result
        assert "test" in result
        assert "123" in result

    def test_multiple_rows(self):
        """Test with multiple rows."""
        result = format_table(["A", "B"], [["1", "2"], ["3", "4"]])
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 rows

    def test_column_width_adjustment(self):
        """Test that columns expand to fit content."""
        result = format_table(["X"], [["very_long_value"]])
        assert "very_long_value" in result

    def test_separator_line(self):
        """Test separator line format."""
        result = format_table(["A", "B"], [["1", "2"]])
        lines = result.split("\n")
        assert "-+-" in lines[1]


class TestFormatPositionsTable:
    """Tests for format_positions_table function."""

    def test_empty_positions(self):
        """Test with empty positions list."""
        result = format_positions_table([])
        assert result == "(empty)"

    def test_none_positions(self):
        """Test with None positions."""
        result = format_positions_table(None)
        assert result == "(empty)"

    def test_position_with_all_fields(self):
        """Test position with all fields."""
        positions = [
            {
                "instrument_name": "BTC-USDT",
                "position_side": "LONG",
                "quantity": "0.5",
                "value": "25000.00",
                "avg_price": "50000.00",
            }
        ]
        result = format_positions_table(positions)
        assert "BTC-USDT" in result
        assert "LONG" in result
        assert "0.500000" in result

    def test_position_with_symbol_instead_of_instrument_name(self):
        """Test position using symbol field."""
        positions = [{"symbol": "ETH-USDT", "position_side": "SHORT"}]
        result = format_positions_table(positions)
        assert "ETH-USDT" in result

    def test_position_with_entry_price_fallback(self):
        """Test entry_price fallback when avg_price missing."""
        positions = [{"symbol": "SOL-USDT", "entry_price": "100.50"}]
        result = format_positions_table(positions)
        assert "100.50" in result

    def test_non_numeric_quantity(self):
        """Test handling of non-numeric quantity."""
        positions = [{"symbol": "TEST", "quantity": "invalid"}]
        result = format_positions_table(positions)
        assert "invalid" in result

    def test_non_numeric_value(self):
        """Test handling of non-numeric value."""
        positions = [{"symbol": "TEST", "value": "N/A"}]
        result = format_positions_table(positions)
        assert "N/A" in result


class TestFormatBalancesTable:
    """Tests for format_balances_table function."""

    def test_empty_balances(self):
        """Test with empty balances."""
        result = format_balances_table({})
        assert result == "(empty)"

    def test_none_balances(self):
        """Test with None balances."""
        result = format_balances_table(None)
        assert result == "(empty)"

    def test_single_balance(self):
        """Test with single balance."""
        result = format_balances_table({"USDT": 10000.0})
        assert "USDT" in result
        assert "10000.00" in result

    def test_multiple_balances_sorted(self):
        """Test multiple balances are sorted alphabetically."""
        balances = {"BTC": 1.5, "ETH": 10.0, "USDT": 5000.0}
        result = format_balances_table(balances)
        lines = result.split("\n")
        # Find data rows (skip header and separator)
        data_lines = lines[2:]
        assert "BTC" in data_lines[0]
        assert "ETH" in data_lines[1]
        assert "USDT" in data_lines[2]

    def test_non_numeric_balance(self):
        """Test handling of non-numeric balance."""
        result = format_balances_table({"TOKEN": "unavailable"})
        assert "unavailable" in result


class TestFormatTargetElements:
    """Tests for format_target_elements function."""

    def test_empty_elements(self):
        """Test with empty elements."""
        result = format_target_elements([])
        assert result == "(empty)"

    def test_none_elements(self):
        """Test with None elements."""
        result = format_target_elements(None)
        assert result == "(empty)"

    def test_single_element(self):
        """Test with single element."""
        elements = [
            {
                "instrument_name": "BTC-USDT-PERP",
                "instrument_type": "future",
                "position_side": "LONG",
                "target_value": 1000.0,
            }
        ]
        result = format_target_elements(elements)
        assert "BTC-USDT-PERP" in result
        assert "future" in result
        assert "LONG" in result
        assert "1000.0" in result

    def test_multiple_elements(self):
        """Test with multiple elements."""
        elements = [
            {"instrument_name": "BTC-USDT", "position_side": "LONG"},
            {"instrument_name": "ETH-USDT", "position_side": "SHORT"},
        ]
        result = format_target_elements(elements)
        assert "BTC-USDT" in result
        assert "ETH-USDT" in result


class TestFormatBatchResult:
    """Tests for format_batch_result function."""

    def test_none_result(self):
        """Test with None result."""
        result = format_batch_result(None)
        assert result == "(empty)"

    def test_list_result(self):
        """Test with list result."""
        data = [{"id": "123", "instrument_name": "BTC-USDT"}]
        result = format_batch_result(data)
        assert "123" in result
        assert "BTC-USDT" in result

    def test_dict_result(self):
        """Test with single dict result."""
        data = {"id": "456", "position_side": "LONG"}
        result = format_batch_result(data)
        assert "456" in result
        assert "LONG" in result

    def test_non_dict_non_list_result(self):
        """Test with unexpected type."""
        result = format_batch_result("just a string")
        assert result == "just a string"

    def test_complete_batch_item(self):
        """Test with complete batch item."""
        data = [
            {
                "id": "task-1",
                "instrument_name": "SOL-USDT-PERP",
                "instrument_type": "future",
                "position_side": "SHORT",
                "target_value": 500.0,
                "create_time": "2024-01-01T00:00:00",
                "update_time": "2024-01-01T00:01:00",
            }
        ]
        result = format_batch_result(data)
        assert "task-1" in result
        assert "SOL-USDT-PERP" in result
        assert "future" in result
        assert "SHORT" in result
        assert "500.0" in result
