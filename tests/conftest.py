"""
Pytest configuration and shared fixtures for test suite.

This module provides:
- Automatic logging cleanup between tests (reset_logging)
- Mock exchanges to prevent network calls in CI (mock_exchanges)
- Custom pytest markers for test categorization
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_exchanges():
    """
    Mock Binance exchange classes to prevent network calls in CI.

    This is necessary because GitHub Actions runs in locations where
    Binance API access is blocked (e.g., US servers).
    """
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv = MagicMock(return_value=[])
    mock_exchange.fetch_trades = MagicMock(return_value=[])
    mock_exchange.fetch_funding_rate_history = MagicMock(return_value=[])
    mock_exchange.fetch_open_interest_history = MagicMock(return_value=[])

    with (
        patch("crypto_backtester_binance.hist_data.binance", return_value=mock_exchange),
        patch("crypto_backtester_binance.hist_data.binance_pro", return_value=mock_exchange),
    ):
        yield


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration after each test.

    This prevents log handlers from accumulating across tests, which can cause
    duplicate log messages and memory leaks in long test runs.
    """
    import logging

    yield
    # Clean up any handlers that might have been added
    logger = logging.getLogger("crypto_backtester_binance.hist_data")
    logger.handlers.clear()


def pytest_configure(config):
    """
    Configure pytest with custom markers for test categorization.

    Markers allow filtering tests by type:
    - @pytest.mark.unit: Unit tests (fast, isolated)
    - @pytest.mark.integration: Integration tests (may be slower)
    - @pytest.mark.slow: Tests that take significant time

    Usage: pytest -m "unit" to run only unit tests
    """
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
