"""
Pytest configuration and shared fixtures for test suite.

This module provides:
- Automatic logging cleanup between tests (reset_logging)
- Custom pytest markers for test categorization
"""

import pytest


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
