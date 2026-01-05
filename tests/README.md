# Test Suite for Crypto Backtester Binance

This directory contains the comprehensive test suite for the crypto-backtester-binance project.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared pytest fixtures and configuration
├── test_hist_data.py        # Tests for HistoricalDataCollector class
├── test_utils.py            # Tests for utility functions
└── README.md                # This file
```

## Test Organization

### test_hist_data.py
Comprehensive tests for the `HistoricalDataCollector` class, including:

- **Initialization Tests**: Test data directory creation, exchange initialization, and data store setup
- **Cache Utility Tests**: Test cache file pattern matching and glob functionality
- **Load Cached Window Tests**: Test loading data from cache files with proper filtering
- **Load From Class Tests**: Test loading data from in-memory class storage
- **Spot OHLCV Tests**: Test collection of spot market OHLCV data
- **Perpetual Mark OHLCV Tests**: Test collection of perpetual futures mark price data
- **Perpetual Index OHLCV Tests**: Test collection of perpetual futures index price data
- **Funding Rates Tests**: Test collection of funding rates data
- **Open Interest Tests**: Test collection of open interest data
- **Perpetual Trades Tests**: Test collection of trades data
- **Load Data Period Tests**: Test the main wrapper method for data loading
- **Loop Data Collection Tests**: Test the internal data collection loop
- **Error Handling Tests**: Test error handling and edge cases
- **Integration Tests**: Test complete workflows combining multiple components

### test_utils.py
Tests for utility functions used throughout the project:

- **Symbol Conversion Tests**: Test conversion of symbols to CCXT format
- **Symbol Normalization Tests**: Test normalization of various symbol formats
- **UTC Validation Tests**: Test timezone validation
- **Period Calculation Tests**: Test calculation of periods between dates
- **Timeframe Conversion Tests**: Test conversion of timeframe strings to minutes
- **Integration Tests**: Test combinations of utility functions

### conftest.py
Shared pytest fixtures and configuration:

- `temp_data_dir`: Temporary directory for test data
- `collector`: Pre-configured HistoricalDataCollector instance
- `sample_ohlcv_data`: Sample OHLCV DataFrame
- `sample_funding_rate_data`: Sample funding rate DataFrame
- `sample_open_interest_data`: Sample open interest DataFrame
- `sample_trades_data`: Sample trades DataFrame
- `utc_datetime_pair`: Pair of UTC datetime objects
- `mock_ccxt_*_response`: Mock CCXT API responses

## Running Tests

### Install Test Dependencies

First, install the test dependencies:

```bash
pip install -e ".[test]"
```

Or if using uv:

```bash
uv pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
pytest --cov=crypto_backtester_binance --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run Specific Test File

```bash
pytest tests/test_hist_data.py
```

### Run Specific Test Class

```bash
pytest tests/test_hist_data.py::TestHistoricalDataCollectorInit
```

### Run Specific Test Method

```bash
pytest tests/test_hist_data.py::TestHistoricalDataCollectorInit::test_init_creates_data_directory
```

### Run Tests with Markers

Run only unit tests:
```bash
pytest -m unit
```

Run only integration tests:
```bash
pytest -m integration
```

Skip slow tests:
```bash
pytest -m "not slow"
```

### Run Tests in Verbose Mode

```bash
pytest -v
```

### Run Tests in Quiet Mode

```bash
pytest -q
```

### Run Tests and Stop at First Failure

```bash
pytest -x
```

### Run Tests with Output Capture Disabled

```bash
pytest -s
```

This is useful when debugging with print statements.

## Test Coverage

The test suite aims for high coverage of the codebase. Key coverage areas:

- **HistoricalDataCollector**: ~95% coverage
  - Initialization and setup
  - Data collection methods
  - Caching mechanisms
  - Data loading and filtering
  - Error handling

- **Utility Functions**: 100% coverage
  - Symbol conversion
  - Timezone validation
  - Period calculations

## Mocking Strategy

The tests use Python's `unittest.mock` and `pytest-mock` to mock external dependencies:

- **CCXT API calls**: Mocked to avoid hitting real APIs during tests
- **File system operations**: Tested using temporary directories
- **Network calls**: Mocked to ensure fast and reliable tests

## Writing New Tests

When adding new functionality, follow these guidelines:

1. **Create tests first** (TDD approach recommended)
2. **Use descriptive test names** that explain what is being tested
3. **Test one thing per test** to make failures easy to diagnose
4. **Use fixtures** from conftest.py for common test data
5. **Add docstrings** to test classes and methods
6. **Mock external dependencies** to keep tests fast and isolated
7. **Test edge cases** and error conditions
8. **Add markers** for test categorization (unit, integration, slow)

### Example Test

```python
class TestNewFeature:
    """Tests for new feature."""

    def test_feature_success_case(self, collector, sample_data):
        """Test that feature works correctly with valid input."""
        result = collector.new_feature(sample_data)

        assert result is not None
        assert len(result) > 0
        assert 'expected_column' in result.columns

    def test_feature_handles_empty_input(self, collector):
        """Test that feature handles empty input gracefully."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            collector.new_feature(None)
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They:

- Use temporary directories (cleaned up automatically)
- Mock all external API calls
- Are deterministic (no random behavior)
- Run quickly (most tests complete in milliseconds)

## Troubleshooting

### Tests Fail Due to Timezone Issues

Ensure all datetime objects use UTC timezone:
```python
from datetime import datetime, timezone
dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
```

### Tests Fail Due to Missing Dependencies

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Coverage Report Not Generated

Ensure pytest-cov is installed:
```bash
pip install pytest-cov
```

### Tests Are Slow

Check if you're accidentally hitting real APIs. All external calls should be mocked.

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
