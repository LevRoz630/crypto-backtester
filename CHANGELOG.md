# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-11

### Added
- Initial release of crypto-backtester-binance
- **Historical Data Collection**:
  - Automated collection of OHLCV data for Binance Spot and Futures markets
  - Support for trades, funding rates, and open interest data
  - Efficient caching with Parquet format
  - Automatic data alignment and timezone handling
- **Backtesting Engine**:
  - Strategy execution over historical data with configurable time steps
  - Support for both spot and perpetual futures markets
  - Parameter permutation testing for strategy optimization
  - Comprehensive trade and order logging
- **Order Management System (OMS)**:
  - Simulated order execution with realistic fills
  - Position tracking with PnL calculations
  - Margin requirement validation for futures
  - Stop-loss and take-profit order support
  - Balance and position management
- **Position Management**:
  - Volatility-based risk screening
  - Inverse-volatility position weighting
  - Kelly Criterion position sizing (future enhancement)
  - Budget constraints and balance validation
- **Performance Analytics**:
  - Returns, Sharpe ratio, and drawdown calculations
  - Portfolio value tracking over time
  - Position exposure history
  - Interactive plotting with Plotly
- **Data Export**:
  - Export results to CSV and Parquet formats
  - JSON serialization for backtest results
  - Trade history and order log exports

### Infrastructure
- Modern Python packaging with pyproject.toml
- MIT License
- Python 3.11+ support
- Type hints throughout codebase with PEP 561 compliance (`py.typed` marker)
- Code quality tools:
  - Ruff for linting and formatting
  - Mypy for static type checking
  - Pre-commit hooks for automated checks
- Comprehensive test suite:
  - 44 tests with pytest
  - 88% coverage for PositionManager
  - 40% coverage for Backtester
  - Test fixtures and mocking for isolated testing
- CI/CD with GitHub Actions:
  - Automated testing on Python 3.11, 3.12, 3.13
  - Code quality checks (linting, formatting, type checking)
  - Automated PyPI publishing on releases
- Documentation:
  - Comprehensive README with examples
  - API documentation with MkDocs
  - Method-level docstrings in Google format

### Dependencies
- aiohttp >= 3.12.15
- ccxt >= 4.5.4
- fastparquet >= 2024.11.0
- numpy >= 2.3.3
- pandas >= 2.3.2
- plotly >= 6.3.1
- pyarrow >= 21.0.0
- python-dotenv >= 1.1.1
- requests >= 2.32.5

[0.1.0]: https://github.com/LevRoz630/crypto-backtester-binance/releases/tag/v0.1.0



