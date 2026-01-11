"""
Crypto Backtester Binance - A backtesting framework for cryptocurrency trading strategies.

This package provides tools for backtesting trading strategies on Binance historical data,
including data collection, order management simulation, and performance analytics.
"""

from .backtester import Backtester
from .hist_data import HistoricalDataCollector
from .oms_simulation import OMSClient

__all__ = [
    "Backtester",
    "OMSClient",
    "HistoricalDataCollector",
]

__version__ = "1.0.0"
