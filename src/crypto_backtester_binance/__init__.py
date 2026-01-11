"""
Crypto Backtester Binance - A backtesting framework for cryptocurrency trading strategies.

This package provides tools for backtesting trading strategies on Binance historical data,
including data collection, order management simulation, and performance analytics.
"""

from .backtester import Backtester
from .hist_data import HistoricalDataCollector
from .oms_simulation import OMSClient
from .position_manager import PositionManager

__all__ = [
    "Backtester",
    "OMSClient",
    "HistoricalDataCollector",
    "PositionManager",
]

__version__ = "0.1.0"
