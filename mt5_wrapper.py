#!/usr/bin/env python3
"""
MT5 Wrapper for Linux VPS
Provides a simulation/demo mode when MetaTrader5 is not available.
"""
import os
import sys
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import MetaTrader5, fall back to mock if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.info("MetaTrader5 module loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available - using simulation mode")

# Simulation mode flag
SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'true').lower() == 'true' or not MT5_AVAILABLE


@dataclass
class AccountInfo:
    """Mock account info structure."""
    login: int = 12345678
    balance: float = 10000.0
    equity: float = 10000.0
    profit: float = 0.0
    margin: float = 0.0
    margin_free: float = 10000.0
    currency: str = "USD"
    server: str = "Demo-Server"
    leverage: int = 100


@dataclass
class SymbolInfo:
    """Mock symbol info structure."""
    name: str = "EURUSD"
    bid: float = 1.0850
    ask: float = 1.0852
    point: float = 0.00001
    digits: int = 5
    spread: int = 20
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01


@dataclass
class OrderResult:
    """Mock order result structure."""
    retcode: int = 10009  # Success
    order: int = 0
    deal: int = 0
    volume: float = 0.01
    price: float = 0.0
    comment: str = ""


@dataclass
class Position:
    """Mock position structure."""
    ticket: int = 0
    symbol: str = "EURUSD"
    type: int = 0  # 0=BUY, 1=SELL
    volume: float = 0.01
    price_open: float = 1.0850
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    time: datetime = None


class MT5Simulator:
    """Simulates MT5 functionality for testing and demo purposes."""
    
    def __init__(self):
        self.initialized = False
        self.logged_in = False
        self.positions = []
        self.orders = []
        self.order_counter = 100000
        self.account = AccountInfo()
        self.prices = {
            'EURUSD': {'bid': 1.0850, 'ask': 1.0852},
            'GBPUSD': {'bid': 1.2650, 'ask': 1.2652},
            'USDJPY': {'bid': 149.50, 'ask': 149.52},
            'XAUUSD': {'bid': 2050.00, 'ask': 2050.50},
        }
        
    def initialize(self) -> bool:
        self.initialized = True
        logger.info("[SIMULATION] MT5 initialized in simulation mode")
        return True
    
    def login(self, login: int, password: str, server: str) -> bool:
        self.logged_in = True
        self.account.login = login
        self.account.server = server
        logger.info(f"[SIMULATION] Logged in as {login} on {server}")
        return True
    
    def shutdown(self):
        self.initialized = False
        self.logged_in = False
        logger.info("[SIMULATION] MT5 shutdown")
    
    def account_info(self) -> AccountInfo:
        return self.account
    
    def symbol_info(self, symbol: str) -> SymbolInfo:
        prices = self.prices.get(symbol, {'bid': 1.0, 'ask': 1.0001})
        return SymbolInfo(
            name=symbol,
            bid=prices['bid'],
            ask=prices['ask']
        )
    
    def symbol_info_tick(self, symbol: str):
        return self.symbol_info(symbol)
    
    def positions_get(self, symbol: str = None):
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return self.positions
    
    def positions_total(self) -> int:
        return len(self.positions)
    
    def copy_rates_from_pos(self, symbol: str, timeframe: int, pos: int, count: int):
        """Generate simulated OHLC data."""
        import numpy as np
        
        base_price = self.prices.get(symbol, {'bid': 1.0})['bid']
        now = datetime.now()
        
        rates = []
        for i in range(count):
            time_offset = (count - i - 1) * 3600  # Hourly bars
            bar_time = now - timedelta(seconds=time_offset)
            
            # Generate realistic OHLC data
            open_price = base_price + random.uniform(-0.01, 0.01)
            high_price = open_price + random.uniform(0, 0.005)
            low_price = open_price - random.uniform(0, 0.005)
            close_price = random.uniform(low_price, high_price)
            
            rates.append((
                int(bar_time.timestamp()),
                open_price,
                high_price,
                low_price,
                close_price,
                random.randint(100, 1000),  # tick_volume
                0,  # spread
                random.randint(1000, 10000)  # real_volume
            ))
            
            base_price = close_price
        
        # Convert to numpy structured array
        dtype = [
            ('time', 'i8'), ('open', 'f8'), ('high', 'f8'),
            ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
            ('spread', 'i4'), ('real_volume', 'i8')
        ]
        return np.array(rates, dtype=dtype)
    
    def order_send(self, request: dict) -> OrderResult:
        """Simulate order execution."""
        self.order_counter += 1
        
        result = OrderResult()
        result.order = self.order_counter
        result.deal = self.order_counter
        result.volume = request.get('volume', 0.01)
        result.price = request.get('price', 0.0)
        
        # Create position for market orders
        action = request.get('action', 0)
        if action == 1:  # TRADE_ACTION_DEAL
            pos = Position(
                ticket=self.order_counter,
                symbol=request.get('symbol', 'EURUSD'),
                type=request.get('type', 0),
                volume=result.volume,
                price_open=result.price,
                sl=request.get('sl', 0.0),
                tp=request.get('tp', 0.0),
                time=datetime.now()
            )
            self.positions.append(pos)
            logger.info(f"[SIMULATION] Order executed: {pos}")
        
        return result
    
    def last_error(self):
        return (0, "No error")


# Global simulator instance
_simulator = MT5Simulator()


# Timeframe constants
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 60
TIMEFRAME_H4 = 240
TIMEFRAME_D1 = 1440
TIMEFRAME_W1 = 10080

# Order types
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1

# Trade actions
TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8
TRADE_ACTION_CLOSE_BY = 10

# Order fill types
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# Trade return codes
TRADE_RETCODE_DONE = 10009


def initialize(*args, **kwargs) -> bool:
    """Initialize MT5 connection."""
    if SIMULATION_MODE:
        return _simulator.initialize()
    return mt5.initialize(*args, **kwargs)


def login(login: int, password: str = "", server: str = "") -> bool:
    """Login to MT5 account."""
    if SIMULATION_MODE:
        return _simulator.login(login, password, server)
    return mt5.login(login, password=password, server=server)


def shutdown():
    """Shutdown MT5 connection."""
    if SIMULATION_MODE:
        return _simulator.shutdown()
    return mt5.shutdown()


def account_info():
    """Get account information."""
    if SIMULATION_MODE:
        return _simulator.account_info()
    return mt5.account_info()


def symbol_info(symbol: str):
    """Get symbol information."""
    if SIMULATION_MODE:
        return _simulator.symbol_info(symbol)
    return mt5.symbol_info(symbol)


def symbol_info_tick(symbol: str):
    """Get symbol tick info."""
    if SIMULATION_MODE:
        return _simulator.symbol_info_tick(symbol)
    return mt5.symbol_info_tick(symbol)


def positions_get(symbol: str = None):
    """Get open positions."""
    if SIMULATION_MODE:
        return _simulator.positions_get(symbol)
    if symbol:
        return mt5.positions_get(symbol=symbol)
    return mt5.positions_get()


def positions_total() -> int:
    """Get total number of open positions."""
    if SIMULATION_MODE:
        return _simulator.positions_total()
    return mt5.positions_total()


def copy_rates_from_pos(symbol: str, timeframe: int, pos: int, count: int):
    """Copy historical rates."""
    if SIMULATION_MODE:
        return _simulator.copy_rates_from_pos(symbol, timeframe, pos, count)
    return mt5.copy_rates_from_pos(symbol, timeframe, pos, count)


def order_send(request: dict):
    """Send trading order."""
    if SIMULATION_MODE:
        return _simulator.order_send(request)
    return mt5.order_send(request)


def last_error():
    """Get last error."""
    if SIMULATION_MODE:
        return _simulator.last_error()
    return mt5.last_error()


def is_simulation_mode() -> bool:
    """Check if running in simulation mode."""
    return SIMULATION_MODE
