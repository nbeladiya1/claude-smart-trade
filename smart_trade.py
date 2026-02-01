#!/usr/bin/env python3
"""
Claude Smart Trade Bot
A MetaTrader 5 trading bot for automated trading.
"""
import logging
import time
import sys
import signal
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/smart-trade/bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Timeframe mapping
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
    'W1': mt5.TIMEFRAME_W1,
}

class SmartTradeBot:
    """Main trading bot class."""

    def __init__(self):
        self.running = True
        self.connected = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.running = False

    def connect(self) -> bool:
        """Connect to MetaTrader 5."""
        logger.info("Initializing MetaTrader 5...")

        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # Login to account if credentials provided
        if config.MT5_LOGIN and config.MT5_PASSWORD:
            logger.info(f"Logging in to account {config.MT5_LOGIN}...")
            authorized = mt5.login(
                login=config.MT5_LOGIN,
                password=config.MT5_PASSWORD,
                server=config.MT5_SERVER
            )
            if not authorized:
                logger.error(f"Login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to account: {account_info.login}")
            logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            logger.info(f"Server: {account_info.server}")

        self.connected = True
        return True

    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader 5")

    def get_market_data(self, symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
        """Get historical market data."""
        tf = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        if rates is None:
            logger.error(f"Failed to get rates for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data['close'].rolling(window=period).mean()

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, data: pd.DataFrame) -> str:
        """Generate trading signal based on indicators."""
        if len(data) < 50:
            return 'HOLD'

        # Calculate indicators
        data['sma_fast'] = self.calculate_sma(data, 10)
        data['sma_slow'] = self.calculate_sma(data, 50)
        data['rsi'] = self.calculate_rsi(data)

        last = data.iloc[-1]
        prev = data.iloc[-2]

        # Simple crossover strategy with RSI filter
        if (prev['sma_fast'] <= prev['sma_slow'] and
            last['sma_fast'] > last['sma_slow'] and
            last['rsi'] < 70):
            return 'BUY'

        if (prev['sma_fast'] >= prev['sma_slow'] and
            last['sma_fast'] < last['sma_slow'] and
            last['rsi'] > 30):
            return 'SELL'

        return 'HOLD'

    def execute_trade(self, symbol: str, signal: str) -> bool:
        """Execute a trade based on the signal."""
        if signal == 'HOLD':
            return True

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return False

        point = symbol_info.point
        price = mt5.symbol_info_tick(symbol).ask if signal == 'BUY' else mt5.symbol_info_tick(symbol).bid

        sl = price - config.STOP_LOSS * point if signal == 'BUY' else price + config.STOP_LOSS * point
        tp = price + config.TAKE_PROFIT * point if signal == 'BUY' else price - config.TAKE_PROFIT * point

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": config.LOT_SIZE,
            "type": mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Smart Trade Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return False

        logger.info(f"Order executed: {signal} {config.LOT_SIZE} {symbol} at {price}")
        return True

    def check_open_positions(self) -> int:
        """Check number of open positions for the symbol."""
        positions = mt5.positions_get(symbol=config.SYMBOL)
        return len(positions) if positions else 0

    def run(self):
        """Main bot loop."""
        logger.info("=" * 50)
        logger.info("Claude Smart Trade Bot Starting")
        logger.info("=" * 50)

        if not self.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        logger.info(f"Trading {config.SYMBOL} on {config.TIMEFRAME}")
        logger.info(f"Lot size: {config.LOT_SIZE}, SL: {config.STOP_LOSS}, TP: {config.TAKE_PROFIT}")
        logger.info(f"Check interval: {config.CHECK_INTERVAL} seconds")
        logger.info("=" * 50)

        while self.running:
            try:
                # Get market data
                data = self.get_market_data(config.SYMBOL, config.TIMEFRAME)

                if data.empty:
                    logger.warning("No market data received")
                    time.sleep(config.CHECK_INTERVAL)
                    continue

                # Generate signal
                signal = self.generate_signal(data)
                current_price = data.iloc[-1]['close']

                logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                           f"{config.SYMBOL}: {current_price:.5f} | Signal: {signal}")

                # Check open positions
                open_positions = self.check_open_positions()

                # Execute trade if signal and no open position
                if signal != 'HOLD' and open_positions == 0:
                    self.execute_trade(config.SYMBOL, signal)
                elif open_positions > 0:
                    logger.info(f"Already have {open_positions} open position(s)")

                time.sleep(config.CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(config.CHECK_INTERVAL)

        self.disconnect()
        logger.info("Bot stopped")


def main():
    """Entry point."""
    bot = SmartTradeBot()
    bot.run()


if __name__ == "__main__":
    main()
