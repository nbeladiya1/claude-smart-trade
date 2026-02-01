#!/usr/bin/env python3
"""
Claude Smart Trade Bot v2.0
A MetaTrader 5 trading bot with Dual-LLM Decision System.

Features:
- Dual-LLM agents (FinGPT + FinLLaMA) for trading decisions
- Advanced money management with risk controls
- Telegram notifications for all updates
- Automated position management with trailing stops
"""
import logging
import time
import sys
import signal
import os
from datetime import datetime, date
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

import config
from llm_agents import DualAgentConsensus, get_financial_news, Signal
from money_management import MoneyManager, RiskMetrics
from telegram_bot import get_notifier, TelegramNotifier

# Ensure log directory exists
os.makedirs('/var/log/smart-trade', exist_ok=True)

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
    """
    Main trading bot class with Dual-LLM decision system.
    Integrates FinGPT (sentiment) and FinLLaMA (technical) agents.
    """

    def __init__(self):
        self.running = True
        self.connected = False

        # Initialize components
        self.telegram = get_notifier()
        self.money_manager = MoneyManager()
        self.dual_agent = DualAgentConsensus() if config.USE_DUAL_LLM else None

        # Track daily stats
        self.trades_today = 0
        self.last_summary_date: Optional[date] = None

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.telegram.notify_bot_stopped(f"Signal {signum} received")
        self.running = False

    def connect(self) -> bool:
        """Connect to MetaTrader 5."""
        logger.info("Initializing MetaTrader 5...")

        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            self.telegram.notify_error(f"MT5 initialization failed: {error}")
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
                error = mt5.last_error()
                logger.error(f"Login failed: {error}")
                self.telegram.notify_error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False

        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Connected to account: {account_info.login}")
            logger.info(f"Balance: {account_info.balance} {account_info.currency}")
            logger.info(f"Server: {account_info.server}")

        # Initialize money manager
        self.money_manager.initialize()

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

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1] if not atr.empty else 0

    def get_fallback_signal(self, data: pd.DataFrame) -> str:
        """Generate fallback signal when LLM is not available."""
        if len(data) < 50:
            return 'HOLD'

        close = data['close']
        sma_fast = close.rolling(window=10).mean()
        sma_slow = close.rolling(window=50).mean()

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        last_sma_fast = sma_fast.iloc[-1]
        last_sma_slow = sma_slow.iloc[-1]
        prev_sma_fast = sma_fast.iloc[-2]
        prev_sma_slow = sma_slow.iloc[-2]
        last_rsi = rsi.iloc[-1]

        # Crossover strategy with RSI filter
        if (prev_sma_fast <= prev_sma_slow and
            last_sma_fast > last_sma_slow and
            last_rsi < 70):
            return 'BUY'

        if (prev_sma_fast >= prev_sma_slow and
            last_sma_fast < last_sma_slow and
            last_rsi > 30):
            return 'SELL'

        return 'HOLD'

    def execute_trade(
        self,
        symbol: str,
        signal_type: str,
        confidence: float = 1.0
    ) -> bool:
        """Execute a trade with money management."""
        if signal_type == 'HOLD':
            return True

        is_buy = signal_type in ['BUY', 'STRONG_BUY']

        # Check if trading is allowed
        can_trade, reason = self.money_manager.can_trade()
        if not can_trade:
            logger.warning(f"Trading not allowed: {reason}")
            self.telegram.notify_risk_alert("Trading Blocked", reason)
            return False

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return False

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return False

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error("Failed to get tick data")
            return False

        price = tick.ask if is_buy else tick.bid

        # Calculate ATR for dynamic SL/TP
        data = self.get_market_data(symbol, config.TIMEFRAME, 50)
        atr = self.calculate_atr(data) if not data.empty else None

        # Calculate stop loss and take profit
        sl_price, sl_pips = self.money_manager.calculate_stop_loss(
            symbol, price, is_buy, atr
        )
        tp_price, tp_pips = self.money_manager.calculate_take_profit(
            symbol, price, is_buy, sl_pips
        )

        # Calculate position size
        position = self.money_manager.calculate_position_size(
            symbol, sl_pips, tp_pips, confidence
        )
        if not position:
            logger.error("Failed to calculate position size")
            return False

        # Execute the trade
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.lots,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": 20,
            "magic": 123456,
            "comment": f"SmartTrade-{signal_type}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.retcode} - {result.comment}"
            logger.error(error_msg)
            self.telegram.notify_error(error_msg)
            return False

        logger.info(f"Order executed: {signal_type} {position.lots} {symbol} at {price}")

        # Send Telegram notification
        self.telegram.notify_trade_opened(
            order_type="BUY" if is_buy else "SELL",
            symbol=symbol,
            lots=position.lots,
            price=price,
            sl=sl_price,
            tp=tp_price,
            risk_amount=position.risk_amount,
            risk_percent=position.risk_percent
        )

        self.trades_today += 1
        return True

    def check_open_positions(self) -> int:
        """Check number of open positions for the symbol."""
        positions = mt5.positions_get(symbol=config.SYMBOL)
        return len(positions) if positions else 0

    def manage_open_positions(self):
        """Manage existing positions (trailing stops, etc.)."""
        positions = mt5.positions_get(symbol=config.SYMBOL)
        if not positions:
            return

        for pos in positions:
            # Update trailing stop
            if config.USE_TRAILING_STOP:
                self.money_manager.update_trailing_stop(pos)

    def send_daily_summary(self):
        """Send daily trading summary if not sent today."""
        today = date.today()
        if self.last_summary_date == today:
            return

        # Only send at specific hour (e.g., 18:00)
        if datetime.now().hour != 18:
            return

        metrics = self.money_manager.get_risk_metrics()
        if not metrics:
            return

        stats = self.money_manager.get_statistics()

        self.telegram.notify_daily_summary(
            balance=metrics.account_balance,
            equity=metrics.account_equity,
            daily_pnl=metrics.daily_pnl,
            daily_pnl_percent=metrics.daily_pnl_percent,
            trades_today=self.trades_today,
            win_rate=stats.get('win_rate', 0)
        )

        self.last_summary_date = today
        self.trades_today = 0  # Reset for next day

    def run(self):
        """Main bot loop with Dual-LLM decision system."""
        logger.info("=" * 60)
        logger.info("  Claude Smart Trade Bot v2.0 - Dual-LLM System")
        logger.info("=" * 60)

        if not self.connect():
            logger.error("Failed to connect to MT5. Exiting.")
            return

        logger.info(f"Trading {config.SYMBOL} on {config.TIMEFRAME}")
        logger.info(f"Risk per trade: {config.RISK_PER_TRADE}%")
        logger.info(f"Max daily loss: {config.MAX_DAILY_LOSS}%")
        logger.info(f"Dual-LLM: {'Enabled' if config.USE_DUAL_LLM else 'Disabled'}")
        logger.info(f"Telegram: {'Enabled' if config.TELEGRAM_ENABLED else 'Disabled'}")
        logger.info("=" * 60)

        # Notify bot started
        self.telegram.notify_bot_started()

        while self.running:
            try:
                # Get market data
                data = self.get_market_data(config.SYMBOL, config.TIMEFRAME)

                if data.empty:
                    logger.warning("No market data received")
                    time.sleep(config.CHECK_INTERVAL)
                    continue

                current_price = data.iloc[-1]['close']
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Get trading signal
                if config.USE_DUAL_LLM and self.dual_agent:
                    # Dual-LLM Decision System
                    logger.info("Consulting LLM agents...")
                    news = get_financial_news(config.SYMBOL)
                    consensus = self.dual_agent.get_consensus(data, news)

                    signal_str = consensus.final_signal.value
                    confidence = consensus.confidence
                    trade_recommended = consensus.trade_recommended

                    # Notify about signal
                    self.telegram.notify_signal(
                        signal=signal_str,
                        confidence=confidence,
                        agent1_signal=consensus.agent1_analysis.signal.value,
                        agent2_signal=consensus.agent2_analysis.signal.value,
                        reasoning=consensus.consensus_reasoning
                    )

                    # Notify about LLM conversation
                    self.telegram.notify_llm_conversation(
                        agent1_name=consensus.agent1_analysis.agent_name,
                        agent1_analysis=consensus.agent1_analysis.reasoning,
                        agent2_name=consensus.agent2_analysis.agent_name,
                        agent2_analysis=consensus.agent2_analysis.reasoning,
                        consensus=f"Final: {signal_str} ({confidence:.1%})"
                    )
                else:
                    # Fallback to simple strategy
                    signal_str = self.get_fallback_signal(data)
                    confidence = 0.7
                    trade_recommended = signal_str != 'HOLD'

                logger.info(f"[{timestamp}] {config.SYMBOL}: {current_price:.5f} | "
                           f"Signal: {signal_str} | Confidence: {confidence:.1%}")

                # Check open positions
                open_positions = self.check_open_positions()

                # Manage existing positions
                self.manage_open_positions()

                # Execute trade if recommended and no open position
                if trade_recommended and open_positions == 0:
                    self.execute_trade(config.SYMBOL, signal_str, confidence)
                elif open_positions > 0:
                    logger.info(f"Already have {open_positions} open position(s)")

                # Check for daily summary
                self.send_daily_summary()

                time.sleep(config.CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.telegram.notify_error(f"Main loop error: {e}")
                time.sleep(config.CHECK_INTERVAL)

        self.disconnect()
        self.telegram.notify_bot_stopped("Normal shutdown")
        logger.info("Bot stopped")


def main():
    """Entry point."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   Claude Smart Trade Bot v2.0                             ║
    ║   Dual-LLM Trading Decision System                        ║
    ║                                                           ║
    ║   FinGPT (Sentiment) + FinLLaMA (Technical)               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    bot = SmartTradeBot()
    bot.run()


if __name__ == "__main__":
    main()
