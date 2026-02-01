"""
Telegram Bot Integration
Sends trading signals, updates, and alerts to Telegram.
"""
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import threading
import queue

import requests

import config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Handles all Telegram notifications for the trading bot.
    Supports async message sending to avoid blocking trades.
    """

    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = config.TELEGRAM_ENABLED and self.bot_token and self.chat_id
        self.message_queue = queue.Queue()
        self.sender_thread = None

        if self.enabled:
            self._start_sender_thread()
            logger.info("Telegram notifier initialized")
        else:
            logger.warning("Telegram notifier disabled (missing token or chat_id)")

    def _start_sender_thread(self):
        """Start background thread for sending messages."""
        self.sender_thread = threading.Thread(target=self._message_sender, daemon=True)
        self.sender_thread.start()

    def _message_sender(self):
        """Background thread that sends queued messages."""
        while True:
            try:
                message = self.message_queue.get(timeout=1)
                if message is None:
                    break
                self._send_message_sync(message)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Telegram sender error: {e}")

    def _send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message synchronously."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API error: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send(self, text: str):
        """Queue a message for sending."""
        if self.enabled:
            self.message_queue.put(text)

    def send_immediate(self, text: str) -> bool:
        """Send message immediately (blocking)."""
        return self._send_message_sync(text)

    # ==========================================
    # Trading Notifications
    # ==========================================

    def notify_bot_started(self):
        """Notify that the bot has started."""
        message = f"""
🤖 <b>SMART TRADE BOT STARTED</b>

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Symbol: {config.SYMBOL}
⏱️ Timeframe: {config.TIMEFRAME}
💰 Risk per trade: {config.RISK_PER_TRADE}%

🧠 <b>Dual-LLM System Active</b>
├─ FinGPT: Sentiment Analysis
└─ FinLLaMA: Technical Analysis

Bot is now monitoring the market...
"""
        self.send(message)

    def notify_bot_stopped(self, reason: str = "Manual stop"):
        """Notify that the bot has stopped."""
        message = f"""
🛑 <b>SMART TRADE BOT STOPPED</b>

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📝 Reason: {reason}
"""
        self.send_immediate(message)

    def notify_signal(
        self,
        signal: str,
        confidence: float,
        agent1_signal: str,
        agent2_signal: str,
        reasoning: str
    ):
        """Notify about a new trading signal."""
        if not config.NOTIFY_ON_SIGNAL:
            return

        emoji = self._get_signal_emoji(signal)

        message = f"""
{emoji} <b>NEW SIGNAL: {signal}</b>

📊 Symbol: {config.SYMBOL}
🎯 Confidence: {confidence:.1%}

<b>🧠 LLM Analysis:</b>
├─ FinGPT (Sentiment): {agent1_signal}
└─ FinLLaMA (Technical): {agent2_signal}

<b>📝 Reasoning:</b>
<i>{reasoning[:500]}...</i>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_trade_opened(
        self,
        order_type: str,
        symbol: str,
        lots: float,
        price: float,
        sl: float,
        tp: float,
        risk_amount: float,
        risk_percent: float
    ):
        """Notify about a trade being opened."""
        if not config.NOTIFY_ON_TRADE:
            return

        emoji = "📈" if "BUY" in order_type.upper() else "📉"

        message = f"""
{emoji} <b>TRADE OPENED</b>

📊 {symbol}
📋 Type: {order_type}
📦 Lots: {lots}
💵 Price: {price:.5f}

🛡️ <b>Risk Management:</b>
├─ Stop Loss: {sl:.5f}
├─ Take Profit: {tp:.5f}
├─ Risk Amount: ${risk_amount:.2f}
└─ Risk %: {risk_percent:.2f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_trade_closed(
        self,
        order_type: str,
        symbol: str,
        lots: float,
        profit: float,
        pips: float,
        close_reason: str = "Take Profit/Stop Loss"
    ):
        """Notify about a trade being closed."""
        if not config.NOTIFY_ON_TRADE:
            return

        emoji = "✅" if profit > 0 else "❌"
        profit_emoji = "💰" if profit > 0 else "💸"

        message = f"""
{emoji} <b>TRADE CLOSED</b>

📊 {symbol}
📋 Type: {order_type}
📦 Lots: {lots}

{profit_emoji} <b>Result:</b>
├─ Profit/Loss: ${profit:+.2f}
├─ Pips: {pips:+.1f}
└─ Reason: {close_reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_llm_conversation(
        self,
        agent1_name: str,
        agent1_analysis: str,
        agent2_name: str,
        agent2_analysis: str,
        consensus: str
    ):
        """Notify about LLM agents' conversation and consensus."""
        message = f"""
🧠 <b>LLM AGENTS DISCUSSION</b>

<b>👤 {agent1_name}:</b>
<i>"{agent1_analysis[:300]}..."</i>

<b>👤 {agent2_name}:</b>
<i>"{agent2_analysis[:300]}..."</i>

<b>🤝 CONSENSUS:</b>
{consensus}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_risk_alert(self, alert_type: str, details: str):
        """Notify about risk management alerts."""
        if not config.NOTIFY_ON_ERROR:
            return

        message = f"""
⚠️ <b>RISK ALERT: {alert_type}</b>

{details}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_immediate(message)

    def notify_error(self, error_message: str):
        """Notify about errors."""
        if not config.NOTIFY_ON_ERROR:
            return

        message = f"""
🚨 <b>ERROR</b>

{error_message}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_daily_summary(
        self,
        balance: float,
        equity: float,
        daily_pnl: float,
        daily_pnl_percent: float,
        trades_today: int,
        win_rate: float
    ):
        """Send daily trading summary."""
        if not config.NOTIFY_DAILY_SUMMARY:
            return

        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        message = f"""
📊 <b>DAILY TRADING SUMMARY</b>

💰 <b>Account Status:</b>
├─ Balance: ${balance:,.2f}
└─ Equity: ${equity:,.2f}

{pnl_emoji} <b>Today's Performance:</b>
├─ P&L: ${daily_pnl:+,.2f}
├─ P&L %: {daily_pnl_percent:+.2f}%
├─ Trades: {trades_today}
└─ Win Rate: {win_rate:.1f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_market_analysis(
        self,
        symbol: str,
        current_price: float,
        trend: str,
        key_levels: Dict[str, float],
        indicators: Dict[str, Any]
    ):
        """Send market analysis update."""
        support = key_levels.get('support', 0)
        resistance = key_levels.get('resistance', 0)
        rsi = indicators.get('rsi', 0)
        macd = indicators.get('macd', 'N/A')

        trend_emoji = "🟢" if trend == "Bullish" else "🔴" if trend == "Bearish" else "🟡"

        message = f"""
📈 <b>MARKET ANALYSIS: {symbol}</b>

💵 Price: {current_price:.5f}
{trend_emoji} Trend: {trend}

📊 <b>Key Levels:</b>
├─ Support: {support:.5f}
└─ Resistance: {resistance:.5f}

📉 <b>Indicators:</b>
├─ RSI: {rsi:.1f}
└─ MACD: {macd}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def notify_position_update(
        self,
        ticket: int,
        symbol: str,
        current_pnl: float,
        pips: float
    ):
        """Notify about position P&L update."""
        emoji = "📈" if current_pnl >= 0 else "📉"

        message = f"""
{emoji} <b>POSITION UPDATE</b>

🎫 Ticket: #{ticket}
📊 {symbol}
💰 Current P&L: ${current_pnl:+.2f}
📍 Pips: {pips:+.1f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send(message)

    def send_custom_message(self, message: str):
        """Send a custom message."""
        self.send(message)

    def _get_signal_emoji(self, signal: str) -> str:
        """Get emoji for signal type."""
        signal_emojis = {
            "STRONG_BUY": "🚀",
            "BUY": "📈",
            "HOLD": "⏸️",
            "SELL": "📉",
            "STRONG_SELL": "🔻"
        }
        return signal_emojis.get(signal.upper(), "📊")


# Global notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


def notify(func):
    """Decorator to send notification after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Notification logic here if needed
        return result
    return wrapper


# ==========================================
# Telegram Command Handlers (for future use)
# ==========================================

class TelegramCommandHandler:
    """
    Handles incoming Telegram commands.
    Can be extended to support interactive bot commands.
    """

    def __init__(self, notifier: TelegramNotifier):
        self.notifier = notifier
        self.commands = {
            "/status": self.cmd_status,
            "/balance": self.cmd_balance,
            "/positions": self.cmd_positions,
            "/stats": self.cmd_stats,
            "/stop": self.cmd_stop,
            "/start": self.cmd_start,
            "/help": self.cmd_help
        }

    def handle_command(self, command: str, args: list = None) -> str:
        """Handle incoming command."""
        handler = self.commands.get(command.lower())
        if handler:
            return handler(args)
        return "Unknown command. Use /help for available commands."

    def cmd_status(self, args=None) -> str:
        """Get bot status."""
        return "Bot is running and monitoring the market."

    def cmd_balance(self, args=None) -> str:
        """Get account balance."""
        return "Use the risk report for detailed balance info."

    def cmd_positions(self, args=None) -> str:
        """Get open positions."""
        return "Position information will be sent shortly."

    def cmd_stats(self, args=None) -> str:
        """Get trading statistics."""
        return "Statistics will be sent shortly."

    def cmd_stop(self, args=None) -> str:
        """Stop the bot."""
        return "Stop command received. Use system controls to stop the bot."

    def cmd_start(self, args=None) -> str:
        """Start trading."""
        return "Bot is already running."

    def cmd_help(self, args=None) -> str:
        """Show help message."""
        return """
<b>Available Commands:</b>
/status - Get bot status
/balance - Get account balance
/positions - Get open positions
/stats - Get trading statistics
/stop - Stop the bot
/help - Show this help message
"""
