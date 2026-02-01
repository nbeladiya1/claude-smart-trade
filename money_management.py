"""
Money Management Module
Handles position sizing, risk management, and portfolio protection.
"""
import logging
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum

# Use MT5 wrapper for cross-platform compatibility
import mt5_wrapper as mt5

import config

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    PERCENT_RISK = "percent_risk"
    KELLY = "kelly"


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    account_balance: float
    account_equity: float
    free_margin: float
    margin_level: float
    daily_pnl: float
    daily_pnl_percent: float
    open_positions: int
    total_exposure: float
    max_drawdown_reached: bool
    daily_loss_limit_reached: bool


@dataclass
class PositionSize:
    """Calculated position size."""
    lots: float
    risk_amount: float
    risk_percent: float
    stop_loss_pips: int
    take_profit_pips: int
    risk_reward_ratio: float


class MoneyManager:
    """
    Comprehensive money management system.
    Handles position sizing, risk limits, and portfolio protection.
    """

    def __init__(self):
        self.daily_starting_balance: Optional[float] = None
        self.daily_start_date: Optional[date] = None
        self.peak_balance: float = 0.0
        self.trade_history: list = []

    def initialize(self) -> bool:
        """Initialize money manager with current account state."""
        account = mt5.account_info()
        if not account:
            logger.error("Failed to get account info")
            return False

        today = date.today()
        if self.daily_start_date != today:
            self.daily_starting_balance = account.balance
            self.daily_start_date = today
            logger.info(f"Daily balance reset: ${self.daily_starting_balance:.2f}")

        if account.balance > self.peak_balance:
            self.peak_balance = account.balance

        logger.info(f"Money Manager initialized - Balance: ${account.balance:.2f}, "
                   f"Peak: ${self.peak_balance:.2f}")
        return True

    def get_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get current risk metrics."""
        account = mt5.account_info()
        if not account:
            return None

        # Calculate daily P&L
        daily_pnl = account.balance - (self.daily_starting_balance or account.balance)
        daily_pnl_percent = (daily_pnl / self.daily_starting_balance * 100) if self.daily_starting_balance else 0

        # Calculate drawdown
        drawdown_percent = ((self.peak_balance - account.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0

        # Check limits
        max_drawdown_reached = drawdown_percent >= config.MAX_DRAWDOWN
        daily_loss_limit_reached = daily_pnl_percent <= -config.MAX_DAILY_LOSS

        # Get open positions
        positions = mt5.positions_get()
        open_positions = len(positions) if positions else 0

        # Calculate total exposure
        total_exposure = sum(p.volume for p in positions) if positions else 0.0

        return RiskMetrics(
            account_balance=account.balance,
            account_equity=account.equity,
            free_margin=account.margin_free,
            margin_level=account.margin_level if account.margin_level else 0,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            open_positions=open_positions,
            total_exposure=total_exposure,
            max_drawdown_reached=max_drawdown_reached,
            daily_loss_limit_reached=daily_loss_limit_reached
        )

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        metrics = self.get_risk_metrics()
        if not metrics:
            return False, "Unable to get account metrics"

        # Check max drawdown
        if metrics.max_drawdown_reached:
            return False, f"Max drawdown limit reached ({config.MAX_DRAWDOWN}%)"

        # Check daily loss limit
        if metrics.daily_loss_limit_reached:
            return False, f"Daily loss limit reached ({config.MAX_DAILY_LOSS}%)"

        # Check max open positions
        if metrics.open_positions >= config.MAX_OPEN_TRADES:
            return False, f"Max open trades reached ({config.MAX_OPEN_TRADES})"

        # Check margin level
        if metrics.margin_level > 0 and metrics.margin_level < 150:
            return False, f"Margin level too low ({metrics.margin_level:.1f}%)"

        return True, "Trading allowed"

    def calculate_position_size(
        self,
        symbol: str,
        stop_loss_pips: int,
        take_profit_pips: int,
        signal_confidence: float = 1.0
    ) -> Optional[PositionSize]:
        """Calculate optimal position size based on risk parameters."""
        account = mt5.account_info()
        symbol_info = mt5.symbol_info(symbol)

        if not account or not symbol_info:
            logger.error("Failed to get account or symbol info")
            return None

        method = config.POSITION_SIZING_METHOD

        if method == "fixed":
            lots = config.LOT_SIZE
        elif method == "percent_risk":
            lots = self._calculate_percent_risk_size(
                account.balance,
                symbol_info,
                stop_loss_pips,
                signal_confidence
            )
        elif method == "kelly":
            lots = self._calculate_kelly_size(
                account.balance,
                symbol_info,
                stop_loss_pips,
                signal_confidence
            )
        else:
            lots = config.LOT_SIZE

        # Apply limits
        lots = max(symbol_info.volume_min, min(lots, symbol_info.volume_max))
        lots = round(lots / symbol_info.volume_step) * symbol_info.volume_step

        # Calculate risk amount
        pip_value = self._get_pip_value(symbol_info, lots)
        risk_amount = stop_loss_pips * pip_value
        risk_percent = (risk_amount / account.balance) * 100

        # Calculate risk-reward ratio
        rr_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 0

        return PositionSize(
            lots=lots,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            stop_loss_pips=stop_loss_pips,
            take_profit_pips=take_profit_pips,
            risk_reward_ratio=rr_ratio
        )

    def _calculate_percent_risk_size(
        self,
        balance: float,
        symbol_info,
        stop_loss_pips: int,
        confidence: float
    ) -> float:
        """Calculate position size based on percentage risk."""
        # Adjust risk based on confidence
        adjusted_risk = config.RISK_PER_TRADE * confidence

        # Risk amount in account currency
        risk_amount = balance * (adjusted_risk / 100)

        # Pip value calculation
        pip_value_per_lot = self._get_pip_value(symbol_info, 1.0)

        if pip_value_per_lot <= 0 or stop_loss_pips <= 0:
            return config.LOT_SIZE

        # Position size in lots
        lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

        return lots

    def _calculate_kelly_size(
        self,
        balance: float,
        symbol_info,
        stop_loss_pips: int,
        confidence: float
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        # Default win rate and risk-reward if no history
        win_rate = 0.55  # Assume 55% win rate
        avg_win = config.TAKE_PROFIT
        avg_loss = config.STOP_LOSS

        # Use trade history if available
        if len(self.trade_history) >= 20:
            wins = [t for t in self.trade_history if t['profit'] > 0]
            losses = [t for t in self.trade_history if t['profit'] < 0]
            if wins and losses:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = sum(t['profit'] for t in wins) / len(wins)
                avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 2
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b if b > 0 else 0
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Adjust by confidence
        kelly_fraction *= confidence

        # Half-Kelly for safety
        kelly_fraction *= 0.5

        # Calculate position size
        risk_amount = balance * kelly_fraction
        pip_value_per_lot = self._get_pip_value(symbol_info, 1.0)

        if pip_value_per_lot <= 0 or stop_loss_pips <= 0:
            return config.LOT_SIZE

        lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

        return lots

    def _get_pip_value(self, symbol_info, lots: float) -> float:
        """Calculate pip value for the given lot size."""
        # For forex pairs
        if symbol_info.digits == 5 or symbol_info.digits == 3:
            # Pip is 0.0001 for 5-digit or 0.01 for 3-digit
            pip_size = 10 * symbol_info.point
        else:
            pip_size = symbol_info.point

        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size

        if tick_size <= 0:
            return 0

        pip_value = (pip_size / tick_size) * tick_value * lots

        return pip_value

    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        is_buy: bool,
        atr_value: float = None
    ) -> Tuple[float, int]:
        """Calculate dynamic stop loss level."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0, config.STOP_LOSS

        point = symbol_info.point
        pip_size = 10 * point if symbol_info.digits in [3, 5] else point

        # Use ATR-based stop if available
        if atr_value and atr_value > 0:
            sl_distance = atr_value * 1.5
            sl_pips = int(sl_distance / pip_size)
        else:
            sl_pips = config.STOP_LOSS

        # Ensure minimum stop loss
        sl_pips = max(sl_pips, 10)

        if is_buy:
            sl_price = entry_price - (sl_pips * pip_size)
        else:
            sl_price = entry_price + (sl_pips * pip_size)

        return sl_price, sl_pips

    def calculate_take_profit(
        self,
        symbol: str,
        entry_price: float,
        is_buy: bool,
        stop_loss_pips: int,
        min_rr_ratio: float = 1.5
    ) -> Tuple[float, int]:
        """Calculate take profit level with minimum risk-reward ratio."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0, config.TAKE_PROFIT

        point = symbol_info.point
        pip_size = 10 * point if symbol_info.digits in [3, 5] else point

        # Calculate TP based on RR ratio
        tp_pips = int(stop_loss_pips * min_rr_ratio)
        tp_pips = max(tp_pips, config.TAKE_PROFIT)

        if is_buy:
            tp_price = entry_price + (tp_pips * pip_size)
        else:
            tp_price = entry_price - (tp_pips * pip_size)

        return tp_price, tp_pips

    def update_trailing_stop(self, position) -> bool:
        """Update trailing stop for an open position."""
        if not config.USE_TRAILING_STOP:
            return False

        symbol_info = mt5.symbol_info(position.symbol)
        if not symbol_info:
            return False

        point = symbol_info.point
        pip_size = 10 * point if symbol_info.digits in [3, 5] else point
        trailing_distance = config.TRAILING_STOP_PIPS * pip_size

        current_price = mt5.symbol_info_tick(position.symbol)
        if not current_price:
            return False

        if position.type == mt5.ORDER_TYPE_BUY:
            new_sl = current_price.bid - trailing_distance
            if new_sl > position.sl and new_sl > position.price_open:
                return self._modify_position_sl(position, new_sl)
        else:
            new_sl = current_price.ask + trailing_distance
            if new_sl < position.sl and new_sl < position.price_open:
                return self._modify_position_sl(position, new_sl)

        return False

    def _modify_position_sl(self, position, new_sl: float) -> bool:
        """Modify position stop loss."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": position.ticket,
            "sl": new_sl,
            "tp": position.tp
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Trailing stop updated for position {position.ticket}: SL={new_sl:.5f}")
            return True
        return False

    def record_trade(self, trade_result: Dict):
        """Record trade result for statistics."""
        self.trade_history.append({
            "timestamp": datetime.now(),
            "symbol": trade_result.get("symbol"),
            "type": trade_result.get("type"),
            "lots": trade_result.get("lots"),
            "profit": trade_result.get("profit", 0),
            "pips": trade_result.get("pips", 0)
        })
        # Keep last 100 trades
        self.trade_history = self.trade_history[-100:]

    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        if not self.trade_history:
            return {"message": "No trade history available"}

        profits = [t['profit'] for t in self.trade_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        total_trades = len(self.trade_history)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0
        total_pnl = sum(profits)

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0
        }

    def get_risk_report(self) -> str:
        """Generate a risk management report."""
        metrics = self.get_risk_metrics()
        stats = self.get_statistics()

        if not metrics:
            return "Unable to generate risk report"

        report = f"""
╔══════════════════════════════════════════════╗
║           RISK MANAGEMENT REPORT              ║
╠══════════════════════════════════════════════╣
║ ACCOUNT STATUS                                ║
╟──────────────────────────────────────────────╢
║ Balance:        ${metrics.account_balance:>15,.2f}        ║
║ Equity:         ${metrics.account_equity:>15,.2f}        ║
║ Free Margin:    ${metrics.free_margin:>15,.2f}        ║
║ Margin Level:   {metrics.margin_level:>15.1f}%        ║
╟──────────────────────────────────────────────╢
║ DAILY PERFORMANCE                             ║
╟──────────────────────────────────────────────╢
║ Daily P&L:      ${metrics.daily_pnl:>+15,.2f}        ║
║ Daily P&L %:    {metrics.daily_pnl_percent:>+15.2f}%        ║
╟──────────────────────────────────────────────╢
║ RISK LIMITS                                   ║
╟──────────────────────────────────────────────╢
║ Open Positions: {metrics.open_positions:>3} / {config.MAX_OPEN_TRADES:<3}                    ║
║ Max Drawdown:   {'REACHED' if metrics.max_drawdown_reached else 'OK':>15}        ║
║ Daily Loss:     {'LIMIT HIT' if metrics.daily_loss_limit_reached else 'OK':>15}        ║
╟──────────────────────────────────────────────╢
║ TRADING STATISTICS                            ║
╟──────────────────────────────────────────────╢
║ Total Trades:   {stats.get('total_trades', 0):>15}        ║
║ Win Rate:       {stats.get('win_rate', 0):>14.1f}%        ║
║ Profit Factor:  {stats.get('profit_factor', 0):>15.2f}        ║
║ Total P&L:      ${stats.get('total_pnl', 0):>+14,.2f}        ║
╚══════════════════════════════════════════════╝
"""
        return report
