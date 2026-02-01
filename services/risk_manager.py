"""
Advanced Risk Management Service
Multi-layered risk controls with circuit breaker
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    account_balance: float
    account_equity: float
    free_margin: float
    daily_pnl: float
    daily_pnl_percent: float
    open_positions: int
    total_exposure: float
    current_drawdown: float
    max_drawdown_reached: bool
    daily_loss_limit_reached: bool
    circuit_breaker_active: bool


@dataclass
class PositionSize:
    """Calculated position size"""
    lots: float
    units: int
    risk_amount: float
    risk_percent: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float


class RiskManager:
    """
    Multi-layered risk management system.
    
    Features:
    - Position sizing based on risk percentage
    - Daily loss limits
    - Maximum drawdown protection
    - Circuit breaker for consecutive losses
    - Correlation-based position limits
    - Exposure limits
    """

    def __init__(self):
        # Account level limits
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', '0.15'))
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        self.max_positions = int(os.getenv('MAX_POSITIONS', '3'))
        self.max_exposure = float(os.getenv('MAX_EXPOSURE', '0.30'))
        
        # Per-trade limits
        self.risk_per_trade = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.min_rr_ratio = float(os.getenv('MIN_RR_RATIO', '1.5'))
        self.max_leverage = int(os.getenv('MAX_LEVERAGE', '10'))
        
        # Circuit breaker state
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.circuit_breaker_active = False
        self.circuit_breaker_reset_time: Optional[datetime] = None
        self.circuit_breaker_duration = timedelta(hours=4)
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_balance = 0.0
        self.last_reset_date: Optional[datetime] = None
        
        # Drawdown tracking
        self.peak_balance = 0.0
        self.current_drawdown = 0.0
        
        # Correlation pairs (avoid correlated positions)
        self.correlated_pairs = {
            'EURUSD': ['GBPUSD', 'EURGBP'],
            'GBPUSD': ['EURUSD', 'EURGBP'],
            'USDJPY': ['EURJPY', 'GBPJPY'],
            'XAUUSD': ['XAGUSD'],
            'BTCUSD': ['ETHUSD', 'BTCUSDT'],
            'ETHUSD': ['BTCUSD', 'ETHUSDT'],
            'BTCUSDT': ['ETHUSDT', 'BTCUSD'],
            'ETHUSDT': ['BTCUSDT', 'ETHUSD'],
        }
        
        # Trade history for statistics
        self.trade_history: List[Dict] = []

    def initialize(self, account_balance: float):
        """Initialize risk manager with account balance"""
        self.peak_balance = account_balance
        self.daily_start_balance = account_balance
        self.last_reset_date = datetime.now().date()
        logger.info(f"Risk Manager initialized - Balance: ${account_balance:.2f}")

    def daily_reset(self, current_balance: float):
        """Reset daily counters"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_start_balance = current_balance
            self.last_reset_date = today
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            logger.info(f"Daily reset - Starting balance: ${current_balance:.2f}")

    def update_metrics(self, current_balance: float, equity: float):
        """Update current risk metrics"""
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculate drawdown
        if self.peak_balance > 0:
            self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Calculate daily P&L
        if self.daily_start_balance > 0:
            self.daily_pnl = current_balance - self.daily_start_balance

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        symbol: str = None,
        take_profit: float = None
    ) -> Optional[PositionSize]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            account_balance: Current account balance
            symbol: Trading symbol (for pip calculation)
            take_profit: Take profit price (optional)
            
        Returns:
            PositionSize object or None if invalid
        """
        if stop_loss == 0 or entry_price == 0:
            return None
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_pips = stop_distance
        
        if stop_distance == 0:
            return None
        
        # Calculate risk amount
        risk_amount = account_balance * self.risk_per_trade
        
        # Calculate base units
        units = risk_amount / stop_distance
        
        # Apply exposure limit
        max_units_by_exposure = (account_balance * self.max_exposure) / entry_price
        units = min(units, max_units_by_exposure)
        
        # Apply leverage limit
        max_units_by_leverage = (account_balance * self.max_leverage) / entry_price
        units = min(units, max_units_by_leverage)
        
        # Round to appropriate lot size
        units = int(units)
        lots = units / 100000 if units > 1000 else units  # Forex vs Crypto
        
        # Calculate actual risk
        actual_risk = units * stop_distance
        risk_percent = actual_risk / account_balance if account_balance > 0 else 0
        
        # Calculate take profit pips and R:R ratio
        tp_pips = 0
        rr_ratio = 0
        if take_profit:
            tp_pips = abs(take_profit - entry_price)
            rr_ratio = tp_pips / stop_pips if stop_pips > 0 else 0
        
        return PositionSize(
            lots=lots,
            units=units,
            risk_amount=actual_risk,
            risk_percent=risk_percent,
            stop_loss_pips=stop_pips,
            take_profit_pips=tp_pips,
            risk_reward_ratio=rr_ratio
        )

    def validate_trade(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        current_positions: List[Dict],
        daily_pnl: float = 0
    ) -> Tuple[bool, str]:
        """
        Validate if a trade can be executed.
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            if self.circuit_breaker_reset_time and datetime.now() < self.circuit_breaker_reset_time:
                remaining = (self.circuit_breaker_reset_time - datetime.now()).seconds // 60
                return False, f"Circuit breaker active - {remaining} minutes remaining"
            else:
                self._reset_circuit_breaker()
        
        # Check daily loss limit
        if account_balance > 0:
            daily_loss_pct = abs(daily_pnl) / account_balance if daily_pnl < 0 else 0
            if daily_loss_pct >= self.max_daily_loss:
                return False, f"Daily loss limit reached ({self.max_daily_loss * 100}%)"
        
        # Check max drawdown
        if self.current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown reached ({self.max_drawdown * 100}%)"
        
        # Check max positions
        if len(current_positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check for duplicate position
        for position in current_positions:
            if position.get('symbol') == symbol:
                pos_direction = 'BUY' if position.get('units', 0) > 0 else 'SELL'
                if pos_direction == action:
                    return False, f"Duplicate position already exists for {symbol}"
        
        # Check for correlated positions
        correlated = self.correlated_pairs.get(symbol, [])
        for position in current_positions:
            if position.get('symbol') in correlated:
                return False, f"Correlated position exists: {position.get('symbol')}"
        
        # Validate risk-reward ratio
        if stop_loss and take_profit and entry_price:
            reward = abs(take_profit - entry_price)
            risk = abs(entry_price - stop_loss)
            
            if risk == 0:
                return False, "Invalid stop loss (same as entry)"
            
            rr_ratio = reward / risk
            if rr_ratio < self.min_rr_ratio:
                return False, f"Risk-reward ratio ({rr_ratio:.2f}) below minimum ({self.min_rr_ratio})"
        
        return True, "Trade validated"

    def record_trade_result(self, pnl: float, is_win: bool):
        """Record trade result for circuit breaker and statistics"""
        self.daily_trades += 1
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            
            # Activate circuit breaker if needed
            if self.consecutive_losses >= self.max_consecutive_losses:
                self._activate_circuit_breaker()
        
        # Record in history
        self.trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl,
            'is_win': is_win,
            'consecutive_losses': self.consecutive_losses
        })

    def _activate_circuit_breaker(self):
        """Activate circuit breaker after consecutive losses"""
        self.circuit_breaker_active = True
        self.circuit_breaker_reset_time = datetime.now() + self.circuit_breaker_duration
        logger.warning(f"Circuit breaker activated! {self.consecutive_losses} consecutive losses. "
                      f"Trading paused until {self.circuit_breaker_reset_time}")

    def _reset_circuit_breaker(self):
        """Reset circuit breaker"""
        self.circuit_breaker_active = False
        self.circuit_breaker_reset_time = None
        self.consecutive_losses = 0
        logger.info("Circuit breaker reset. Trading resumed.")

    def can_trade(self) -> Tuple[bool, str]:
        """Quick check if trading is allowed"""
        if self.circuit_breaker_active:
            if self.circuit_breaker_reset_time and datetime.now() < self.circuit_breaker_reset_time:
                return False, "Circuit breaker active"
            self._reset_circuit_breaker()
        
        if self.current_drawdown >= self.max_drawdown:
            return False, f"Max drawdown ({self.max_drawdown*100}%) reached"
        
        return True, "Trading allowed"

    def get_risk_metrics(self, account_balance: float, equity: float, open_positions: int) -> RiskMetrics:
        """Get current risk metrics snapshot"""
        self.update_metrics(account_balance, equity)
        
        daily_pnl_pct = 0
        if self.daily_start_balance > 0:
            daily_pnl_pct = (self.daily_pnl / self.daily_start_balance) * 100
        
        return RiskMetrics(
            account_balance=account_balance,
            account_equity=equity,
            free_margin=equity - (account_balance * 0.1 * open_positions),  # Rough estimate
            daily_pnl=self.daily_pnl,
            daily_pnl_percent=daily_pnl_pct,
            open_positions=open_positions,
            total_exposure=open_positions * 0.1,  # Rough estimate
            current_drawdown=self.current_drawdown,
            max_drawdown_reached=self.current_drawdown >= self.max_drawdown,
            daily_loss_limit_reached=abs(daily_pnl_pct) >= self.max_daily_loss * 100 if self.daily_pnl < 0 else False,
            circuit_breaker_active=self.circuit_breaker_active
        )

    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        if not self.trade_history:
            return {'trades': 0, 'win_rate': 0}
        
        wins = sum(1 for t in self.trade_history if t['is_win'])
        total = len(self.trade_history)
        
        return {
            'trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': sum(t['pnl'] for t in self.trade_history),
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.circuit_breaker_active
        }


# Global instance
risk_manager = RiskManager()
