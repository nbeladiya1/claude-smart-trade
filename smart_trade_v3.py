#!/usr/bin/env python3
"""
Claude Smart Trade Bot v3.0
Advanced AI-Powered Trading with OpenAI Consensus Engine

Features:
- OpenAI GPT-4 dual-analyst consensus system
- XGBoost ML predictions
- 15+ Technical indicators
- Advanced risk management with circuit breaker
- Binance crypto trading support
- Telegram notifications
"""
import logging
import time
import sys
import signal
import os
from datetime import datetime, date
from typing import Optional, Dict, List
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Import services
from services.binance import binance
from services.indicators import TechnicalIndicators
from services.risk_manager import risk_manager

# Import models
from models.openai_consensus import consensus_engine
from models.xgboost_model import get_predictor

# Ensure log directory exists
log_dir = os.getenv('LOG_DIR', '/var/log/smart-trade')
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, 'bot.log'))
    ]
)
logger = logging.getLogger(__name__)


class SmartTradeBot:
    """
    Advanced AI Trading Bot v3.0
    
    Combines:
    - OpenAI GPT-4 multi-analyst consensus
    - XGBoost machine learning predictions
    - Comprehensive technical analysis
    - Multi-layered risk management
    """

    def __init__(self):
        self.running = True
        self.connected = False
        
        # Configuration
        self.symbols = self._parse_symbols()
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '60'))
        self.trading_enabled = os.getenv('TRADING_MODE', 'demo') == 'live'
        self.simulation_mode = os.getenv('SIMULATION_MODE', 'true').lower() == 'true'
        
        # Track stats
        self.trades_today = 0
        self.last_summary_date: Optional[date] = None
        self.recent_signals: Dict = {}
        self.active_positions: Dict = {}
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _parse_symbols(self) -> List[str]:
        """Parse trading symbols from config"""
        symbols_str = os.getenv('CRYPTO_SYMBOLS', 'BTCUSD,ETHUSD')
        return [s.strip() for s in symbols_str.split(',') if s.strip()]

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.running = False

    def connect(self) -> bool:
        """Initialize connections and services"""
        logger.info("Initializing Smart Trade Bot v3.0...")
        
        # Check Binance connection
        if binance.is_available:
            if binance.check_connection():
                logger.info(f"✓ Connected to Binance{'(US)' if binance.is_binance_us else ''}")
                
                # Get account info
                account = binance.get_account_info()
                if account.get('success'):
                    logger.info(f"  Account can trade: {account.get('can_trade')}")
            else:
                logger.warning("✗ Binance connection failed")
        else:
            logger.warning("✗ Binance API keys not configured")
        
        # Check OpenAI
        if consensus_engine.is_available:
            logger.info(f"✓ OpenAI API configured (model: {consensus_engine.model})")
        else:
            logger.warning("✗ OpenAI API not configured - using fallback signals")
        
        # Initialize risk manager
        initial_balance = float(os.getenv('INITIAL_BALANCE', '10000'))
        risk_manager.initialize(initial_balance)
        logger.info(f"✓ Risk Manager initialized (Balance: ${initial_balance:.2f})")
        
        self.connected = True
        return True

    def get_market_data(self, symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
        """Get market data with calculated indicators"""
        # Get candles from Binance
        candles = binance.get_spot_klines(symbol, interval, limit)
        
        if not candles:
            logger.warning(f"No candle data for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Calculate all technical indicators
        df = TechnicalIndicators.calculate_all(df)
        
        return df

    def analyze_symbol(self, symbol: str) -> Dict:
        """Perform full analysis on a symbol"""
        logger.info(f"Analyzing {symbol}...")
        
        # Get market data
        df = self.get_market_data(symbol)
        if df.empty:
            return {'symbol': symbol, 'error': 'No market data'}
        
        latest = df.iloc[-1]
        current_price = float(latest['close'])
        
        # Calculate 24h change
        if len(df) >= 24:
            price_24h_ago = float(df.iloc[-24]['close'])
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
        else:
            change_24h = 0
        
        # Prepare indicators for AI
        indicators = {
            'rsi': float(latest.get('rsi', 50)),
            'macd': float(latest.get('macd', 0)),
            'macd_signal': float(latest.get('macd_signal', 0)),
            'ema20': float(latest.get('ema20', current_price)),
            'ema50': float(latest.get('ema50', current_price)),
            'ema200': float(latest.get('ema200', current_price)),
            'adx': float(latest.get('adx', 0)),
            'atr': float(latest.get('atr', 0)),
            'bb_upper': float(latest.get('bb_upper', current_price)),
            'bb_lower': float(latest.get('bb_lower', current_price)),
            'stoch_k': float(latest.get('stoch_k', 50)),
            'williams_r': float(latest.get('williams_r', -50)),
            'cci': float(latest.get('cci', 0)),
            'change_24h': change_24h
        }
        
        # Get XGBoost prediction
        xgb_signal = None
        xgb_confidence = None
        try:
            predictor = get_predictor(symbol)
            xgb_result = predictor.predict(df)
            if xgb_result.get('available'):
                xgb_signal = xgb_result['signal']
                xgb_confidence = xgb_result['confidence']
                logger.info(f"  XGBoost: {xgb_signal} ({xgb_confidence:.1f}%)")
        except Exception as e:
            logger.debug(f"XGBoost not available: {e}")
        
        # Get AI consensus
        consensus_result = consensus_engine.get_consensus(
            symbol=symbol,
            current_price=current_price,
            indicators=indicators,
            market_type='spot',
            timeframe='1h',
            xgb_signal=xgb_signal,
            xgb_confidence=xgb_confidence
        )
        
        consensus = consensus_result.get('consensus', {})
        
        # Determine if we should trade
        should_trade = (
            consensus.get('should_execute', False) and
            consensus.get('recommendation') != 'HOLD' and
            consensus.get('confidence', 0) >= 65 and
            symbol not in self.active_positions
        )
        
        # Get signal strength from indicators
        signal_strength = TechnicalIndicators.get_signal_strength(df)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'change_24h': change_24h,
            'indicators': indicators,
            'signal_strength': signal_strength,
            'xgb_signal': xgb_signal,
            'xgb_confidence': xgb_confidence,
            'consensus': consensus,
            'dialogue': consensus_result.get('dialogue', []),
            'should_trade': should_trade,
            'timestamp': datetime.now().isoformat()
        }

    def execute_trade(self, analysis: Dict) -> Dict:
        """Execute a trade based on analysis"""
        symbol = analysis['symbol']
        consensus = analysis['consensus']
        current_price = analysis['current_price']
        
        recommendation = consensus.get('recommendation')
        entry_price = consensus.get('entry_price', current_price)
        stop_loss = consensus.get('stop_loss')
        take_profit = consensus.get('take_profit')
        confidence = consensus.get('confidence', 0)
        
        logger.info(f"Attempting {recommendation} on {symbol} at {current_price}")
        
        # Validate with risk manager
        can_trade, reason = risk_manager.can_trade()
        if not can_trade:
            logger.warning(f"Trade blocked: {reason}")
            return {'success': False, 'error': reason}
        
        # Validate specific trade
        is_valid, reason = risk_manager.validate_trade(
            symbol=symbol,
            action=recommendation,
            entry_price=entry_price,
            stop_loss=stop_loss or entry_price * 0.98,
            take_profit=take_profit or entry_price * 1.04,
            account_balance=10000,  # TODO: Get from account
            current_positions=list(self.active_positions.values()),
            daily_pnl=risk_manager.daily_pnl
        )
        
        if not is_valid:
            logger.warning(f"Trade validation failed: {reason}")
            return {'success': False, 'error': reason}
        
        # Execute trade if not simulation
        if self.simulation_mode or not self.trading_enabled:
            logger.info(f"[SIMULATION] Would execute {recommendation} {symbol} at {current_price}")
            self.active_positions[symbol] = {
                'symbol': symbol,
                'side': recommendation,
                'entry_price': current_price,
                'units': 100,
                'timestamp': datetime.now().isoformat()
            }
            self.trades_today += 1
            return {
                'success': True,
                'simulation': True,
                'order': {
                    'symbol': symbol,
                    'side': recommendation,
                    'price': current_price
                }
            }
        
        # Real trade execution
        try:
            # Calculate quantity based on risk
            quote_qty = 100  # USD amount to trade
            
            if recommendation == 'BUY':
                result = binance.spot_market_order(
                    symbol=symbol,
                    side='BUY',
                    quote_quantity=quote_qty
                )
            else:
                # For sell, we need to have the asset
                result = binance.spot_market_order(
                    symbol=symbol,
                    side='SELL',
                    quote_quantity=quote_qty
                )
            
            if result.get('success'):
                logger.info(f"Trade executed: {result}")
                self.active_positions[symbol] = result
                self.trades_today += 1
                return {'success': True, 'order': result}
            else:
                logger.error(f"Trade failed: {result.get('error')}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {'success': False, 'error': str(e)}

    def scan_markets(self):
        """Scan all configured symbols"""
        logger.info(f"Scanning {len(self.symbols)} symbols...")
        
        results = []
        for symbol in self.symbols:
            try:
                analysis = self.analyze_symbol(symbol)
                results.append(analysis)
                
                consensus = analysis.get('consensus', {})
                recommendation = consensus.get('recommendation', 'HOLD')
                confidence = consensus.get('confidence', 0)
                
                # Log signal
                logger.info(
                    f"  {symbol}: ${analysis['current_price']:.2f} | "
                    f"{recommendation} ({confidence:.0f}%) | "
                    f"24h: {analysis['change_24h']:+.2f}%"
                )
                
                # Execute if should trade
                if analysis.get('should_trade'):
                    trade_result = self.execute_trade(analysis)
                    analysis['trade_result'] = trade_result
                
                # Store signal
                self.recent_signals[symbol] = {
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'price': analysis['current_price'],
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})
        
        # Summary
        actionable = [r for r in results if r.get('consensus', {}).get('recommendation') != 'HOLD']
        logger.info(f"Scan complete. {len(actionable)} actionable signals.")
        
        return results

    def run(self):
        """Main bot loop"""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   Claude Smart Trade Bot v3.0                                ║
    ║   AI-Powered Trading with OpenAI Consensus                   ║
    ║                                                              ║
    ║   Features:                                                  ║
    ║   • GPT-4 Dual-Analyst Consensus                             ║
    ║   • XGBoost ML Predictions                                   ║
    ║   • 15+ Technical Indicators                                 ║
    ║   • Circuit Breaker Risk Management                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        
        if not self.connect():
            logger.error("Failed to initialize. Exiting.")
            return
        
        logger.info("=" * 60)
        logger.info(f"Trading symbols: {', '.join(self.symbols)}")
        logger.info(f"Check interval: {self.check_interval}s")
        logger.info(f"Mode: {'SIMULATION' if self.simulation_mode else 'LIVE'}")
        logger.info(f"Risk per trade: {risk_manager.risk_per_trade * 100}%")
        logger.info(f"Max daily loss: {risk_manager.max_daily_loss * 100}%")
        logger.info("=" * 60)
        
        # Initial scan
        self.scan_markets()
        
        # Main loop
        while self.running:
            try:
                time.sleep(self.check_interval)
                
                # Daily reset check
                risk_manager.daily_reset(10000)  # TODO: Get real balance
                
                # Scan markets
                self.scan_markets()
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)
        
        logger.info("Bot stopped.")


def main():
    """Entry point"""
    bot = SmartTradeBot()
    bot.run()


if __name__ == "__main__":
    main()
