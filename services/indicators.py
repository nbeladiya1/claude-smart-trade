"""
Technical Indicators Service
Comprehensive indicator calculations for trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for market data"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Ensure column names are lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # RSI
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        # MACD
        macd_data = TechnicalIndicators.macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_diff'] = macd_data['histogram']
        
        # EMAs
        df['ema20'] = TechnicalIndicators.ema(df['close'], 20)
        df['ema50'] = TechnicalIndicators.ema(df['close'], 50)
        df['ema200'] = TechnicalIndicators.ema(df['close'], 200)
        
        # SMAs
        df['sma10'] = TechnicalIndicators.sma(df['close'], 10)
        df['sma50'] = TechnicalIndicators.sma(df['close'], 50)
        
        # ADX
        adx_data = TechnicalIndicators.adx(df)
        df['adx'] = adx_data['adx']
        df['plus_di'] = adx_data['plus_di']
        df['minus_di'] = adx_data['minus_di']
        
        # ATR
        df['atr'] = TechnicalIndicators.atr(df)
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(df['close'])
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Additional features for ML
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['price_to_ema50'] = df['close'] / df['ema50']
        df['price_to_ema200'] = df['close'] / df['ema200']
        df['ema_crossover'] = (df['ema50'] > df['ema200']).astype(int)
        
        # Stochastic
        stoch_data = TechnicalIndicators.stochastic(df)
        df['stoch_k'] = stoch_data['k']
        df['stoch_d'] = stoch_data['d']
        
        # Williams %R
        df['williams_r'] = TechnicalIndicators.williams_r(df)
        
        # CCI
        df['cci'] = TechnicalIndicators.cci(df)
        
        # MFI (if volume available)
        if 'volume' in df.columns:
            df['mfi'] = TechnicalIndicators.mfi(df)
        
        # OBV (if volume available)
        if 'volume' in df.columns:
            df['obv'] = TechnicalIndicators.obv(df)
        
        return df

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return series.rolling(window=period).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(window=period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        wr = -100 * (high_max - df['close']) / (high_max - low_min + 0.0001)
        return wr

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_dev + 0.0001)
        return cci

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 0.0001)))
        return mfi

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def get_signal_strength(df: pd.DataFrame) -> Dict:
        """
        Calculate overall signal strength from multiple indicators.
        Returns a composite score and individual indicator signals.
        """
        latest = df.iloc[-1]
        signals = {}
        bullish_count = 0
        bearish_count = 0
        
        # RSI Signal
        rsi = latest.get('rsi', 50)
        if rsi < 30:
            signals['rsi'] = 'OVERSOLD (Bullish)'
            bullish_count += 1
        elif rsi > 70:
            signals['rsi'] = 'OVERBOUGHT (Bearish)'
            bearish_count += 1
        else:
            signals['rsi'] = 'NEUTRAL'
        
        # MACD Signal
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        if macd > macd_signal:
            signals['macd'] = 'BULLISH'
            bullish_count += 1
        else:
            signals['macd'] = 'BEARISH'
            bearish_count += 1
        
        # EMA Crossover
        ema50 = latest.get('ema50', 0)
        ema200 = latest.get('ema200', 0)
        if ema50 > ema200:
            signals['ema_cross'] = 'BULLISH (Golden Cross)'
            bullish_count += 1
        else:
            signals['ema_cross'] = 'BEARISH (Death Cross)'
            bearish_count += 1
        
        # Stochastic
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k < 20:
            signals['stochastic'] = 'OVERSOLD (Bullish)'
            bullish_count += 1
        elif stoch_k > 80:
            signals['stochastic'] = 'OVERBOUGHT (Bearish)'
            bearish_count += 1
        else:
            signals['stochastic'] = 'NEUTRAL'
        
        # ADX Trend Strength
        adx = latest.get('adx', 0)
        if adx > 25:
            signals['adx'] = f'STRONG TREND ({adx:.1f})'
        else:
            signals['adx'] = f'WEAK TREND ({adx:.1f})'
        
        # Calculate composite score
        total_signals = bullish_count + bearish_count
        if total_signals > 0:
            bullish_ratio = bullish_count / total_signals
        else:
            bullish_ratio = 0.5
        
        if bullish_ratio > 0.6:
            overall = 'BULLISH'
        elif bullish_ratio < 0.4:
            overall = 'BEARISH'
        else:
            overall = 'NEUTRAL'
        
        return {
            'overall': overall,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'bullish_ratio': bullish_ratio,
            'signals': signals
        }
