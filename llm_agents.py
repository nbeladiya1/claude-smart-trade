"""
Dual-LLM Trading Agent System
Implements FinGPT (Sentiment) and FinLLaMA (Technical) agents that communicate
to reach consensus on trading decisions.
"""
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import requests
import pandas as pd

import config

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class AgentAnalysis:
    """Result from an LLM agent analysis."""
    agent_name: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    reasoning: str
    sentiment_score: float  # -1.0 to 1.0
    key_factors: List[str]


@dataclass
class ConsensusResult:
    """Result from dual-agent consensus."""
    final_signal: Signal
    confidence: float
    agent1_analysis: AgentAnalysis
    agent2_analysis: AgentAnalysis
    consensus_reasoning: str
    trade_recommended: bool


class BaseLLMAgent(ABC):
    """Base class for LLM trading agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, market_data: pd.DataFrame, news: List[str] = None) -> AgentAnalysis:
        """Analyze market conditions and return trading signal."""
        pass

    def _parse_signal(self, text: str) -> Signal:
        """Parse signal from LLM response."""
        text_upper = text.upper()
        if "STRONG_BUY" in text_upper or "STRONG BUY" in text_upper:
            return Signal.STRONG_BUY
        elif "STRONG_SELL" in text_upper or "STRONG SELL" in text_upper:
            return Signal.STRONG_SELL
        elif "BUY" in text_upper:
            return Signal.BUY
        elif "SELL" in text_upper:
            return Signal.SELL
        return Signal.HOLD


class FinGPTSentimentAgent(BaseLLMAgent):
    """
    FinGPT-style Sentiment Analysis Agent
    Focuses on market sentiment, news, and fundamental analysis.
    Uses OpenAI/Anthropic API with financial prompting.
    """

    def __init__(self):
        super().__init__("FinGPT-Sentiment")
        self.system_prompt = """You are FinGPT, an expert financial sentiment analysis AI trained on vast amounts of financial data.
Your role is to analyze market sentiment, news, and market conditions to provide trading recommendations.

You specialize in:
- Sentiment analysis of financial news and social media
- Identifying market trends and momentum
- Assessing risk sentiment (risk-on vs risk-off)
- Understanding macroeconomic factors

When analyzing, consider:
1. Overall market sentiment
2. News impact on the asset
3. Institutional positioning
4. Global economic conditions
5. Currency strength/weakness

Provide your analysis in JSON format:
{
    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "reasoning": "detailed explanation",
    "key_factors": ["factor1", "factor2", "factor3"]
}"""

    def analyze(self, market_data: pd.DataFrame, news: List[str] = None) -> AgentAnalysis:
        """Analyze market sentiment."""
        try:
            # Prepare market summary
            market_summary = self._prepare_market_summary(market_data)
            news_summary = "\n".join(news[:5]) if news else "No recent news available"

            prompt = f"""Analyze the following market data and news for {config.SYMBOL}:

MARKET DATA:
{market_summary}

RECENT NEWS:
{news_summary}

Provide your trading recommendation based on sentiment analysis."""

            response = self._call_llm(prompt)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"FinGPT analysis error: {e}")
            return AgentAnalysis(
                agent_name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                reasoning=f"Analysis failed: {e}",
                sentiment_score=0.0,
                key_factors=["Error in analysis"]
            )

    def _prepare_market_summary(self, data: pd.DataFrame) -> str:
        """Prepare market data summary for LLM."""
        if data.empty:
            return "No market data available"

        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else last

        change = ((last['close'] - prev['close']) / prev['close']) * 100
        high_low_range = last['high'] - last['low']

        # Calculate simple indicators
        sma_20 = data['close'].tail(20).mean() if len(data) >= 20 else last['close']
        sma_50 = data['close'].tail(50).mean() if len(data) >= 50 else last['close']

        return f"""
Current Price: {last['close']:.5f}
Change: {change:+.2f}%
High: {last['high']:.5f}
Low: {last['low']:.5f}
Range: {high_low_range:.5f}
SMA(20): {sma_20:.5f}
SMA(50): {sma_50:.5f}
Trend: {"Bullish" if last['close'] > sma_20 > sma_50 else "Bearish" if last['close'] < sma_20 < sma_50 else "Neutral"}
"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if config.LLM_PROVIDER == 'openai' and config.OPENAI_API_KEY:
            return self._call_openai(prompt)
        elif config.LLM_PROVIDER == 'anthropic' and config.ANTHROPIC_API_KEY:
            return self._call_anthropic(prompt)
        elif config.LLM_PROVIDER == 'huggingface' and config.HUGGINGFACE_API_KEY:
            return self._call_huggingface(prompt)
        else:
            return self._fallback_analysis(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.LLM_TEMPERATURE
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": config.ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['content'][0]['text']

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
        data = {"inputs": f"{self.system_prompt}\n\n{prompt}"}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{config.FINGPT_MODEL}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()[0]['generated_text']

    def _fallback_analysis(self, prompt: str) -> str:
        """Fallback rule-based analysis when no API is available."""
        logger.warning("No LLM API configured, using fallback analysis")
        return json.dumps({
            "signal": "HOLD",
            "confidence": 0.5,
            "sentiment_score": 0.0,
            "reasoning": "No LLM API configured. Using neutral stance.",
            "key_factors": ["No API available"]
        })

    def _parse_response(self, response: str) -> AgentAnalysis:
        """Parse LLM response into AgentAnalysis."""
        try:
            # Try to extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return AgentAnalysis(
                    agent_name=self.name,
                    signal=self._parse_signal(data.get('signal', 'HOLD')),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', ''),
                    sentiment_score=float(data.get('sentiment_score', 0.0)),
                    key_factors=data.get('key_factors', [])
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        # Fallback: parse from text
        return AgentAnalysis(
            agent_name=self.name,
            signal=self._parse_signal(response),
            confidence=0.5,
            reasoning=response[:500],
            sentiment_score=0.0,
            key_factors=["Parsed from text"]
        )


class FinLLaMATechnicalAgent(BaseLLMAgent):
    """
    FinLLaMA-style Technical Analysis Agent
    Focuses on technical indicators, price patterns, and quantitative analysis.
    """

    def __init__(self):
        super().__init__("FinLLaMA-Technical")
        self.system_prompt = """You are FinLLaMA, an expert financial technical analysis AI specialized in quantitative trading.
Your role is to analyze price action, technical indicators, and chart patterns to provide trading recommendations.

You specialize in:
- Technical indicator analysis (RSI, MACD, Moving Averages, Bollinger Bands)
- Price action and candlestick patterns
- Support and resistance levels
- Volume analysis
- Statistical analysis and quantitative metrics

When analyzing, consider:
1. Trend direction and strength
2. Momentum indicators
3. Overbought/oversold conditions
4. Key price levels
5. Pattern recognition

Provide your analysis in JSON format:
{
    "signal": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 0.0-1.0,
    "sentiment_score": -1.0 to 1.0,
    "reasoning": "detailed technical explanation",
    "key_factors": ["indicator1", "pattern1", "level1"]
}"""

    def analyze(self, market_data: pd.DataFrame, news: List[str] = None) -> AgentAnalysis:
        """Analyze technical indicators."""
        try:
            if market_data.empty:
                raise ValueError("No market data provided")

            # Calculate indicators
            indicators = self._calculate_indicators(market_data)
            prompt = f"""Analyze the following technical data for {config.SYMBOL}:

TECHNICAL INDICATORS:
{indicators}

Provide your trading recommendation based on technical analysis."""

            response = self._call_llm(prompt)
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"FinLLaMA analysis error: {e}")
            return AgentAnalysis(
                agent_name=self.name,
                signal=Signal.HOLD,
                confidence=0.0,
                reasoning=f"Analysis failed: {e}",
                sentiment_score=0.0,
                key_factors=["Error in analysis"]
            )

    def _calculate_indicators(self, data: pd.DataFrame) -> str:
        """Calculate technical indicators."""
        close = data['close']
        high = data['high']
        low = data['low']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Moving Averages
        sma_10 = close.rolling(window=10).mean()
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # MACD
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # Stochastic
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3).mean()

        last = data.iloc[-1]

        return f"""
Price: {last['close']:.5f}

MOVING AVERAGES:
- SMA(10): {sma_10.iloc[-1]:.5f}
- SMA(20): {sma_20.iloc[-1]:.5f}
- SMA(50): {sma_50.iloc[-1]:.5f}
- EMA(12): {ema_12.iloc[-1]:.5f}
- EMA(26): {ema_26.iloc[-1]:.5f}

MOMENTUM:
- RSI(14): {rsi.iloc[-1]:.2f}
- MACD Line: {macd_line.iloc[-1]:.5f}
- MACD Signal: {signal_line.iloc[-1]:.5f}
- MACD Histogram: {macd_histogram.iloc[-1]:.5f}

VOLATILITY:
- ATR(14): {atr.iloc[-1]:.5f}
- Bollinger Upper: {bb_upper.iloc[-1]:.5f}
- Bollinger Middle: {bb_middle.iloc[-1]:.5f}
- Bollinger Lower: {bb_lower.iloc[-1]:.5f}

STOCHASTIC:
- %K: {stoch_k.iloc[-1]:.2f}
- %D: {stoch_d.iloc[-1]:.2f}

TREND ANALYSIS:
- Price vs SMA20: {"Above" if last['close'] > sma_20.iloc[-1] else "Below"}
- Price vs SMA50: {"Above" if last['close'] > sma_50.iloc[-1] else "Below"}
- SMA10 vs SMA20: {"Golden Cross" if sma_10.iloc[-1] > sma_20.iloc[-1] else "Death Cross"}
- MACD Crossover: {"Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"}
"""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API (uses Anthropic by default for technical analysis)."""
        if config.ANTHROPIC_API_KEY:
            return self._call_anthropic(prompt)
        elif config.OPENAI_API_KEY:
            return self._call_openai(prompt)
        elif config.HUGGINGFACE_API_KEY:
            return self._call_huggingface(prompt)
        else:
            return self._fallback_analysis(prompt)

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": config.LLM_TEMPERATURE
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        headers = {
            "x-api-key": config.ANTHROPIC_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        data = {
            "model": config.ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['content'][0]['text']

    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API for FinLLaMA."""
        headers = {"Authorization": f"Bearer {config.HUGGINGFACE_API_KEY}"}
        data = {"inputs": f"{self.system_prompt}\n\n{prompt}"}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{config.FINLLAMA_MODEL}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()[0]['generated_text']

    def _fallback_analysis(self, prompt: str) -> str:
        """Fallback rule-based analysis."""
        logger.warning("No LLM API configured, using fallback analysis")
        return json.dumps({
            "signal": "HOLD",
            "confidence": 0.5,
            "sentiment_score": 0.0,
            "reasoning": "No LLM API configured. Using neutral stance.",
            "key_factors": ["No API available"]
        })

    def _parse_response(self, response: str) -> AgentAnalysis:
        """Parse LLM response into AgentAnalysis."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return AgentAnalysis(
                    agent_name=self.name,
                    signal=self._parse_signal(data.get('signal', 'HOLD')),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=data.get('reasoning', ''),
                    sentiment_score=float(data.get('sentiment_score', 0.0)),
                    key_factors=data.get('key_factors', [])
                )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")

        return AgentAnalysis(
            agent_name=self.name,
            signal=self._parse_signal(response),
            confidence=0.5,
            reasoning=response[:500],
            sentiment_score=0.0,
            key_factors=["Parsed from text"]
        )


class DualAgentConsensus:
    """
    Manages communication between two LLM agents to reach trading consensus.
    Implements the conversation between FinGPT and FinLLaMA.
    """

    def __init__(self):
        self.sentiment_agent = FinGPTSentimentAgent()
        self.technical_agent = FinLLaMATechnicalAgent()

    def get_consensus(self, market_data: pd.DataFrame, news: List[str] = None) -> ConsensusResult:
        """
        Get trading consensus from both agents.
        They analyze independently, then we determine consensus.
        """
        logger.info("=== Dual-LLM Consensus Analysis Starting ===")

        # Get analysis from both agents
        logger.info(f"[{self.sentiment_agent.name}] Analyzing market sentiment...")
        sentiment_analysis = self.sentiment_agent.analyze(market_data, news)
        logger.info(f"[{self.sentiment_agent.name}] Signal: {sentiment_analysis.signal.value}, "
                   f"Confidence: {sentiment_analysis.confidence:.2f}")

        logger.info(f"[{self.technical_agent.name}] Analyzing technical indicators...")
        technical_analysis = self.technical_agent.analyze(market_data, news)
        logger.info(f"[{self.technical_agent.name}] Signal: {technical_analysis.signal.value}, "
                   f"Confidence: {technical_analysis.confidence:.2f}")

        # Determine consensus
        consensus = self._determine_consensus(sentiment_analysis, technical_analysis)

        logger.info(f"=== Consensus Result: {consensus.final_signal.value} ===")
        logger.info(f"Trade Recommended: {consensus.trade_recommended}")

        return consensus

    def _determine_consensus(self, analysis1: AgentAnalysis, analysis2: AgentAnalysis) -> ConsensusResult:
        """Determine final consensus from both analyses."""

        signal_strength = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2
        }

        score1 = signal_strength[analysis1.signal]
        score2 = signal_strength[analysis2.signal]
        avg_score = (score1 + score2) / 2

        # Determine final signal based on average
        if avg_score >= 1.5:
            final_signal = Signal.STRONG_BUY
        elif avg_score >= 0.5:
            final_signal = Signal.BUY
        elif avg_score <= -1.5:
            final_signal = Signal.STRONG_SELL
        elif avg_score <= -0.5:
            final_signal = Signal.SELL
        else:
            final_signal = Signal.HOLD

        # Calculate combined confidence
        if analysis1.signal == analysis2.signal:
            # Full agreement
            combined_confidence = (analysis1.confidence + analysis2.confidence) / 2 * 1.2
        elif abs(score1 - score2) <= 1:
            # Partial agreement
            combined_confidence = (analysis1.confidence + analysis2.confidence) / 2
        else:
            # Disagreement
            combined_confidence = min(analysis1.confidence, analysis2.confidence) * 0.7

        combined_confidence = min(combined_confidence, 1.0)

        # Trade is recommended if both agree on direction or strong signal with high confidence
        same_direction = (score1 > 0 and score2 > 0) or (score1 < 0 and score2 < 0)
        trade_recommended = (
            (same_direction and combined_confidence >= 0.6) or
            (abs(avg_score) >= 1.5 and combined_confidence >= 0.7)
        )

        if config.LLM_CONSENSUS_REQUIRED and not same_direction:
            trade_recommended = False

        reasoning = f"""
DUAL-LLM CONSENSUS ANALYSIS:

Agent 1 ({analysis1.agent_name}):
- Signal: {analysis1.signal.value}
- Confidence: {analysis1.confidence:.2%}
- Key Factors: {', '.join(analysis1.key_factors)}
- Reasoning: {analysis1.reasoning[:200]}...

Agent 2 ({analysis2.agent_name}):
- Signal: {analysis2.signal.value}
- Confidence: {analysis2.confidence:.2%}
- Key Factors: {', '.join(analysis2.key_factors)}
- Reasoning: {analysis2.reasoning[:200]}...

CONSENSUS:
- Final Signal: {final_signal.value}
- Combined Confidence: {combined_confidence:.2%}
- Agreement: {"Full" if analysis1.signal == analysis2.signal else "Partial" if abs(score1 - score2) <= 1 else "Disagreement"}
- Trade Recommended: {"Yes" if trade_recommended else "No"}
"""

        return ConsensusResult(
            final_signal=final_signal,
            confidence=combined_confidence,
            agent1_analysis=analysis1,
            agent2_analysis=analysis2,
            consensus_reasoning=reasoning,
            trade_recommended=trade_recommended
        )


def get_financial_news(symbol: str = None) -> List[str]:
    """Fetch recent financial news."""
    news = []

    # Try NewsAPI
    if config.NEWS_API_KEY:
        try:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": symbol or "forex market",
                    "apiKey": config.NEWS_API_KEY,
                    "sortBy": "publishedAt",
                    "pageSize": 10
                },
                timeout=10
            )
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                news.extend([f"{a['title']}: {a['description']}" for a in articles if a.get('description')])
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")

    # Try Finnhub
    if config.FINNHUB_API_KEY:
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/news",
                params={
                    "category": "forex",
                    "token": config.FINNHUB_API_KEY
                },
                timeout=10
            )
            if response.status_code == 200:
                articles = response.json()
                news.extend([f"{a['headline']}: {a['summary']}" for a in articles[:10] if a.get('summary')])
        except Exception as e:
            logger.warning(f"Finnhub error: {e}")

    if not news:
        news = ["No recent news available. Using technical analysis only."]

    return news
