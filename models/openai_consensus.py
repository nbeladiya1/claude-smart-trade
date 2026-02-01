"""
OpenAI-Based AI Consensus Engine
Multi-round AI discussion using GPT-4 for trading decisions
Replaces local LLaMA/DeepSeek with OpenAI API
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIConsensusEngine:
    """
    Multi-AI consensus system using OpenAI GPT-4.
    Simulates two trading experts having a discussion:
    - Expert 1: Sentiment & Fundamental Analyst
    - Expert 2: Technical Analyst
    
    Flow:
    1. Expert 1 provides initial analysis (sentiment/fundamental)
    2. Expert 2 reviews and provides technical analysis
    3. Expert 1 responds to technical points
    4. System synthesizes final consensus
    """

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
        self.temperature = float(os.getenv('AI_TEMPERATURE', '0.3'))
        self.max_rounds = int(os.getenv('AI_CONSENSUS_ROUNDS', '3'))
        self.required_agreement = float(os.getenv('AI_CONSENSUS_REQUIRED_AGREEMENT', '0.7'))
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.is_available = bool(self.api_key)

    def _call_openai(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        """Make OpenAI API call with error handling."""
        if not self.client:
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""

    def get_consensus(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict,
        market_type: str = 'spot',
        timeframe: str = '1h',
        xgb_signal: str = None,
        xgb_confidence: float = None,
        lstm_prediction: float = None
    ) -> Dict:
        """
        Run multi-round AI consensus process.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            indicators: Technical indicators dict
            market_type: 'spot', 'futures', or 'options'
            timeframe: Analysis timeframe
            xgb_signal: XGBoost model signal (optional)
            xgb_confidence: XGBoost confidence (optional)
            lstm_prediction: LSTM price prediction (optional)
            
        Returns:
            Consensus decision with full dialogue history
        """
        dialogue = []
        
        if not self.is_available:
            return self._fallback_decision(symbol, indicators, xgb_signal, xgb_confidence)

        # Prepare market context
        market_context = self._prepare_market_context(
            symbol, current_price, indicators, market_type, timeframe,
            xgb_signal, xgb_confidence, lstm_prediction
        )

        # ROUND 1: Sentiment Analyst Initial Analysis
        round1_start = datetime.utcnow()
        
        sentiment_prompt = f"""You are an expert financial sentiment and fundamental analyst. Analyze this trading opportunity:

{market_context}

Provide your analysis in JSON format:
{{
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "sentiment": "bullish" or "bearish" or "neutral",
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "risk_level": "Low" or "Medium" or "High",
    "key_factors": ["factor1", "factor2"],
    "reasoning": "brief explanation"
}}"""

        sentiment_response = self._call_openai([
            {"role": "system", "content": "You are a professional trading analyst. Always respond with valid JSON only."},
            {"role": "user", "content": sentiment_prompt}
        ])
        
        sentiment_analysis = self._parse_json_response(sentiment_response)
        
        dialogue.append({
            'round': 1,
            'model': 'GPT-4 (Sentiment Analyst)',
            'role': 'Initial Analysis',
            'recommendation': sentiment_analysis.get('recommendation', 'HOLD'),
            'confidence': sentiment_analysis.get('confidence', 50),
            'entry': sentiment_analysis.get('entry_price'),
            'stop_loss': sentiment_analysis.get('stop_loss'),
            'take_profit': sentiment_analysis.get('take_profit'),
            'risk': sentiment_analysis.get('risk_level', 'Medium'),
            'reasoning': sentiment_analysis.get('reasoning', ''),
            'timestamp': round1_start.isoformat()
        })

        # ROUND 2: Technical Analyst Review
        round2_start = datetime.utcnow()
        
        technical_prompt = f"""You are an expert technical analyst. Review this trading analysis and provide your perspective:

{market_context}

Previous Analyst's View:
- Recommendation: {sentiment_analysis.get('recommendation', 'HOLD')}
- Confidence: {sentiment_analysis.get('confidence', 50)}%
- Entry: {sentiment_analysis.get('entry_price')}
- Stop Loss: {sentiment_analysis.get('stop_loss')}
- Take Profit: {sentiment_analysis.get('take_profit')}
- Reasoning: {sentiment_analysis.get('reasoning', '')}

Based on the technical indicators, do you agree? Provide your analysis in JSON:
{{
    "agrees_with_analyst": true or false,
    "recommendation": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "technical_outlook": "bullish" or "bearish" or "neutral",
    "entry_price": number (your suggested entry),
    "stop_loss": number (your suggested SL),
    "take_profit": number (your suggested TP),
    "concerns": ["concern1", "concern2"],
    "key_levels": {{"support": number, "resistance": number}},
    "reasoning": "brief explanation"
}}"""

        technical_response = self._call_openai([
            {"role": "system", "content": "You are a professional technical analyst. Always respond with valid JSON only."},
            {"role": "user", "content": technical_prompt}
        ])
        
        technical_analysis = self._parse_json_response(technical_response)
        
        dialogue.append({
            'round': 2,
            'model': 'GPT-4 (Technical Analyst)',
            'role': 'Technical Review',
            'agrees': technical_analysis.get('agrees_with_analyst', True),
            'recommendation': technical_analysis.get('recommendation', 'HOLD'),
            'confidence': technical_analysis.get('confidence', 50),
            'entry': technical_analysis.get('entry_price'),
            'stop_loss': technical_analysis.get('stop_loss'),
            'take_profit': technical_analysis.get('take_profit'),
            'concerns': technical_analysis.get('concerns', []),
            'reasoning': technical_analysis.get('reasoning', ''),
            'timestamp': round2_start.isoformat()
        })

        # ROUND 3: Final Synthesis (if disagreement)
        if sentiment_analysis.get('recommendation') != technical_analysis.get('recommendation'):
            round3_start = datetime.utcnow()
            
            synthesis_prompt = f"""Two analysts have different views on this trade:

Sentiment Analyst: {sentiment_analysis.get('recommendation')} ({sentiment_analysis.get('confidence')}% confident)
- {sentiment_analysis.get('reasoning', '')}

Technical Analyst: {technical_analysis.get('recommendation')} ({technical_analysis.get('confidence')}% confident)
- {technical_analysis.get('reasoning', '')}
- Concerns: {technical_analysis.get('concerns', [])}

As a senior risk manager, synthesize their views and provide the final trading decision in JSON:
{{
    "final_recommendation": "BUY" or "SELL" or "HOLD",
    "final_confidence": 0-100,
    "entry_price": number,
    "stop_loss": number,
    "take_profit": number,
    "risk_level": "Low" or "Medium" or "High",
    "should_execute": true or false,
    "reasoning": "synthesis explanation"
}}"""

            synthesis_response = self._call_openai([
                {"role": "system", "content": "You are a senior risk manager. Always respond with valid JSON only."},
                {"role": "user", "content": synthesis_prompt}
            ])
            
            synthesis = self._parse_json_response(synthesis_response)
            
            dialogue.append({
                'round': 3,
                'model': 'GPT-4 (Risk Manager)',
                'role': 'Final Synthesis',
                'recommendation': synthesis.get('final_recommendation', 'HOLD'),
                'confidence': synthesis.get('final_confidence', 50),
                'should_execute': synthesis.get('should_execute', False),
                'reasoning': synthesis.get('reasoning', ''),
                'timestamp': round3_start.isoformat()
            })
            
            final_analysis = synthesis
        else:
            # Analysts agree - use average
            final_analysis = {
                'final_recommendation': sentiment_analysis.get('recommendation', 'HOLD'),
                'final_confidence': (
                    sentiment_analysis.get('confidence', 50) + 
                    technical_analysis.get('confidence', 50)
                ) / 2 * 1.1,  # Boost for agreement
                'entry_price': technical_analysis.get('entry_price', current_price),
                'stop_loss': technical_analysis.get('stop_loss'),
                'take_profit': technical_analysis.get('take_profit'),
                'should_execute': True
            }

        # Build consensus result
        consensus = self._build_consensus(
            sentiment_analysis, technical_analysis, final_analysis,
            current_price, xgb_signal, xgb_confidence
        )

        return {
            'symbol': symbol,
            'market_type': market_type,
            'timeframe': timeframe,
            'current_price': current_price,
            'consensus': consensus,
            'dialogue': dialogue,
            'models_available': {'openai': True},
            'timestamp': datetime.utcnow().isoformat()
        }

    def _prepare_market_context(
        self, symbol, current_price, indicators, market_type, timeframe,
        xgb_signal, xgb_confidence, lstm_prediction
    ) -> str:
        """Prepare market context string for AI prompts."""
        context = f"""Symbol: {symbol}
Market Type: {market_type}
Timeframe: {timeframe}
Current Price: ${current_price:.4f}

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A'):.2f}
- MACD: {indicators.get('macd', 'N/A'):.4f}
- MACD Signal: {indicators.get('macd_signal', 'N/A'):.4f}
- EMA 20: {indicators.get('ema20', 'N/A'):.4f}
- EMA 50: {indicators.get('ema50', 'N/A'):.4f}
- EMA 200: {indicators.get('ema200', 'N/A'):.4f}
- ADX: {indicators.get('adx', 'N/A'):.2f}
- ATR: {indicators.get('atr', 'N/A'):.4f}
- 24h Change: {indicators.get('change_24h', 0):.2f}%"""

        if xgb_signal:
            context += f"\n\nML Model Prediction:\n- XGBoost Signal: {xgb_signal} ({xgb_confidence:.1f}% confidence)"
        
        if lstm_prediction:
            context += f"\n- LSTM Price Prediction: ${lstm_prediction:.4f}"

        return context

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from OpenAI response."""
        if not response:
            return {}
        
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return {}

    def _build_consensus(
        self, sentiment: Dict, technical: Dict, final: Dict,
        current_price: float, xgb_signal: str, xgb_confidence: float
    ) -> Dict:
        """Build final consensus object."""
        recommendation = final.get('final_recommendation', 'HOLD')
        confidence = min(100, max(0, final.get('final_confidence', 50)))
        
        # Apply XGBoost influence
        if xgb_signal and xgb_confidence:
            if xgb_signal == recommendation:
                confidence = min(100, confidence * 1.1)
            elif xgb_signal != 'HOLD' and recommendation != 'HOLD':
                confidence = confidence * 0.8
        
        # Determine if should execute
        should_execute = (
            recommendation != 'HOLD' and
            confidence >= 60 and
            final.get('should_execute', True)
        )
        
        return {
            'recommendation': recommendation,
            'confidence': round(confidence, 1),
            'entry_price': final.get('entry_price', current_price),
            'stop_loss': final.get('stop_loss'),
            'take_profit': final.get('take_profit'),
            'risk_level': final.get('risk_level', 'Medium'),
            'should_execute': should_execute,
            'sentiment_view': sentiment.get('recommendation', 'HOLD'),
            'technical_view': technical.get('recommendation', 'HOLD'),
            'xgb_signal': xgb_signal,
            'xgb_confidence': xgb_confidence,
            'models_agree': sentiment.get('recommendation') == technical.get('recommendation')
        }

    def _fallback_decision(
        self, symbol: str, indicators: Dict, 
        xgb_signal: str, xgb_confidence: float
    ) -> Dict:
        """Fallback decision when OpenAI is not available."""
        logger.warning("OpenAI not available, using fallback decision")
        
        recommendation = 'HOLD'
        confidence = 50
        
        # Use XGBoost if available
        if xgb_signal and xgb_confidence:
            recommendation = xgb_signal
            confidence = xgb_confidence
        else:
            # Simple RSI-based fallback
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                recommendation = 'BUY'
                confidence = 70
            elif rsi > 70:
                recommendation = 'SELL'
                confidence = 70
        
        return {
            'symbol': symbol,
            'consensus': {
                'recommendation': recommendation,
                'confidence': confidence,
                'should_execute': recommendation != 'HOLD' and confidence >= 65,
                'fallback': True
            },
            'dialogue': [],
            'models_available': {'openai': False}
        }


# Global instance
consensus_engine = OpenAIConsensusEngine()
