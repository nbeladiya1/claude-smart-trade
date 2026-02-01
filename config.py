"""Configuration module for Smart Trade Bot with Dual-LLM System."""
import os
from dotenv import load_dotenv

load_dotenv()

# MetaTrader 5 Configuration
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')

# Trading Configuration
SYMBOL = os.getenv('SYMBOL', 'EURUSD')
TIMEFRAME = os.getenv('TIMEFRAME', 'H1')
LOT_SIZE = float(os.getenv('LOT_SIZE', '0.01'))
STOP_LOSS = int(os.getenv('STOP_LOSS', '50'))
TAKE_PROFIT = int(os.getenv('TAKE_PROFIT', '100'))

# Bot Settings
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '60'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ===========================================
# LLM Configuration (Dual-Agent System)
# ===========================================

# OpenAI API (for FinGPT-style sentiment analysis)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')

# Anthropic API (for technical analysis agent)
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')

# HuggingFace (for FinGPT/FinLLaMA models)
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')
FINGPT_MODEL = os.getenv('FINGPT_MODEL', 'FinGPT/fingpt-forecaster_sz50_llama2-7b_lora')
FINLLAMA_MODEL = os.getenv('FINLLAMA_MODEL', 'TheFinAI/FinLLaMA')

# LLM Agent Settings
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # openai, anthropic, huggingface, local
USE_DUAL_LLM = os.getenv('USE_DUAL_LLM', 'true').lower() == 'true'
LLM_CONSENSUS_REQUIRED = os.getenv('LLM_CONSENSUS_REQUIRED', 'true').lower() == 'true'
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))

# ===========================================
# Telegram Bot Configuration
# ===========================================
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'

# Notification Settings
NOTIFY_ON_SIGNAL = os.getenv('NOTIFY_ON_SIGNAL', 'true').lower() == 'true'
NOTIFY_ON_TRADE = os.getenv('NOTIFY_ON_TRADE', 'true').lower() == 'true'
NOTIFY_ON_ERROR = os.getenv('NOTIFY_ON_ERROR', 'true').lower() == 'true'
NOTIFY_DAILY_SUMMARY = os.getenv('NOTIFY_DAILY_SUMMARY', 'true').lower() == 'true'

# ===========================================
# Money Management Configuration
# ===========================================
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '2.0'))  # % of account balance
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '5.0'))  # % of account balance
MAX_OPEN_TRADES = int(os.getenv('MAX_OPEN_TRADES', '3'))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '10.0'))  # % max drawdown before stopping
USE_TRAILING_STOP = os.getenv('USE_TRAILING_STOP', 'true').lower() == 'true'
TRAILING_STOP_PIPS = int(os.getenv('TRAILING_STOP_PIPS', '30'))

# Position Sizing Method: fixed, percent_risk, kelly
POSITION_SIZING_METHOD = os.getenv('POSITION_SIZING_METHOD', 'percent_risk')

# ===========================================
# News & Sentiment Sources
# ===========================================
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
