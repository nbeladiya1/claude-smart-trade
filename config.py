"""Configuration module for Smart Trade Bot."""
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
