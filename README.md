# Claude Smart Trade Bot v2.0

Advanced automated trading bot for MetaTrader 5 with **Dual-LLM Decision System**.

## Features

### Dual-LLM Trading System
- **FinGPT Agent**: Sentiment analysis using financial news and market sentiment
- **FinLLaMA Agent**: Technical analysis using indicators and price patterns
- **Consensus Mechanism**: Both agents discuss and reach consensus before trading

### Money Management
- Risk-based position sizing (fixed, percent_risk, or Kelly criterion)
- Maximum daily loss protection
- Maximum drawdown protection
- Trailing stop loss
- Dynamic stop loss based on ATR

### Telegram Notifications
- Real-time trade alerts
- LLM agent discussions and reasoning
- Daily trading summaries
- Risk alerts and error notifications

### Supported LLM Providers
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- HuggingFace (FinGPT, FinLLaMA)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Smart Trade Bot v2.0                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌─────────────┐                       │
│  │   FinGPT    │     │  FinLLaMA   │                       │
│  │ (Sentiment) │     │ (Technical) │                       │
│  └──────┬──────┘     └──────┬──────┘                       │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   │                                         │
│         ┌─────────▼─────────┐                               │
│         │ Consensus Engine  │                               │
│         └─────────┬─────────┘                               │
│                   │                                         │
│         ┌─────────▼─────────┐                               │
│         │ Money Management  │                               │
│         └─────────┬─────────┘                               │
│                   │                                         │
│         ┌─────────▼─────────┐    ┌──────────────┐          │
│         │   Trade Engine    │───▶│   Telegram   │          │
│         └─────────┬─────────┘    │ Notifications │          │
│                   │              └──────────────┘          │
│         ┌─────────▼─────────┐                               │
│         │   MetaTrader 5    │                               │
│         └───────────────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Ubuntu 24.04 with MetaTrader 5 installed
- Python 3.10+
- MetaTrader 5 account credentials
- API keys (OpenAI/Anthropic/HuggingFace)
- Telegram bot token

### Quick Install on VPS

SSH into your VPS and run:

```bash
curl -sSL https://raw.githubusercontent.com/naushik007/claude-smart-trade/main/setup-vps.sh | bash
```

Or manually:

```bash
git clone https://github.com/naushik007/claude-smart-trade.git /opt/smart-trade
cd /opt/smart-trade
chmod +x deploy.sh
./deploy.sh
```

## Configuration

Copy the example environment file and edit with your credentials:

```bash
cp /opt/smart-trade/.env.example /opt/smart-trade/.env
nano /opt/smart-trade/.env
```

### Required Settings

```env
# MetaTrader 5
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server

# At least one LLM API key
OPENAI_API_KEY=sk-your-openai-api-key
# OR
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key

# Telegram (for notifications)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Trading Settings

```env
SYMBOL=EURUSD           # Trading pair
TIMEFRAME=H1            # Timeframe (M1, M5, M15, M30, H1, H4, D1)
RISK_PER_TRADE=2.0      # Risk % per trade
MAX_DAILY_LOSS=5.0      # Max daily loss %
MAX_OPEN_TRADES=3       # Maximum concurrent trades
```

### LLM Settings

```env
USE_DUAL_LLM=true              # Enable dual-LLM system
LLM_CONSENSUS_REQUIRED=true    # Require both agents to agree
LLM_PROVIDER=openai            # Primary provider: openai, anthropic, huggingface
```

## Service Commands

```bash
# Start the bot
systemctl start smart-trade

# Stop the bot
systemctl stop smart-trade

# Restart the bot
systemctl restart smart-trade

# Check status
systemctl status smart-trade

# View logs
tail -f /var/log/smart-trade/bot.log

# Disable autostart
systemctl disable smart-trade

# Enable autostart
systemctl enable smart-trade
```

## Telegram Commands

Once configured, you'll receive:
- **Signal notifications** with LLM reasoning
- **Trade opened/closed** alerts
- **Daily performance** summaries
- **Risk alerts** when limits are approached

## How It Works

### 1. Data Collection
The bot fetches market data from MetaTrader 5 and financial news from configured sources.

### 2. Dual-LLM Analysis

**FinGPT Agent** analyzes:
- Market sentiment from news
- Risk-on/risk-off environment
- Macroeconomic factors
- Institutional positioning

**FinLLaMA Agent** analyzes:
- Technical indicators (RSI, MACD, SMA, Bollinger Bands)
- Price patterns and trends
- Support/resistance levels
- Volume analysis

### 3. Consensus Decision
Both agents provide their signals and reasoning. The consensus engine:
- Compares signals from both agents
- Weighs confidence levels
- Determines if there's agreement
- Decides whether to trade

### 4. Risk Management
Before executing any trade:
- Checks daily loss limits
- Verifies maximum drawdown
- Calculates optimal position size
- Sets dynamic stop loss/take profit

### 5. Execution & Notification
- Executes trade on MetaTrader 5
- Sends detailed notification to Telegram
- Manages trailing stops
- Records trade for statistics

## Files Structure

```
/opt/smart-trade/
├── smart_trade.py       # Main bot entry point
├── llm_agents.py        # Dual-LLM agent system
├── money_management.py  # Risk and position sizing
├── telegram_bot.py      # Telegram notifications
├── config.py            # Configuration loader
├── requirements.txt     # Python dependencies
├── .env                 # Your configuration (create from .env.example)
├── .env.example         # Configuration template
├── smart-trade.service  # Systemd service file
├── setup-vps.sh         # One-click VPS setup
└── deploy.sh            # Manual deployment script
```

## Disclaimer

This trading bot is for educational purposes. Trading forex/CFDs carries significant risk. Only trade with money you can afford to lose. Past performance does not guarantee future results.

## Sources

This bot integrates concepts from:
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) - Open-Source Financial LLMs
- [Open-FinLLMs](https://arxiv.org/abs/2408.11878) - Financial LLM Research
- [OpenClaw.ai](https://openclaw.ai/) - Open-source AI assistant framework

## License

MIT
