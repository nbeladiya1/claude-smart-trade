# Claude Smart Trade Bot

Automated trading bot for MetaTrader 5.

## Features

- Connects to MetaTrader 5 terminal
- SMA crossover strategy with RSI filter
- Automatic order execution with Stop Loss and Take Profit
- Runs as a systemd service with auto-restart
- Logging to file and console

## Installation

### Prerequisites

- Ubuntu 24.04 with MetaTrader 5 installed
- Python 3.10+
- MetaTrader 5 account credentials

### Deploy to VPS

1. Copy files to VPS:
```bash
scp -r ./* root@your-vps-ip:/tmp/smart-trade/
```

2. SSH into VPS and run deployment:
```bash
ssh root@your-vps-ip
cd /tmp/smart-trade
chmod +x deploy.sh
./deploy.sh
```

3. Configure your MT5 credentials:
```bash
nano /opt/smart-trade/.env
```

4. Start the service:
```bash
systemctl start smart-trade
```

## Configuration

Edit `/opt/smart-trade/.env` file:

```env
# MetaTrader 5 Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server

# Trading Configuration
SYMBOL=EURUSD
TIMEFRAME=H1
LOT_SIZE=0.01
STOP_LOSS=50
TAKE_PROFIT=100

# Bot Settings
CHECK_INTERVAL=60
LOG_LEVEL=INFO
```

## Service Commands

```bash
# Start bot
systemctl start smart-trade

# Stop bot
systemctl stop smart-trade

# Restart bot
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

## License

MIT
