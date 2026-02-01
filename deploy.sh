#!/bin/bash
# Deployment script for Claude Smart Trade Bot

set -e

APP_DIR="/opt/smart-trade"
LOG_DIR="/var/log/smart-trade"
SERVICE_NAME="smart-trade"

echo "=========================================="
echo "Claude Smart Trade Bot - Deployment Script"
echo "=========================================="

# Create directories
echo "[1/7] Creating directories..."
mkdir -p $APP_DIR
mkdir -p $LOG_DIR

# Copy application files
echo "[2/7] Copying application files..."
cp -r /tmp/smart-trade/* $APP_DIR/

# Create virtual environment
echo "[3/7] Creating Python virtual environment..."
cd $APP_DIR
python3 -m venv venv

# Install dependencies
echo "[4/7] Installing dependencies..."
$APP_DIR/venv/bin/pip install --upgrade pip
$APP_DIR/venv/bin/pip install -r requirements.txt

# Copy environment file if not exists
echo "[5/7] Setting up configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp $APP_DIR/.env.example $APP_DIR/.env
    echo "Please edit /opt/smart-trade/.env with your MT5 credentials"
fi

# Install systemd service
echo "[6/7] Installing systemd service..."
cp $APP_DIR/smart-trade.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME

# Set permissions
echo "[7/7] Setting permissions..."
chmod +x $APP_DIR/smart_trade.py
chown -R root:root $APP_DIR
chown -R root:root $LOG_DIR

echo "=========================================="
echo "Deployment complete!"
echo ""
echo "IMPORTANT - Configure these in /opt/smart-trade/.env:"
echo ""
echo "1. MetaTrader 5 credentials (MT5_LOGIN, MT5_PASSWORD, MT5_SERVER)"
echo "2. LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
echo "3. Telegram bot (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)"
echo ""
echo "Commands:"
echo "- Edit config:  nano /opt/smart-trade/.env"
echo "- Start bot:    systemctl start smart-trade"
echo "- Check status: systemctl status smart-trade"
echo "- View logs:    tail -f /var/log/smart-trade/bot.log"
echo "=========================================="
