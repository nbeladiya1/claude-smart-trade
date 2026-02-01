#!/bin/bash
# One-click VPS Setup Script for Claude Smart Trade Bot
# Run this on your Hostinger VPS terminal

set -e

APP_DIR="/opt/smart-trade"
LOG_DIR="/var/log/smart-trade"
REPO_URL="https://github.com/naushik007/claude-smart-trade.git"

echo "=========================================="
echo "Claude Smart Trade Bot - VPS Setup"
echo "=========================================="

# Update system
echo "[1/9] Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install required packages
echo "[2/9] Installing required packages..."
apt-get install -y python3 python3-pip python3-venv git curl

# Create directories
echo "[3/9] Creating application directories..."
mkdir -p $APP_DIR
mkdir -p $LOG_DIR

# Clone repository
echo "[4/9] Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
    cd $APP_DIR && git pull
else
    rm -rf $APP_DIR/*
    git clone $REPO_URL $APP_DIR
fi

cd $APP_DIR

# Create virtual environment
echo "[5/9] Creating Python virtual environment..."
python3 -m venv venv

# Install dependencies
echo "[6/9] Installing Python dependencies..."
$APP_DIR/venv/bin/pip install --upgrade pip
$APP_DIR/venv/bin/pip install -r requirements.txt

# Setup configuration
echo "[7/9] Setting up configuration..."
if [ ! -f "$APP_DIR/.env" ]; then
    cp $APP_DIR/.env.example $APP_DIR/.env
    echo ""
    echo "!!! IMPORTANT: Edit /opt/smart-trade/.env with your MT5 credentials !!!"
    echo ""
fi

# Install and enable systemd service
echo "[8/9] Installing systemd service..."
cp $APP_DIR/smart-trade.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable smart-trade

# Set permissions
echo "[9/9] Setting permissions..."
chmod +x $APP_DIR/smart_trade.py
chown -R root:root $APP_DIR
chown -R root:root $LOG_DIR

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Edit your MT5 credentials:"
echo "   nano /opt/smart-trade/.env"
echo ""
echo "2. Start the trading bot:"
echo "   systemctl start smart-trade"
echo ""
echo "3. Check status:"
echo "   systemctl status smart-trade"
echo ""
echo "4. View logs:"
echo "   tail -f /var/log/smart-trade/bot.log"
echo ""
echo "The bot will automatically start on system reboot."
echo "=========================================="
