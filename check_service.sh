#!/bin/bash
# Check and fix smart-trade service

echo "=== Checking service logs ==="
journalctl -u smart-trade -n 50 --no-pager

echo ""
echo "=== Checking error log ==="
cat /var/log/smart-trade/error.log 2>/dev/null || echo "No error log yet"

echo ""
echo "=== Testing Python script manually ==="
cd /opt/smart-trade
/opt/smart-trade/venv/bin/python -c "import smart_trade; print('Import successful')" 2>&1
