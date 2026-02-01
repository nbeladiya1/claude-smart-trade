"""
Binance Trading Service
Supports Spot trading (Binance.US compatible)
"""
import hashlib
import hmac
import time
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlencode
import os

logger = logging.getLogger(__name__)


class BinanceService:
    """Binance API integration for crypto trading"""

    def __init__(self):
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY', '')
        self.testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
        self.is_binance_us = os.getenv('BINANCE_US', 'true').lower() == 'true'
        
        # Set base URL
        if self.is_binance_us:
            self.base_url = 'https://api.binance.us'
        elif self.testnet:
            self.base_url = 'https://testnet.binance.vision'
        else:
            self.base_url = 'https://api.binance.com'
        
        self.is_available = bool(self.api_key and self.secret_key)

    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)

    def _make_request(
        self,
        method: str,
        url: str,
        params: Dict = None,
        signed: bool = False
    ) -> Dict:
        """Make HTTP request to Binance API"""
        headers = {'X-MBX-APIKEY': self.api_key}
        params = params or {}

        if signed:
            params['timestamp'] = self._get_timestamp()
            params['signature'] = self._generate_signature(params)

        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, timeout=30)
            else:
                return {'success': False, 'error': f'Invalid method: {method}'}

            data = response.json()

            if response.status_code != 200:
                return {
                    'success': False,
                    'error': data.get('msg', 'Unknown error'),
                    'code': data.get('code')
                }

            return {'success': True, 'data': data}

        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Request timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ===========================================
    # MARKET DATA
    # ===========================================

    def get_spot_price(self, symbol: str) -> Dict:
        """Get current spot price for a symbol"""
        url = f"{self.base_url}/api/v3/ticker/price"
        result = self._make_request('GET', url, {'symbol': symbol})

        if result['success']:
            return {
                'success': True,
                'symbol': symbol,
                'price': float(result['data']['price']),
                'timestamp': datetime.utcnow().isoformat()
            }
        return result

    def get_spot_prices(self, symbols: List[str] = None) -> Dict:
        """Get current prices for multiple symbols"""
        url = f"{self.base_url}/api/v3/ticker/price"
        result = self._make_request('GET', url)

        if result['success']:
            prices = {}
            for item in result['data']:
                if symbols is None or item['symbol'] in symbols:
                    prices[item['symbol']] = float(item['price'])
            return {'success': True, 'prices': prices}
        return result

    def get_spot_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 500
    ) -> List[Dict]:
        """Get candlestick/kline data"""
        url = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        result = self._make_request('GET', url, params)

        if result['success']:
            candles = []
            for k in result['data']:
                candles.append({
                    'timestamp': datetime.fromtimestamp(k[0] / 1000).isoformat(),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': datetime.fromtimestamp(k[6] / 1000).isoformat(),
                    'quote_volume': float(k[7]),
                    'trades': int(k[8])
                })
            return candles
        return []

    def get_24hr_ticker(self, symbol: str) -> Dict:
        """Get 24hr ticker statistics"""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        result = self._make_request('GET', url, {'symbol': symbol})
        
        if result['success']:
            data = result['data']
            return {
                'success': True,
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'change_24h': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'quote_volume_24h': float(data['quoteVolume'])
            }
        return result

    # ===========================================
    # ACCOUNT DATA
    # ===========================================

    def get_spot_balance(self) -> Dict:
        """Get spot account balance"""
        url = f"{self.base_url}/api/v3/account"
        result = self._make_request('GET', url, signed=True)

        if result['success']:
            balances = {}
            for asset in result['data'].get('balances', []):
                free = float(asset['free'])
                locked = float(asset['locked'])
                if free > 0 or locked > 0:
                    balances[asset['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    }
            return {'success': True, 'balances': balances}
        return result

    def get_account_info(self) -> Dict:
        """Get account information"""
        url = f"{self.base_url}/api/v3/account"
        result = self._make_request('GET', url, signed=True)
        
        if result['success']:
            data = result['data']
            return {
                'success': True,
                'can_trade': data.get('canTrade', False),
                'can_withdraw': data.get('canWithdraw', False),
                'account_type': data.get('accountType', 'SPOT'),
                'balances': {
                    asset['asset']: float(asset['free']) + float(asset['locked'])
                    for asset in data.get('balances', [])
                    if float(asset['free']) + float(asset['locked']) > 0
                }
            }
        return result

    # ===========================================
    # SPOT TRADING
    # ===========================================

    def spot_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float = None,
        quote_quantity: float = None
    ) -> Dict:
        """Execute spot market order"""
        url = f"{self.base_url}/api/v3/order"
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET'
        }

        if quantity:
            params['quantity'] = quantity
        elif quote_quantity:
            params['quoteOrderQty'] = quote_quantity
        else:
            return {'success': False, 'error': 'Must specify quantity or quote_quantity'}

        result = self._make_request('POST', url, params, signed=True)

        if result['success']:
            order = result['data']
            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'status': order['status'],
                'executed_qty': float(order['executedQty']),
                'price': float(order['fills'][0]['price']) if order.get('fills') else None,
                'timestamp': datetime.utcnow().isoformat()
            }
        return result

    def spot_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> Dict:
        """Execute spot limit order"""
        url = f"{self.base_url}/api/v3/order"
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': quantity,
            'price': price
        }

        result = self._make_request('POST', url, params, signed=True)

        if result['success']:
            order = result['data']
            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'status': order['status'],
                'quantity': float(order['origQty']),
                'price': float(order['price']),
                'timestamp': datetime.utcnow().isoformat()
            }
        return result

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order"""
        url = f"{self.base_url}/api/v3/order"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('DELETE', url, params, signed=True)

    def get_open_orders(self, symbol: str = None) -> Dict:
        """Get open orders"""
        url = f"{self.base_url}/api/v3/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', url, params, signed=True)

    def get_order_status(self, symbol: str, order_id: int) -> Dict:
        """Get order status"""
        url = f"{self.base_url}/api/v3/order"
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('GET', url, params, signed=True)

    def get_my_trades(self, symbol: str, limit: int = 50) -> Dict:
        """Get recent trades"""
        url = f"{self.base_url}/api/v3/myTrades"
        params = {
            'symbol': symbol,
            'limit': limit
        }
        result = self._make_request('GET', url, params, signed=True)
        
        if result['success']:
            trades = []
            for t in result['data']:
                trades.append({
                    'id': t['id'],
                    'order_id': t['orderId'],
                    'symbol': t['symbol'],
                    'side': 'BUY' if t['isBuyer'] else 'SELL',
                    'price': float(t['price']),
                    'quantity': float(t['qty']),
                    'commission': float(t['commission']),
                    'commission_asset': t['commissionAsset'],
                    'timestamp': datetime.fromtimestamp(t['time'] / 1000).isoformat()
                })
            return {'success': True, 'trades': trades}
        return result

    # ===========================================
    # SYMBOL INFO
    # ===========================================

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading rules for a symbol"""
        url = f"{self.base_url}/api/v3/exchangeInfo"
        params = {'symbol': symbol}
        result = self._make_request('GET', url, params)
        
        if result['success']:
            for s in result['data'].get('symbols', []):
                if s['symbol'] == symbol:
                    filters = {f['filterType']: f for f in s.get('filters', [])}
                    lot_size = filters.get('LOT_SIZE', {})
                    price_filter = filters.get('PRICE_FILTER', {})
                    
                    return {
                        'success': True,
                        'symbol': symbol,
                        'base_asset': s['baseAsset'],
                        'quote_asset': s['quoteAsset'],
                        'status': s['status'],
                        'min_qty': float(lot_size.get('minQty', 0)),
                        'max_qty': float(lot_size.get('maxQty', 0)),
                        'step_size': float(lot_size.get('stepSize', 0)),
                        'min_price': float(price_filter.get('minPrice', 0)),
                        'tick_size': float(price_filter.get('tickSize', 0))
                    }
            return {'success': False, 'error': f'Symbol {symbol} not found'}
        return result

    def check_connection(self) -> bool:
        """Check if API connection is working"""
        url = f"{self.base_url}/api/v3/ping"
        result = self._make_request('GET', url)
        return result['success']


# Global instance
binance = BinanceService()
