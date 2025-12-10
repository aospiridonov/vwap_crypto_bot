"""
Bybit Exchange Connector using pybit
Handles connection to Bybit with Demo/Live trading modes and provides methods for:
- Fetching historical and latest candles with automatic precision handling
- Placing market orders with TP/SL (automatically rounded to qtyStep/tickSize)
- Querying positions and balance
- Closing positions
- Canceling orders
"""

import os
import time
import logging
import math
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import pandas as pd
from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """
    Decorator for retrying API calls with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    error_msg = str(e).lower()
                    last_exception = e

                    # Check if error is retryable
                    is_network_error = any(x in error_msg for x in ['network', 'connection', 'timeout'])
                    is_rate_limit = any(x in error_msg for x in ['rate limit', 'too many'])
                    is_unavailable = any(x in error_msg for x in ['unavailable', 'maintenance'])
                    is_ddos = 'ddos' in error_msg

                    # Don't retry on invalid params, insufficient balance, etc.
                    is_permanent_error = any(x in error_msg for x in [
                        'invalid', 'insufficient', 'balance', 'parameter', 'permission'
                    ])

                    if is_permanent_error:
                        logger.error(f"❌ Permanent error in {func.__name__}: {e}")
                        raise

                    if attempt < max_retries and (is_network_error or is_rate_limit or is_unavailable or is_ddos):
                        # Calculate delay
                        if is_rate_limit:
                            delay = min(base_delay * (2 ** (attempt + 2)), max_delay)
                        elif is_ddos:
                            delay = min(base_delay * (2 ** (attempt + 3)), max_delay)
                        else:
                            delay = min(base_delay * (2 ** attempt), max_delay)

                        error_type = 'Rate limit' if is_rate_limit else 'DDoS' if is_ddos else 'Network/Unavailable'
                        logger.warning(f"⚠️ {error_type} error in {func.__name__}, retry {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"❌ Error in {func.__name__} after {max_retries} retries: {e}")
                        raise

            # All retries exhausted
            raise last_exception

        return wrapper
    return decorator


class BybitConnector:
    """
    Wrapper for pybit Bybit exchange with retry logic and error handling
    """

    def __init__(self, config: Dict, demo: bool = True):
        """
        Initialize Bybit connector

        Args:
            config: Trading parameters from trading_params.json
            demo: Use demo trading mode (True) or real trading (False)
        """
        self.config = config
        self.demo = demo
        self.symbol = config['symbol']
        self.leverage = config['leverage']

        # Instrument precision (will be fetched from API)
        self.qty_step = None
        self.tick_size = None

        # Get API keys from environment (different keys for demo vs live)
        if demo:
            api_key = os.getenv('BYBIT_DEMO_API_KEY')
            api_secret = os.getenv('BYBIT_DEMO_SECRET')
            if not api_key or not api_secret:
                raise ValueError(f"Missing Bybit Demo API credentials. Set BYBIT_DEMO_API_KEY and BYBIT_DEMO_SECRET in .env")
        else:
            api_key = os.getenv('BYBIT_LIVE_API_KEY')
            api_secret = os.getenv('BYBIT_LIVE_SECRET')
            if not api_key or not api_secret:
                raise ValueError(f"Missing Bybit Live API credentials. Set BYBIT_LIVE_API_KEY and BYBIT_LIVE_SECRET in .env")

        # Initialize HTTP session (always mainnet, demo flag controls virtual/real money)
        # recv_window=60000 (60 seconds) - very large window to handle clock skew
        # Bybit rejects requests where local time is ahead of server time even within recv_window
        self.session = HTTP(
            testnet=False,  # Always mainnet
            api_key=api_key,
            api_secret=api_secret,
            demo=demo,
            recv_window=60000
        )

        # Sync time offset with Bybit server
        self._sync_time_offset()

        if demo:
            logger.info("🎮 Bybit Demo Trading mode (virtual money)")
        else:
            logger.info("🚀 Bybit LIVE Trading mode (REAL MONEY!)")

        # Note: Leverage will be set automatically when placing first order
        # No need to set it explicitly in advance
        logger.info(f"📊 Leverage configured: {self.leverage}x (will apply on order placement)")

        # Fetch instrument info for precision
        self._fetch_instrument_info()

    def _sync_time_offset(self):
        """Sync time offset with Bybit server"""
        try:
            import time
            response = self.session.get_server_time()
            if response['retCode'] == 0:
                server_time_ms = int(response['result']['timeNano']) // 1000000
                local_time_ms = int(time.time() * 1000)
                offset_ms = server_time_ms - local_time_ms

                # Store offset for future use (pybit will handle it internally)
                logger.info(f"⏰ Time offset: {offset_ms}ms (server ahead)" if offset_ms > 0 else f"⏰ Time offset: {abs(offset_ms)}ms (local ahead)")

                # pybit will automatically adjust after first retry
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync time offset: {e} (will use retry mechanism)")

    def _fetch_instrument_info(self):
        """Fetch instrument info to get qtyStep and tickSize"""
        try:
            response = self.session.get_instruments_info(
                category="linear",
                symbol=self.symbol
            )

            if response['retCode'] != 0:
                logger.warning(f"⚠️ Failed to fetch instrument info: {response.get('retMsg')}")
                # Use defaults for BTCUSDT
                self.qty_step = 0.001
                self.tick_size = 0.01
                return

            instruments = response['result']['list']
            if instruments and len(instruments) > 0:
                instrument = instruments[0]
                lot_size_filter = instrument.get('lotSizeFilter', {})
                price_filter = instrument.get('priceFilter', {})

                self.qty_step = float(lot_size_filter.get('qtyStep', 0.001))
                self.tick_size = float(price_filter.get('tickSize', 0.01))

                logger.info(f"📊 Instrument info: qtyStep={self.qty_step}, tickSize={self.tick_size}")
            else:
                # Defaults
                self.qty_step = 0.001
                self.tick_size = 0.01
                logger.warning(f"⚠️ No instrument info found, using defaults")

        except Exception as e:
            logger.warning(f"⚠️ Error fetching instrument info: {e}, using defaults")
            self.qty_step = 0.001
            self.tick_size = 0.01

    def _round_qty(self, qty: float) -> float:
        """Round quantity DOWN to qtyStep precision to avoid exceeding available margin"""
        if self.qty_step is None:
            return round(qty, 3)  # Fallback

        # Round DOWN (floor) to ensure we don't exceed available balance
        # Example: qty=0.0762 with qtyStep=0.001 -> floor(0.0762/0.001)*0.001 = 0.076
        return round(math.floor(qty / self.qty_step) * self.qty_step, 10)

    def _round_price(self, price: float) -> float:
        """Round price to tickSize precision"""
        if self.tick_size is None:
            return round(price, 2)  # Fallback

        # Round to nearest tick and clean floating point errors
        return round(round(price / self.tick_size) * self.tick_size, 10)

    @retry_on_error(max_retries=3, base_delay=1.0)
    def get_leverage(self) -> float:
        """
        Get current leverage setting for the symbol

        Returns:
            Current leverage as float (e.g., 3.0)
        """
        response = self.session.get_positions(
            category="linear",
            symbol=self.symbol
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch leverage: {response.get('retMsg', 'Unknown error')}")

        positions = response['result']['list']
        if positions and len(positions) > 0:
            # Get leverage from first position (Buy or Sell side)
            current_leverage = float(positions[0].get('leverage', self.leverage))
            logger.debug(f"📊 Current leverage: {current_leverage}x")
            return current_leverage
        else:
            # No positions yet, return configured leverage
            logger.debug(f"📊 No positions found, using configured leverage: {self.leverage}x")
            return self.leverage

    def fetch_balance(self) -> Dict:
        """
        Fetch USDT balance

        Returns:
            Dict with 'free', 'used', 'total' balance
        """
        response = self.session.get_wallet_balance(
            accountType="UNIFIED",
            coin="USDT"
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch balance: {response.get('retMsg', 'Unknown error')}")

        # Parse balance - find USDT coin
        usdt_coin = None
        if 'list' in response['result'] and len(response['result']['list']) > 0:
            for coin in response['result']['list'][0].get('coin', []):
                if coin.get('coin') == 'USDT':
                    usdt_coin = coin
                    break

        if not usdt_coin:
            raise Exception("USDT not found in wallet")

        # Handle empty strings in response
        # For Unified Trading Account (UTA):
        # - totalAvailableBalance (account-level): Available for trading
        # - totalPositionIM (coin-level): Total initial margin locked in positions
        # - walletBalance (coin-level): Total wallet balance
        account = response['result']['list'][0]
        free_str = account.get('totalAvailableBalance', '0')  # Account-level available balance (USD)
        used_str = usdt_coin.get('totalPositionIM', '0')      # Margin in positions
        total_str = usdt_coin.get('walletBalance', '0')       # Total balance

        usdt_balance = {
            'free': float(free_str) if free_str else 0.0,
            'used': float(used_str) if used_str else 0.0,
            'total': float(total_str) if total_str else 0.0
        }

        logger.info(f"💰 Balance: Free=${usdt_balance['free']:.2f}, Used=${usdt_balance['used']:.2f}, Total=${usdt_balance['total']:.2f}")
        return usdt_balance

    @retry_on_error(max_retries=3, base_delay=1.0)
    def fetch_ohlcv(self, since: Optional[datetime] = None, limit: int = 500, interval: str = "60") -> pd.DataFrame:
        """
        Fetch historical OHLCV data

        Args:
            since: Start datetime (default: limit intervals ago)
            limit: Number of candles (default: 500, max: 200 per request for Bybit)
            interval: Candle interval - "60" for 1H, "240" for 4H (default: "60")

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Calculate hours per candle based on interval
        hours_per_candle = int(interval) // 60

        if since is None:
            since = datetime.utcnow() - timedelta(hours=limit * hours_per_candle)

        since_ms = int(since.timestamp() * 1000)

        # Bybit limit is 200 per request, so we need to fetch in batches
        all_candles = []
        current_since = since_ms

        while len(all_candles) < limit:
            batch_limit = min(200, limit - len(all_candles))

            response = self.session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=interval,
                start=current_since,
                limit=batch_limit
            )

            if response['retCode'] != 0:
                raise Exception(f"Failed to fetch OHLCV: {response.get('retMsg', 'Unknown error')}")

            candles = response['result']['list']

            if not candles:
                break

            # Bybit returns newest first, reverse it
            candles.reverse()
            all_candles.extend(candles)

            # Get timestamp of last candle
            current_since = int(candles[-1][0]) + (hours_per_candle * 3600000)  # +interval in ms

            if len(candles) < batch_limit:
                break

        # Convert to DataFrame
        df_data = []
        for candle in all_candles:
            df_data.append({
                'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })

        df = pd.DataFrame(df_data)
        df = df.set_index('timestamp').sort_index()

        logger.info(f"📊 Fetched {len(df)} candles ({interval}min interval) from {df.index.min()} to {df.index.max()}")
        return df

    @retry_on_error(max_retries=3, base_delay=1.0)
    def fetch_ohlcv_4h(self, since: Optional[datetime] = None, limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical 4H OHLCV data (convenience method for V38 strategy)

        Args:
            since: Start datetime (default: limit*4 hours ago)
            limit: Number of 4H candles (default: 100)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        return self.fetch_ohlcv(since=since, limit=limit, interval="240")

    @retry_on_error(max_retries=3, base_delay=1.0)
    def fetch_latest_candle(self, interval: str = "60") -> Dict:
        """
        Fetch the most recent closed candle

        Args:
            interval: Candle interval - "60" for 1H, "240" for 4H (default: "60")

        Returns:
            Dict with OHLCV data for latest candle
        """
        response = self.session.get_kline(
            category="linear",
            symbol=self.symbol,
            interval=interval,
            limit=2
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch latest candle: {response.get('retMsg', 'Unknown error')}")

        candles = response['result']['list']

        # Bybit returns newest first, so get second one (last closed candle)
        candle = candles[1] if len(candles) > 1 else candles[0]

        result = {
            'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4]),
            'volume': float(candle[5])
        }

        logger.info(f"📊 Latest {interval}min candle: {result['timestamp']} | Close=${result['close']:.2f}")
        return result

    @retry_on_error(max_retries=3, base_delay=1.0)
    def fetch_latest_4h_candle(self) -> Dict:
        """
        Fetch the most recent closed 4H candle (convenience method for V38 strategy)

        Returns:
            Dict with OHLCV data for latest 4H candle
        """
        return self.fetch_latest_candle(interval="240")

    @retry_on_error(max_retries=3, base_delay=1.0)
    def get_current_price(self) -> float:
        """Get current BTC price"""
        response = self.session.get_tickers(
            category="linear",
            symbol=self.symbol
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch ticker: {response.get('retMsg', 'Unknown error')}")

        price = float(response['result']['list'][0]['lastPrice'])
        logger.debug(f"💵 Current price: ${price:.2f}")
        return price

    @retry_on_error(max_retries=2, base_delay=2.0)
    def place_market_order(
        self,
        side: str,
        amount_usdt: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        max_slippage_pct: float = 0.5
    ) -> Dict:
        """
        Place market order with TP/SL and slippage protection

        Args:
            side: 'Buy' for LONG, 'Sell' for SHORT
            amount_usdt: Position size in USDT
            tp_price: Take profit price (optional)
            sl_price: Stop loss price (optional)
            max_slippage_pct: Maximum allowed slippage in % (default: 0.5%)

        Returns:
            Order info dict

        Raises:
            ValueError: If slippage exceeds max_slippage_pct
        """
        # Get expected price before order
        expected_price = self.get_current_price()

        # Calculate quantity (amount in base currency - BTC)
        # Round according to instrument's qtyStep
        qty = self._round_qty(amount_usdt / expected_price)

        # Validate TP/SL prices
        if tp_price and sl_price:
            if side == 'Buy':  # LONG position
                if tp_price <= expected_price:
                    raise ValueError(f"TP price ${tp_price:.2f} must be > entry price ${expected_price:.2f} for LONG")
                if sl_price >= expected_price:
                    raise ValueError(f"SL price ${sl_price:.2f} must be < entry price ${expected_price:.2f} for LONG")
            else:  # SHORT position
                if tp_price >= expected_price:
                    raise ValueError(f"TP price ${tp_price:.2f} must be < entry price ${expected_price:.2f} for SHORT")
                if sl_price <= expected_price:
                    raise ValueError(f"SL price ${sl_price:.2f} must be > entry price ${expected_price:.2f} for SHORT")

        # Round TP/SL prices according to instrument's tickSize
        tp_price_rounded = self._round_price(tp_price) if tp_price else None
        sl_price_rounded = self._round_price(sl_price) if sl_price else None

        # Place market order
        response = self.session.place_order(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            takeProfit=str(tp_price_rounded) if tp_price_rounded else None,
            stopLoss=str(sl_price_rounded) if sl_price_rounded else None,
            tpslMode="Full",  # TP/SL applies to full position
            tpOrderType="Market",  # TP triggers market order
            slOrderType="Market",  # SL triggers market order
            tpTriggerBy="MarkPrice",  # Use mark price to avoid manipulation
            slTriggerBy="MarkPrice"
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to place order: {response.get('retMsg', 'Unknown error')}")

        order_id = response['result']['orderId']

        # Wait for order to execute (market orders are usually instant)
        time.sleep(1)

        # Get order details to check execution
        order_info = self._get_order_details(order_id)

        # Check order status (but don't fail if empty - WebSocket will confirm)
        order_status = order_info.get('orderStatus', '')
        if order_status != 'Filled':
            logger.warning(f"⚠️ Order status is '{order_status}', expected 'Filled'. Waiting...")
            # Try one more time after delay
            time.sleep(2)
            order_info = self._get_order_details(order_id)
            order_status = order_info.get('orderStatus', '')

        # Log status but don't fail - WebSocket confirmation is more reliable
        if order_status == 'Filled':
            logger.info(f"✅ Order confirmed via REST API: Status={order_status}")
        else:
            logger.warning(f"⚠️ REST API status unclear ('{order_status}'), relying on WebSocket confirmation")

        # Get executed price
        executed_price = order_info.get('avgPrice')
        if executed_price:
            executed_price = float(executed_price)
        else:
            logger.warning(f"⚠️ No avgPrice in order info, using expected price")
            executed_price = expected_price

        # Calculate and log slippage
        if side == 'Buy':
            slippage_pct = ((executed_price - expected_price) / expected_price) * 100
        else:  # Sell
            slippage_pct = ((expected_price - executed_price) / expected_price) * 100

        if slippage_pct > max_slippage_pct:
            logger.warning(
                f"⚠️ High slippage detected: {slippage_pct:.2f}% "
                f"(expected: ${expected_price:.2f}, executed: ${executed_price:.2f})"
            )

        # Verify TP/SL were set by checking position
        position_info = self.get_position_with_tpsl(side)
        actual_tp = position_info.get('take_profit') if position_info else None
        actual_sl = position_info.get('stop_loss') if position_info else None

        if tp_price and (not actual_tp or actual_tp == 0):
            logger.error(f"❌ TP was not set! Expected: ${tp_price:.2f}, Actual: {actual_tp}")
        if sl_price and (not actual_sl or actual_sl == 0):
            logger.error(f"❌ SL was not set! Expected: ${sl_price:.2f}, Actual: {actual_sl}")

        logger.info(
            f"✅ {'LONG' if side == 'Buy' else 'SHORT'} order FILLED: "
            f"${amount_usdt:.2f} @ ${executed_price:.2f} | "
            f"TP=${actual_tp or tp_price:.2f} | SL=${actual_sl or sl_price:.2f}"
        )

        return {
            'orderId': order_id,
            'symbol': self.symbol,
            'side': side,
            'price': executed_price,
            'qty': float(order_info.get('cumExecQty', qty)),
            'status': order_status,
            'actual_tp': actual_tp,
            'actual_sl': actual_sl,
            'orderInfo': order_info
        }

    def _get_order_details(self, order_id: str) -> Dict:
        """Get order details by ID"""
        response = self.session.get_order_history(
            category="linear",
            symbol=self.symbol,
            orderId=order_id
        )

        if response['retCode'] != 0:
            logger.warning(f"⚠️ Failed to fetch order details: {response.get('retMsg', 'Unknown error')}")
            return {}

        orders = response['result']['list']
        return orders[0] if orders else {}

    @retry_on_error(max_retries=2, base_delay=2.0)
    def close_position(self, side: str) -> Dict:
        """
        Close position by placing opposite market order

        Args:
            side: 'long' or 'short'

        Returns:
            Order info dict
        """
        # Get current position size
        positions = self.get_open_positions()

        # Convert side: 'long' -> 'Buy', 'short' -> 'Sell'
        position_side = 'Buy' if side.lower() == 'long' else 'Sell'
        position = next((p for p in positions if p['side'] == position_side), None)

        if not position:
            logger.warning(f"⚠️ No open {side} position to close")
            return None

        # Close position with opposite order (reduceOnly)
        close_side = 'Sell' if side.lower() == 'long' else 'Buy'
        qty = abs(float(position['size']))

        response = self.session.place_order(
            category="linear",
            symbol=self.symbol,
            side=close_side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to close position: {response.get('retMsg', 'Unknown error')}")

        logger.info(f"✅ Position closed: {side.upper()} | Qty: {qty:.6f} BTC")
        return response['result']

    @retry_on_error(max_retries=3, base_delay=1.0)
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions

        Returns:
            List of position dicts
        """
        response = self.session.get_positions(
            category="linear",
            symbol=self.symbol
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch positions: {response.get('retMsg', 'Unknown error')}")

        positions = response['result']['list']

        # Filter only positions with size > 0
        open_positions = [p for p in positions if float(p['size']) != 0]

        logger.info(f"📊 Open positions: {len(open_positions)}")
        return open_positions

    @retry_on_error(max_retries=3, base_delay=1.0)
    def get_position_with_tpsl(self, side: str) -> Optional[Dict]:
        """
        Get position details including TP/SL prices

        Args:
            side: 'Buy' for LONG, 'Sell' for SHORT

        Returns:
            Dict with position info including takeProfit and stopLoss, or None
        """
        response = self.session.get_positions(
            category="linear",
            symbol=self.symbol
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to fetch positions: {response.get('retMsg', 'Unknown error')}")

        positions = response['result']['list']

        for pos in positions:
            if pos['side'] == side and float(pos['size']) != 0:
                return {
                    'side': pos['side'],
                    'size': float(pos['size']),
                    'entry_price': float(pos['avgPrice']) if pos['avgPrice'] else 0,
                    'take_profit': float(pos['takeProfit']) if pos['takeProfit'] else None,
                    'stop_loss': float(pos['stopLoss']) if pos['stopLoss'] else None,
                    'leverage': float(pos['leverage']) if pos['leverage'] else self.leverage,
                    'unrealised_pnl': float(pos['unrealisedPnl']) if pos['unrealisedPnl'] else 0,
                    'position_value': float(pos['positionValue']) if pos['positionValue'] else 0
                }

        return None

    def get_position_open_time(self, side: str) -> Optional[datetime]:
        """
        Get the time when current position was opened by looking at order history.

        Bybit's position.createdTime is unreliable (persists across position cycles).
        Instead, we find the last filled order that opened the position.

        Args:
            side: 'Buy' for LONG, 'Sell' for SHORT

        Returns:
            datetime of position open, or None if not found
        """
        try:
            # Get recent filled orders for this side
            response = self.session.get_order_history(
                category="linear",
                symbol=self.symbol,
                orderStatus="Filled",
                limit=10
            )

            if response['retCode'] != 0:
                logger.warning(f"⚠️ Failed to fetch order history: {response.get('retMsg')}")
                return None

            orders = response['result']['list']

            # Find the most recent filled order matching the side
            for order in orders:
                if order['side'] == side and order['orderStatus'] == 'Filled':
                    created_time_ms = int(order['createdTime'])
                    open_time = datetime.fromtimestamp(created_time_ms / 1000)
                    logger.debug(f"Found position open time from order history: {open_time}")
                    return open_time

            logger.warning(f"⚠️ No matching filled order found for {side}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting position open time: {e}")
            return None

    @retry_on_error(max_retries=3, base_delay=1.0)
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders for the symbol

        Returns:
            Number of canceled orders
        """
        response = self.session.cancel_all_orders(
            category="linear",
            symbol=self.symbol
        )

        if response['retCode'] != 0:
            raise Exception(f"Failed to cancel orders: {response.get('retMsg', 'Unknown error')}")

        # Get list of canceled orders
        canceled = response['result']['list']
        count = len(canceled) if canceled else 0

        logger.info(f"🗑️ Canceled {count} orders")
        return count

    def test_connection(self) -> bool:
        """Test if connection works"""
        try:
            self.fetch_balance()
            self.get_current_price()
            logger.info("✅ Bybit connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Bybit connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Test connection
    logging.basicConfig(level=logging.INFO)

    config = {
        'symbol': 'BTCUSDT',
        'leverage': 3.0
    }

    try:
        connector = BybitConnector(config, demo=True)
        connector.test_connection()

        # Test fetching data
        df = connector.fetch_ohlcv(limit=10)
        print(df.tail())

        # Test balance
        balance = connector.fetch_balance()
        print(f"Balance: {balance}")

    except Exception as e:
        print(f"Error: {e}")
