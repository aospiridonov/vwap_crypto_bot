"""
Telegram Command Handler
Handles incoming Telegram commands like /balance and /stats
"""

import os
import logging
import requests
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class TelegramCommandHandler:
    """
    Handles incoming Telegram commands via long polling
    """

    def __init__(
        self,
        exchange_connector,
        position_tracker,
        trade_logger=None,
        data_manager=None,
        enabled: bool = True
    ):
        """
        Initialize command handler

        Args:
            exchange_connector: ExchangeConnector instance
            position_tracker: PositionTracker instance
            trade_logger: TradeLogger instance (optional)
            data_manager: DataManager instance (optional, for health check)
            enabled: If False, don't start polling
        """
        self.exchange = exchange_connector
        self.position_tracker = position_tracker
        self.trade_logger = trade_logger
        self.data_manager = data_manager
        self.enabled = enabled

        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.allowed_chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if self.enabled:
            if not self.bot_token or not self.allowed_chat_id:
                logger.warning("⚠️ Telegram credentials not set. Commands disabled.")
                self.enabled = False
            else:
                # Convert chat_id to int for comparison
                try:
                    self.allowed_chat_id = int(self.allowed_chat_id)
                except ValueError:
                    logger.error(f"❌ Invalid TELEGRAM_CHAT_ID: {self.allowed_chat_id}")
                    self.enabled = False

        self.last_update_id = 0
        self.polling_thread = None
        self.should_stop = False

        if self.enabled:
            logger.info("✅ Telegram command handler initialized")
            self.start_polling()

    def start_polling(self):
        """Start polling for commands in background thread"""
        if not self.enabled:
            return

        self.should_stop = False
        self.polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.polling_thread.start()
        logger.info("🔄 Telegram command polling started")

    def stop_polling(self):
        """Stop polling"""
        self.should_stop = True
        if self.polling_thread:
            self.polling_thread.join(timeout=5)
        logger.info("🛑 Telegram command polling stopped")

    def _poll_loop(self):
        """Main polling loop"""
        while not self.should_stop:
            try:
                self._check_updates()
                time.sleep(1)  # Poll every second
            except Exception as e:
                logger.error(f"❌ Polling error: {e}")
                time.sleep(5)  # Wait longer on error

    def _check_updates(self):
        """Check for new messages"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30,  # Long polling
                'allowed_updates': ['message']
            }

            response = requests.get(url, params=params, timeout=35)

            if response.status_code != 200:
                logger.error(f"❌ getUpdates failed: {response.status_code}")
                return

            data = response.json()

            if not data.get('ok'):
                logger.error(f"❌ Telegram API error: {data}")
                return

            updates = data.get('result', [])

            for update in updates:
                self.last_update_id = update['update_id']
                self._process_update(update)

        except requests.exceptions.Timeout:
            # Normal timeout, just continue
            pass
        except Exception as e:
            logger.error(f"❌ Check updates error: {e}")

    def _process_update(self, update: Dict):
        """Process a single update"""
        try:
            message = update.get('message')
            if not message:
                return

            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '')

            # Check if message is from allowed chat
            if chat_id != self.allowed_chat_id:
                logger.warning(f"⚠️ Ignored command from unauthorized chat: {chat_id}")
                return

            # Strip bot username from command (for group chats)
            # /command@botname -> /command
            if '@' in text:
                text = text.split('@')[0]

            # Handle commands
            if text == '/help':
                self._handle_help(chat_id)
            elif text == '/balance':
                self._handle_balance(chat_id)
            elif text == '/position':
                self._handle_position(chat_id)
            elif text == '/stats':
                self._handle_stats(chat_id)
            elif text == '/status':
                self._handle_status(chat_id)
            elif text == '/health':
                self._handle_health(chat_id)
            elif text == '/close':
                self._handle_close(chat_id)
            elif text.startswith('/'):
                # Unknown command
                self._send_message(chat_id, f"Unknown command: {text}\nSend /help for available commands")

        except Exception as e:
            logger.error(f"❌ Process update error: {e}")

    def _handle_balance(self, chat_id: int):
        """Handle /balance command"""
        try:
            balance = self.exchange.fetch_balance()

            # Get current positions value from tracker and calculate P&L
            positions_value = 0.0
            total_pnl_pct = 0.0
            current_price = None

            open_positions = self.position_tracker.get_open_positions()
            if open_positions:
                current_price = self.exchange.get_current_price()

                for pos in open_positions:
                    positions_value += pos.size_usdt

                    # Calculate P&L percentage for this position
                    if pos.action == 'LONG':
                        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    else:  # SHORT
                        pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100

                    total_pnl_pct += pnl_pct

            # Also show margin used from exchange (more reliable)
            margin_used = balance['used']

            # Get leverage
            leverage = getattr(self.exchange, 'leverage', 3.0)

            # Build message
            message = f"💰 *Balance*\n\nAvailable: ${balance['free']:.2f}\n"

            if open_positions:
                pnl_emoji = "🟢" if total_pnl_pct >= 0 else "🔴"
                # Show actual position value (not multiplied by leverage)
                # Leverage affects margin requirement, not position value
                message += f"In positions: ${positions_value:.2f} ({leverage:.0f}x) {pnl_emoji} {total_pnl_pct:+.2f}%\n"
            else:
                message += f"In positions: $0.00\n"

            message += f"Margin used: ${margin_used:.2f}\nTotal: ${balance['total']:.2f}"

            self._send_message(chat_id, message)
            logger.info("✅ Sent balance info")

        except Exception as e:
            logger.error(f"❌ Balance command error: {e}")
            self._send_message(chat_id, "❌ Error getting balance")

    def _handle_stats(self, chat_id: int):
        """Handle /stats command"""
        try:
            if not self.trade_logger:
                self._send_message(chat_id, "⚠️ Trade logging not enabled")
                return

            # Get stats from last 30 days
            cutoff = datetime.utcnow() - timedelta(days=30)
            trades = self.trade_logger.get_trades_since(cutoff)

            if not trades:
                self._send_message(chat_id, "📊 No trades in the last 30 days")
                return

            # Calculate stats
            total_trades = len(trades)
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losses = total_trades - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(t.get('pnl', 0) for t in trades)
            best_trade = max((t.get('pnl', 0) for t in trades), default=0)
            worst_trade = min((t.get('pnl', 0) for t in trades), default=0)

            # Average hold time
            hold_times = [t.get('hold_hours', 0) for t in trades]
            avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

            # Count LONG vs SHORT
            long_count = sum(1 for t in trades if t.get('action') == 'LONG')
            short_count = total_trades - long_count

            message = (
                f"📊 *Stats (Last 30 days)*\n\n"
                f"Trades: {total_trades} ({long_count} LONG, {short_count} SHORT)\n"
                f"Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)\n"
                f"Total PnL: ${total_pnl:+.2f}\n"
                f"Best trade: ${best_trade:+.2f}\n"
                f"Worst trade: ${worst_trade:+.2f}\n"
                f"Avg hold time: {avg_hold:.1f}h"
            )

            self._send_message(chat_id, message)
            logger.info("✅ Sent stats info")

        except Exception as e:
            logger.error(f"❌ Stats command error: {e}")
            self._send_message(chat_id, "❌ Error getting stats")

    def _handle_help(self, chat_id: int):
        """Handle /help command"""
        message = (
            "🤖 *AI Crypto Oracle Bot - Commands*\n\n"
            "*Account & Positions:*\n"
            "/balance - Show account balance and P&L\n"
            "/position - Show current open position details\n"
            "/close - Manually close current position\n\n"
            "*Statistics:*\n"
            "/stats - Trading statistics (last 30 days)\n"
            "/status - Bot status and last signal\n"
            "/health - Data health check\n\n"
            "*Other:*\n"
            "/help - Show this help message"
        )
        self._send_message(chat_id, message)
        logger.info("✅ Sent help info")

    def _handle_position(self, chat_id: int):
        """Handle /position command"""
        try:
            open_positions = self.position_tracker.get_open_positions()

            if not open_positions:
                self._send_message(chat_id, "📭 No open positions")
                return

            current_price = self.exchange.get_current_price()

            for pos in open_positions:
                # Calculate P&L
                if pos.action == 'LONG':
                    pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                    pnl_points = current_price - pos.entry_price
                else:  # SHORT
                    pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                    pnl_points = pos.entry_price - current_price

                # Get leverage from exchange
                leverage = getattr(self.exchange, 'leverage', 7.0)
                pnl_leveraged = pnl_pct * leverage

                # Calculate distances to TP/SL
                if pos.action == 'LONG':
                    tp_distance = ((pos.tp_price - current_price) / current_price) * 100
                    sl_distance = ((current_price - pos.sl_price) / current_price) * 100
                else:
                    tp_distance = ((current_price - pos.tp_price) / current_price) * 100
                    sl_distance = ((pos.sl_price - current_price) / current_price) * 100

                # Time info
                now = datetime.utcnow()
                hold_hours = (now - pos.entry_time).total_seconds() / 3600
                time_left_hours = pos.max_hold_hours - hold_hours
                expiry_time = pos.entry_time + timedelta(hours=pos.max_hold_hours)

                # Emoji for P&L
                pnl_emoji = "🟢" if pnl_leveraged >= 0 else "🔴"
                side_emoji = "📈" if pos.action == 'LONG' else "📉"

                message = (
                    f"{side_emoji} *{pos.action} Position*\n\n"
                    f"*Entry:* ${pos.entry_price:,.2f}\n"
                    f"*Current:* ${current_price:,.2f} ({pnl_points:+.2f})\n\n"
                    f"*P&L:* {pnl_emoji} {pnl_leveraged:+.2f}% ({leverage:.0f}x)\n"
                    f"*Size:* ${pos.size_usdt:,.2f}\n\n"
                    f"*Take Profit:* ${pos.tp_price:,.2f} (+{tp_distance:.2f}% away)\n"
                    f"*Stop Loss:* ${pos.sl_price:,.2f} (-{sl_distance:.2f}% away)\n\n"
                    f"*Opened:* {pos.entry_time.strftime('%Y-%m-%d %H:%M')} UTC\n"
                    f"*Hold time:* {hold_hours:.1f}h / {pos.max_hold_hours}h\n"
                    f"*Expires:* {expiry_time.strftime('%Y-%m-%d %H:%M')} UTC ({time_left_hours:.1f}h left)"
                )

                self._send_message(chat_id, message)
                logger.info("✅ Sent position info")

        except Exception as e:
            logger.error(f"❌ Position command error: {e}")
            self._send_message(chat_id, "❌ Error getting position info")

    def _handle_status(self, chat_id: int):
        """Handle /status command"""
        try:
            # Bot uptime (would need to track start time - simplified for now)
            # Get current price and market info
            current_price = self.exchange.get_current_price()

            # Check if we have open positions
            open_positions = self.position_tracker.get_open_positions()
            has_position = len(open_positions) > 0

            # Get data info from data_manager if available
            data_info = "N/A"
            if self.data_manager:
                summary = self.data_manager.get_summary()
                data_info = f"{summary['total_rows']} bars, latest: {summary['end_date']}"

            status_emoji = "🟢" if has_position else "⚪️"

            message = (
                f"{status_emoji} *Bot Status*\n\n"
                f"*Current Price:* ${current_price:,.2f}\n"
                f"*Data:* {data_info}\n"
                f"*Open Positions:* {len(open_positions)}\n"
                f"*Strategy:* VWAP Mean Reversion v3\n"
                f"*Timeframe:* 4H\n"
                f"*Leverage:* {getattr(self.exchange, 'leverage', 7.0):.0f}x\n"
                f"*Mode:* {'🎮 DEMO' if getattr(self.exchange, 'demo_mode', False) else '🚀 LIVE'}"
            )

            self._send_message(chat_id, message)
            logger.info("✅ Sent status info")

        except Exception as e:
            logger.error(f"❌ Status command error: {e}")
            self._send_message(chat_id, "❌ Error getting status")

    def _handle_health(self, chat_id: int):
        """Handle /health command - data integrity check"""
        try:
            if not self.data_manager:
                self._send_message(chat_id, "⚠️ Data manager not available")
                return

            # Run health check
            health = self.data_manager.health_check()

            # Determine emoji based on status
            status_emoji = {
                'HEALTHY': '🟢',
                'WARNING': '🟡',
                'CRITICAL': '🔴',
                'EMPTY': '⚪️'
            }.get(health['status'], '❓')

            # Build message
            message = f"{status_emoji} *Data Health: {health['status']}*\n\n"
            message += f"*Total Rows:* {health.get('total_rows', 0)}\n"

            if health.get('latest_timestamp'):
                message += f"*Latest:* {health['latest_timestamp']}\n"
                message += f"*Age:* {health.get('hours_old', 0):.1f}h\n"

            if health.get('gaps', 0) > 0:
                message += f"*Gaps:* {health['gaps']} detected\n"

            # Issues
            if health.get('issues'):
                message += f"\n*Issues:*\n"
                for issue in health['issues'][:3]:  # Show first 3
                    message += f"• {issue}\n"

            # Warnings
            if health.get('warnings'):
                message += f"\n*Warnings:*\n"
                for warning in health['warnings'][:3]:  # Show first 3
                    message += f"• {warning}\n"

            # Action
            if health.get('action'):
                message += f"\n*Action:* {health['action']}"

            self._send_message(chat_id, message)
            logger.info(f"✅ Sent health check: {health['status']}")

        except Exception as e:
            logger.error(f"❌ Health command error: {e}")
            self._send_message(chat_id, "❌ Error running health check")

    def _handle_close(self, chat_id: int):
        """Handle /close command - manually close position"""
        try:
            open_positions = self.position_tracker.get_open_positions()

            if not open_positions:
                self._send_message(chat_id, "📭 No open positions to close")
                return

            # Close all open positions (usually just 1)
            current_price = self.exchange.get_current_price()

            for pos in open_positions:
                try:
                    # Close on exchange
                    self.exchange.close_position(pos.action.lower())

                    # Close in tracker
                    leverage = getattr(self.exchange, 'leverage', 7.0)
                    result = self.position_tracker.close_position(
                        action=pos.action,
                        exit_price=current_price,
                        reason='Manual',
                        leverage=leverage
                    )

                    if result:
                        # Log trade if logger available
                        if self.trade_logger:
                            self.trade_logger.log_trade(result)

                        pnl_emoji = "🟢" if result['pnl'] >= 0 else "🔴"

                        message = (
                            f"✅ *Position Closed Manually*\n\n"
                            f"Side: {result['action']}\n"
                            f"Entry: ${result['entry_price']:,.2f}\n"
                            f"Exit: ${result['exit_price']:,.2f}\n"
                            f"P&L: {pnl_emoji} {result['pnl_pct']:+.2f}% (${result['pnl']:+.2f})\n"
                            f"Hold time: {result['hold_hours']:.1f}h"
                        )

                        self._send_message(chat_id, message)
                        logger.info(f"✅ Manually closed {pos.action} position via Telegram")

                except Exception as close_error:
                    logger.error(f"❌ Failed to close position: {close_error}")
                    self._send_message(chat_id, f"❌ Error closing position: {close_error}")

        except Exception as e:
            logger.error(f"❌ Close command error: {e}")
            self._send_message(chat_id, "❌ Error closing position")

    def _send_message(self, chat_id: int, message: str):
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code != 200:
                logger.error(f"❌ Send message failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"❌ Send message error: {e}")


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    # Mock objects for testing
    class MockExchange:
        def get_balance(self):
            return {'free': 4523.45, 'used': 1200.0, 'total': 5723.45}

    class MockTracker:
        def get_open_positions(self):
            return []

    handler = TelegramCommandHandler(
        exchange_connector=MockExchange(),
        position_tracker=MockTracker(),
        enabled=True
    )

    print("Polling for commands... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        handler.stop_polling()
