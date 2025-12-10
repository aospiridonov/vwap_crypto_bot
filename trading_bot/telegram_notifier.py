"""
Telegram Notifier
Sends notifications about trading events to Telegram
"""

import os
import logging
import requests
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends Telegram notifications for bot events
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize Telegram notifier

        Args:
            enabled: If False, just log messages instead of sending
        """
        self.enabled = enabled
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if self.enabled:
            if not self.bot_token or not self.chat_id:
                logger.warning("⚠️ Telegram credentials not set. Notifications disabled.")
                self.enabled = False
            else:
                logger.info("✅ Telegram notifier initialized")

    def send(self, message: str, silent: bool = False) -> bool:
        """
        Send message to Telegram

        Args:
            message: Message text (supports Markdown)
            silent: Send without notification sound

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.info(f"[Telegram] {message}")
            return True

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_notification': silent
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.debug(f"✅ Telegram sent: {message[:50]}...")
                return True
            else:
                logger.error(f"❌ Telegram failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
            return False

    def notify_signal(self, signal: Dict, position_size_usdt: float):
        """
        Notify about new trading signal

        Args:
            signal: Signal dict from signal_generator
            position_size_usdt: Position size in USDT
        """
        action = signal['action']
        entry = signal['entry_price']
        tp = signal['tp_price']
        sl = signal['sl_price']
        reason = signal.get('reason', 'N/A')

        icon = "📈" if action == "LONG" else "📉"

        # Calculate TP/SL percentages
        if action == "LONG":
            tp_pct = (tp / entry - 1) * 100
            sl_pct = (1 - sl / entry) * 100
        else:
            tp_pct = (1 - tp / entry) * 100
            sl_pct = (sl / entry - 1) * 100

        # Build message - V38 format (indicator-based, no probability)
        message = (
            f"{icon} *{action} Signal Detected*\n\n"
            f"📋 Reason: `{reason}`\n"
            f"💰 Entry: `${entry:,.2f}`\n"
            f"🎯 TP: `${tp:,.2f}` (+{tp_pct:.1f}%)\n"
            f"🛡️ SL: `${sl:,.2f}` (-{sl_pct:.1f}%)\n"
        )

        # Add indicator values if available (V38)
        if 'ao' in signal:
            message += f"📊 AO: `{signal['ao']:.2f}`\n"
        if 'macd' in signal:
            message += f"📈 MACD: `{signal['macd']:.2f}/{signal.get('macd_signal', 0):.2f}`\n"
        if 'adx' in signal:
            message += f"💪 ADX: `{signal['adx']:.1f}`\n"
        if 'chop' in signal:
            message += f"🌊 Chop: `{signal['chop']:.1f}`\n"
        if 'fng' in signal:
            message += f"😱 F&G: `{signal['fng']}`\n"

        message += (
            f"💵 Position: `${position_size_usdt:,.2f}`\n"
            f"⏰ Time: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`"
        )

        self.send(message)

    def notify_position_opened(self, action: str, entry: float, tp: float, sl: float, size: float):
        """Notify when position is opened"""
        icon = "📈" if action == "LONG" else "📉"

        message = (
            f"{icon} *Position Opened*\n\n"
            f"Direction: `{action}`\n"
            f"Entry: `${entry:,.2f}`\n"
            f"TP: `${tp:,.2f}`\n"
            f"SL: `${sl:,.2f}`\n"
            f"Size: `${size:,.2f}`"
        )

        self.send(message)

    def notify_position_closed(
        self,
        action: str,
        entry: float,
        exit: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
        fees: float = 0.0
    ):
        """
        Notify when position is closed

        Args:
            action: 'LONG' or 'SHORT'
            entry: Entry price
            exit: Exit price
            pnl: P&L in USDT
            pnl_pct: P&L percentage
            reason: 'TP', 'SL', 'Timeout', 'Manual'
            fees: Trading fees
        """
        if pnl > 0:
            icon = "💰"
            pnl_emoji = "✅"
        else:
            icon = "❌"
            pnl_emoji = "❌"

        pnl_after_fees = pnl - fees

        message = (
            f"{icon} *Position Closed: {reason}*\n\n"
            f"Direction: `{action}`\n"
            f"Entry: `${entry:,.2f}`\n"
            f"Exit: `${exit:,.2f}`\n"
            f"{pnl_emoji} PnL: `${pnl:+,.2f}` ({pnl_pct:+.2f}%)\n"
            f"💸 Fees: `${fees:.2f}`\n"
            f"💵 Net PnL: `${pnl_after_fees:+,.2f}`"
        )

        self.send(message)

    def notify_monthly_retrain_start(self, month: str):
        """Notify when monthly retrain starts"""
        message = (
            f"🔄 *Monthly Retrain Started*\n\n"
            f"Month: `{month}`\n"
            f"Time: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`\n\n"
            f"Models will be updated shortly..."
        )

        self.send(message)

    def notify_monthly_retrain_complete(self, month: str, train_size: int, metrics: Optional[Dict] = None):
        """Notify when monthly retrain completes"""
        message = (
            f"✅ *Monthly Retrain Completed*\n\n"
            f"Month: `{month}`\n"
            f"Training samples: `{train_size:,}`\n"
        )

        if metrics:
            message += f"\n📊 Validation Metrics:\n"
            for key, value in metrics.items():
                message += f"  • {key}: `{value:.4f}`\n"

        message += f"\n🚀 New models are now active!"

        self.send(message)

    def notify_error(self, error_msg: str, context: str = ""):
        """Notify about errors"""
        message = (
            f"🔴 *Error Occurred*\n\n"
            f"Context: `{context}`\n"
            f"Error: `{error_msg}`\n"
            f"Time: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`"
        )

        self.send(message)

    def notify_connection_lost(self):
        """Notify when connection is lost"""
        message = "⚠️ *Connection Lost*\n\nAttempting to reconnect..."
        self.send(message)

    def notify_reconnected(self):
        """Notify when reconnected"""
        message = "✅ *Reconnected*\n\nTrading bot is back online"
        self.send(message)

    def notify_balance_low(self, balance: float, min_balance: float):
        """Notify when balance is too low"""
        message = (
            f"⚠️ *Low Balance Warning*\n\n"
            f"Current: `${balance:,.2f}`\n"
            f"Minimum: `${min_balance:,.2f}`\n\n"
            f"Please deposit funds to continue trading"
        )

        self.send(message)

    def notify_bot_stopped(self, reason: str = "Manual stop"):
        """Notify when bot stops"""
        message = (
            f"🛑 *Bot Stopped*\n\n"
            f"Reason: `{reason}`\n"
            f"Time: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC`"
        )

        self.send(message)

    def get_balance_info(self, balance: Dict) -> str:
        """
        Format balance information

        Args:
            balance: Dict with 'free', 'used', 'total'

        Returns:
            Formatted message
        """
        message = (
            f"💰 *Balance*\n\n"
            f"Available: `${balance['free']:,.2f}`\n"
            f"In positions: `${balance['used']:,.2f}`\n"
            f"Total: `${balance['total']:,.2f}`"
        )

        return message

    def get_stats(self, stats: Dict) -> str:
        """
        Format trading statistics

        Args:
            stats: Dict with trading stats

        Returns:
            Formatted message
        """
        message = (
            f"📊 *Trading Stats (Last 30 days)*\n\n"
            f"Trades: `{stats.get('total_trades', 0)}`\n"
            f"Win Rate: `{stats.get('win_rate', 0):.1f}%`\n"
            f"Total PnL: `${stats.get('total_pnl', 0):+,.2f}` ({stats.get('total_pnl_pct', 0):+.1f}%)\n"
            f"Best trade: `${stats.get('best_trade', 0):+,.2f}`\n"
            f"Worst trade: `${stats.get('worst_trade', 0):+,.2f}`\n"
            f"Avg hold time: `{stats.get('avg_hold_hours', 0):.1f}h`"
        )

        return message


if __name__ == "__main__":
    # Test Telegram notifier
    logging.basicConfig(level=logging.INFO)

    notifier = TelegramNotifier(enabled=False)  # Set to True to actually send

    # Test signal notification
    test_signal = {
        'action': 'LONG',
        'probability': 0.7523,
        'regime': 7,
        'entry_price': 43250.0,
        'tp_price': 44980.0,
        'sl_price': 42385.0,
        'confidence': 'HIGH'
    }

    notifier.notify_signal(test_signal, position_size_usdt=1500.0)

    # Test position closed notification
    notifier.notify_position_closed(
        action='LONG',
        entry=43250.0,
        exit=44980.0,
        pnl=520.0,
        pnl_pct=4.0,
        reason='TP',
        fees=5.0
    )

    # Test balance info
    print(notifier.get_balance_info({'free': 4523.45, 'used': 1200.0, 'total': 5723.45}))
