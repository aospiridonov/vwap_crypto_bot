"""
Trade Logger
Logs all trades to CSV for statistics and analysis
"""

import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs all trades to CSV file
    """

    def __init__(self, log_file: str = 'data/trades.csv'):
        """
        Initialize trade logger

        Args:
            log_file: Path to trades CSV file
        """
        self.log_file = log_file

        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Load existing trades or create new file
        if os.path.exists(log_file):
            try:
                self.trades = pd.read_csv(log_file, parse_dates=['entry_time', 'exit_time'])
                logger.info(f"📦 Loaded {len(self.trades)} trades from {log_file}")
            except Exception as e:
                logger.error(f"❌ Failed to load trades: {e}")
                self.trades = pd.DataFrame()
        else:
            self.trades = pd.DataFrame()
            logger.info(f"📝 Created new trade log: {log_file}")

    def log_trade(self, trade: Dict):
        """
        Log a completed trade

        Args:
            trade: Trade dict with keys:
                - action: 'LONG' or 'SHORT'
                - entry_price: Entry price
                - exit_price: Exit price
                - tp_price: Take profit price
                - sl_price: Stop loss price
                - pnl: P&L in USDT
                - pnl_pct: P&L percentage
                - size_usdt: Position size
                - reason: 'TP', 'SL', 'Timeout', 'Manual'
                - hold_hours: Hold time in hours
                - regime: Market regime
                - probability: Model probability
                - entry_time: Entry timestamp
                - exit_time: Exit timestamp
        """
        try:
            # Add to dataframe
            trade_df = pd.DataFrame([trade])
            self.trades = pd.concat([self.trades, trade_df], ignore_index=True)

            # Save to CSV
            self.trades.to_csv(self.log_file, index=False)

            logger.info(
                f"✅ Trade logged: {trade['action']} | "
                f"PnL=${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) | "
                f"Reason={trade['reason']}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")

    def get_trades_since(self, cutoff: datetime) -> List[Dict]:
        """
        Get trades since a certain date

        Args:
            cutoff: Only return trades after this date

        Returns:
            List of trade dicts
        """
        if len(self.trades) == 0:
            return []

        try:
            # Filter by exit_time
            filtered = self.trades[self.trades['exit_time'] >= cutoff]
            return filtered.to_dict('records')

        except Exception as e:
            logger.error(f"❌ Failed to get trades: {e}")
            return []

    def get_stats(self, days: int = 30) -> Dict:
        """
        Get trading statistics for last N days

        Args:
            days: Number of days to analyze

        Returns:
            Dict with statistics
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_hold_hours': 0
            }

        try:
            cutoff = datetime.utcnow() - pd.Timedelta(days=days)
            recent = self.trades[self.trades['exit_time'] >= cutoff]

            if len(recent) == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'avg_hold_hours': 0
                }

            total_trades = len(recent)
            wins = (recent['pnl'] > 0).sum()
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': recent['pnl'].sum(),
                'total_pnl_pct': recent['pnl_pct'].sum(),
                'best_trade': recent['pnl'].max(),
                'worst_trade': recent['pnl'].min(),
                'avg_hold_hours': recent['hold_hours'].mean()
            }

        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    logger_obj = TradeLogger(log_file='test_trades.csv')

    # Log test trade
    test_trade = {
        'action': 'LONG',
        'entry_price': 43250.0,
        'exit_price': 44980.0,
        'tp_price': 44980.0,
        'sl_price': 42385.0,
        'pnl': 520.0,
        'pnl_pct': 4.0,
        'size_usdt': 1500.0,
        'reason': 'TP',
        'hold_hours': 5.2,
        'regime': 7,
        'probability': 0.75,
        'entry_time': datetime(2025, 11, 25, 2, 0, 0),
        'exit_time': datetime(2025, 11, 25, 7, 12, 0)
    }

    logger_obj.log_trade(test_trade)

    # Get stats
    stats = logger_obj.get_stats(30)
    print(f"Stats: {stats}")

    # Cleanup
    if os.path.exists('test_trades.csv'):
        os.remove('test_trades.csv')
