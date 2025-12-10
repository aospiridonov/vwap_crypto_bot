"""
Position Tracker
Tracks open positions and manages TP/SL/Timeout logic
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


class Position:
    """
    Represents a single open position
    """

    def __init__(
        self,
        action: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        size_usdt: float,
        max_hold_hours: int,
        entry_time: Optional[datetime] = None,
        regime: int = 7,
        probability: float = 0.0
    ):
        """
        Initialize position

        Args:
            action: 'LONG' or 'SHORT'
            entry_price: Entry price
            tp_price: Take profit price
            sl_price: Stop loss price
            size_usdt: Position size in USDT
            max_hold_hours: Maximum hold time
            entry_time: Entry timestamp (default: now)
            regime: Market regime at entry
            probability: Model probability at entry
        """
        self.action = action
        self.entry_price = entry_price
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.size_usdt = size_usdt
        self.max_hold_hours = max_hold_hours
        self.entry_time = entry_time or datetime.utcnow()
        self.regime = regime
        self.probability = probability

        # Calculated fields
        self.expiry_time = self.entry_time + timedelta(hours=max_hold_hours)

    def check_exit(self, current_price: float) -> Optional[str]:
        """
        Check if position should be exited

        Args:
            current_price: Current market price

        Returns:
            Exit reason or None: 'TP', 'SL', 'Timeout'
        """
        # Check TP
        if self.action == 'LONG' and current_price >= self.tp_price:
            return 'TP'
        elif self.action == 'SHORT' and current_price <= self.tp_price:
            return 'TP'

        # Check SL
        if self.action == 'LONG' and current_price <= self.sl_price:
            return 'SL'
        elif self.action == 'SHORT' and current_price >= self.sl_price:
            return 'SL'

        # Check timeout
        if datetime.utcnow() >= self.expiry_time:
            return 'Timeout'

        return None

    def calculate_pnl(self, exit_price: float, leverage: float = 3.0) -> Dict:
        """
        Calculate P&L at exit price

        Args:
            exit_price: Exit price
            leverage: Leverage multiplier

        Returns:
            Dict with pnl, pnl_pct, exit_price
        """
        # Prevent division by zero
        if self.entry_price <= 0:
            logger.error(f"❌ Invalid entry_price: {self.entry_price}, cannot calculate PnL")
            return {'pnl': 0.0, 'pnl_pct': 0.0, 'exit_price': exit_price}

        if self.action == 'LONG':
            pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100 * leverage
        else:  # SHORT
            pnl_pct = ((self.entry_price - exit_price) / self.entry_price) * 100 * leverage

        pnl_usdt = self.size_usdt * (pnl_pct / 100)

        return {
            'pnl': pnl_usdt,
            'pnl_pct': pnl_pct,
            'exit_price': exit_price
        }

    def get_hold_time_hours(self) -> float:
        """Get current hold time in hours"""
        return (datetime.utcnow() - self.entry_time).total_seconds() / 3600

    def to_dict(self) -> Dict:
        """Convert to dict for serialization"""
        return {
            'action': self.action,
            'entry_price': self.entry_price,
            'tp_price': self.tp_price,
            'sl_price': self.sl_price,
            'size_usdt': self.size_usdt,
            'max_hold_hours': self.max_hold_hours,
            'entry_time': self.entry_time.isoformat(),
            'regime': self.regime,
            'probability': self.probability
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Position':
        """Create Position from dict"""
        return Position(
            action=data['action'],
            entry_price=data['entry_price'],
            tp_price=data['tp_price'],
            sl_price=data['sl_price'],
            size_usdt=data['size_usdt'],
            max_hold_hours=data['max_hold_hours'],
            entry_time=datetime.fromisoformat(data['entry_time']),
            regime=data.get('regime', 7),
            probability=data.get('probability', 0.0)
        )


class PositionTracker:
    """
    Tracks all open positions and manages state
    """

    def __init__(self, state_file: str = 'data/positions.json', max_hold_hours: int = 160):
        """
        Initialize position tracker

        Args:
            state_file: Path to save/load positions
            max_hold_hours: Maximum hold time for positions (default: 160h = 40 bars × 4h)
        """
        self.state_file = state_file
        self.max_hold_hours = max_hold_hours
        self.positions: Dict[str, Position] = {}

        # Load existing positions
        self.load_positions()

        logger.info(f"✅ Position tracker initialized with {len(self.positions)} positions (max_hold={max_hold_hours}h)")

    def open_position(self, position: Position) -> bool:
        """
        Open new position

        Args:
            position: Position object

        Returns:
            True if opened successfully
        """
        # Check if position already exists
        if position.action in self.positions:
            logger.warning(f"⚠️ Position {position.action} already exists")
            return False

        self.positions[position.action] = position

        logger.info(
            f"✅ Position opened: {position.action} | "
            f"Entry=${position.entry_price:.2f} | "
            f"TP=${position.tp_price:.2f} | "
            f"SL=${position.sl_price:.2f} | "
            f"Size=${position.size_usdt:.2f} | "
            f"Expiry={position.expiry_time.strftime('%Y-%m-%d %H:%M')}"
        )

        self.save_positions()
        return True

    def close_position(self, action: str, exit_price: float, reason: str, leverage: float = 3.0) -> Optional[Dict]:
        """
        Close position and calculate P&L

        Args:
            action: 'LONG' or 'SHORT'
            exit_price: Exit price
            reason: Exit reason ('TP', 'SL', 'Timeout', 'Manual')
            leverage: Leverage multiplier

        Returns:
            Dict with position details and P&L or None if no position
        """
        if action not in self.positions:
            logger.warning(f"⚠️ No {action} position to close")
            return None

        position = self.positions.pop(action)

        # Calculate P&L
        pnl_info = position.calculate_pnl(exit_price, leverage)
        hold_hours = position.get_hold_time_hours()

        result = {
            'action': action,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'tp_price': position.tp_price,
            'sl_price': position.sl_price,
            'pnl': pnl_info['pnl'],
            'pnl_pct': pnl_info['pnl_pct'],
            'size_usdt': position.size_usdt,
            'reason': reason,
            'hold_hours': hold_hours,
            'regime': position.regime,
            'probability': position.probability,
            'entry_time': position.entry_time,
            'exit_time': datetime.utcnow()
        }

        logger.info(
            f"✅ Position closed: {action} | "
            f"Reason={reason} | "
            f"Entry=${position.entry_price:.2f} | "
            f"Exit=${exit_price:.2f} | "
            f"PnL=${pnl_info['pnl']:+.2f} ({pnl_info['pnl_pct']:+.2f}%) | "
            f"Hold={hold_hours:.1f}h"
        )

        self.save_positions()
        return result

    def check_all_positions(self, current_price: float, leverage: float = 3.0) -> list:
        """
        Check all open positions for exit conditions

        Args:
            current_price: Current market price
            leverage: Leverage multiplier

        Returns:
            List of closed position results
        """
        closed_positions = []

        for action in list(self.positions.keys()):
            position = self.positions[action]

            exit_reason = position.check_exit(current_price)

            if exit_reason:
                # Determine exit price based on reason
                if exit_reason == 'TP':
                    exit_price = position.tp_price
                elif exit_reason == 'SL':
                    exit_price = position.sl_price
                else:  # Timeout
                    exit_price = current_price

                result = self.close_position(action, exit_price, exit_reason, leverage)
                if result:
                    closed_positions.append(result)

        return closed_positions

    def get_open_positions(self) -> list:
        """Get list of all open positions"""
        return list(self.positions.values())

    def has_position(self, action: Optional[str] = None) -> bool:
        """
        Check if there are any open positions

        Args:
            action: If specified, check for specific action (LONG/SHORT)

        Returns:
            True if position exists
        """
        if action:
            return action in self.positions
        return len(self.positions) > 0

    def get_position_summary(self) -> str:
        """Get human-readable summary of positions"""
        if not self.positions:
            return "No open positions"

        summary = []
        for action, pos in self.positions.items():
            hold_hours = pos.get_hold_time_hours()
            summary.append(
                f"{action}: Entry=${pos.entry_price:.2f}, TP=${pos.tp_price:.2f}, "
                f"SL=${pos.sl_price:.2f}, Hold={hold_hours:.1f}h/{pos.max_hold_hours}h"
            )

        return "\n".join(summary)

    def sync_position_from_websocket(self, data: Dict):
        """
        Sync position from WebSocket/API update

        Args:
            data: Position data with keys:
                - symbol, side, size, entry_price, mark_price, pnl
                - take_profit, stop_loss (optional, from exchange API)
        """
        try:
            side = 'LONG' if data['side'] == 'Buy' else 'SHORT'
            size = data.get('size', 0)

            # If position doesn't exist in tracker, add it
            if size > 0 and not self.has_position(side):
                # Create Position object from data
                entry_price = data.get('entry_price', 0)

                # Validate entry_price to prevent division by zero
                if not entry_price or entry_price <= 0:
                    logger.warning(f"⚠️ Invalid entry_price ({entry_price}) from sync data, skipping")
                    return

                position_value = data.get('size', 0) * entry_price

                # Use TP/SL from exchange if available, otherwise use defaults
                tp_price = data.get('take_profit')
                sl_price = data.get('stop_loss')

                # Validate TP/SL prices (must be positive numbers)
                if not tp_price or tp_price <= 0 or not sl_price or sl_price <= 0:
                    logger.warning(f"⚠️ Invalid TP/SL from exchange (TP={tp_price}, SL={sl_price}), using defaults (+4%/-2%)")
                    tp_price = entry_price * 1.04 if side == 'LONG' else entry_price * 0.96
                    sl_price = entry_price * 0.98 if side == 'LONG' else entry_price * 1.02

                position = Position(
                    action=side,
                    entry_price=entry_price,
                    size_usdt=position_value,
                    tp_price=tp_price,
                    sl_price=sl_price,
                    max_hold_hours=self.max_hold_hours
                )

                self.positions[side] = position
                self.save_positions()
                logger.info(f"📌 Synced position: {side} @ ${entry_price:.2f} | TP=${tp_price:.2f} | SL=${sl_price:.2f}")

        except Exception as e:
            logger.error(f"❌ Error syncing position from WebSocket: {e}")

    def save_positions(self):
        """Save positions to file"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

            data = {
                action: pos.to_dict()
                for action, pos in self.positions.items()
            }

            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"💾 Positions saved to {self.state_file}")

        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def load_positions(self):
        """Load positions from file"""
        try:
            if not os.path.exists(self.state_file):
                logger.info("No existing positions file")
                return

            with open(self.state_file, 'r') as f:
                data = json.load(f)

            self.positions = {
                action: Position.from_dict(pos_data)
                for action, pos_data in data.items()
            }

            logger.info(f"📦 Loaded {len(self.positions)} positions from {self.state_file}")

        except Exception as e:
            logger.error(f"❌ Failed to load positions: {e}")


if __name__ == "__main__":
    # Test position tracker
    logging.basicConfig(level=logging.INFO)

    tracker = PositionTracker(state_file='test_positions.json')

    # Create test position
    pos = Position(
        action='LONG',
        entry_price=43250.0,
        tp_price=44980.0,
        sl_price=42385.0,
        size_usdt=1500.0,
        max_hold_hours=18,
        regime=7,
        probability=0.75
    )

    # Open position
    tracker.open_position(pos)

    # Check exit conditions
    print(f"Check at $44000: {pos.check_exit(44000.0)}")  # None
    print(f"Check at $45000: {pos.check_exit(45000.0)}")  # TP
    print(f"Check at $42000: {pos.check_exit(42000.0)}")  # SL

    # Close position
    result = tracker.close_position('LONG', 44980.0, 'TP', leverage=3.0)
    print(f"Closed position: {result}")

    # Cleanup
    if os.path.exists('test_positions.json'):
        os.remove('test_positions.json')
