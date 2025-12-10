"""
Data Manager for VWAP Strategy
Manages 4H historical data with caching and incremental updates
Downloads only necessary bars for VWAP calculation (minimum 1 day = 6 bars per day)
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import json

logger = logging.getLogger(__name__)


class TradingDataManagerVWAP:
    """
    Data manager for VWAP Mean Reversion strategy using 4H candles
    Downloads and caches data with minimum 2 days (12 bars) for VWAP calculation
    """

    def __init__(
        self,
        historical_csv_path: str,
        state_file_path: str = 'data/state_vwap.json',
        min_days: int = 7  # Minimum 7 days for stable VWAP calculation
    ):
        """
        Initialize VWAP data manager

        Args:
            historical_csv_path: Path to save 4H data CSV
            state_file_path: Path to save/load state
            min_days: Minimum number of days to keep (7 days = ~42 bars on 4H)
        """
        self.historical_csv_path = historical_csv_path
        self.state_file_path = state_file_path
        self.min_days = min_days
        self.min_bars = min_days * 6  # 4H timeframe: 6 bars per day

        # DataFrame for 4H data
        self.df_4h = None

        # Load existing data or initialize empty
        if os.path.exists(state_file_path):
            logger.info(f"📦 Loading state from {state_file_path}")
            self.load_state()
        elif os.path.exists(historical_csv_path):
            logger.info(f"📊 Loading historical data from {historical_csv_path}")
            self._load_historical_csv()
        else:
            logger.info(f"📊 Initializing empty 4H dataset (will download {self.min_bars} bars on first run)")
            self.df_4h = pd.DataFrame()

        if self.df_4h is not None and len(self.df_4h) > 0:
            logger.info(f"✅ Data manager initialized with {len(self.df_4h)} 4H bars (~{len(self.df_4h)/6:.1f} days)")
            logger.info(f"   Latest timestamp: {self.df_4h.index[-1]}")
        else:
            logger.info(f"✅ Data manager initialized (empty, will download {self.min_bars} bars)")

    def _load_historical_csv(self):
        """Load 4H data from CSV"""
        try:
            df = pd.read_csv(self.historical_csv_path, parse_dates=['timestamp'])

            if len(df) == 0:
                logger.warning("⚠️ CSV is empty, will re-download data")
                self.df_4h = pd.DataFrame()
                return

            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"❌ CSV missing columns: {missing_cols}. Will re-download data")
                self.df_4h = pd.DataFrame()
                return

            # Set index and sort
            df = df.set_index('timestamp').sort_index()
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Drop NaN and Inf
            df = df.dropna()
            df = df.replace([float('inf'), float('-inf')], pd.NA).dropna()

            logger.info(f"✅ Loaded {len(df)} bars from CSV")
            self.df_4h = df

        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            self.df_4h = pd.DataFrame()

    def save_state(self):
        """Save current state to JSON"""
        try:
            state = {
                'last_update': datetime.utcnow().isoformat(),
                'bars_count': len(self.df_4h) if self.df_4h is not None else 0,
                'latest_timestamp': str(self.df_4h.index[-1]) if self.df_4h is not None and len(self.df_4h) > 0 else None
            }

            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            with open(self.state_file_path, 'w') as f:
                json.dump(state, f, indent=2)

            # Save CSV
            if self.df_4h is not None and len(self.df_4h) > 0:
                df_save = self.df_4h.copy()
                df_save.reset_index(inplace=True)
                os.makedirs(os.path.dirname(self.historical_csv_path), exist_ok=True)
                df_save.to_csv(self.historical_csv_path, index=False)
                logger.debug(f"💾 Saved {len(df_save)} bars to {self.historical_csv_path}")

        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")

    def load_state(self):
        """Load state from JSON"""
        try:
            with open(self.state_file_path, 'r') as f:
                state = json.load(f)

            logger.info(f"📦 State loaded: {state.get('bars_count', 0)} bars, last update: {state.get('last_update')}")

            # Load CSV data
            if os.path.exists(self.historical_csv_path):
                self._load_historical_csv()
            else:
                logger.warning("⚠️ State exists but CSV missing, will re-download")
                self.df_4h = pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Error loading state: {e}")
            self.df_4h = pd.DataFrame()

    def update_historical_data(self, exchange_connector) -> bool:
        """
        Update historical data by fetching new candles

        Args:
            exchange_connector: BybitConnector instance

        Returns:
            True if update successful
        """
        try:
            # Determine how many bars to fetch
            if self.df_4h is None or len(self.df_4h) == 0:
                # Initial download: fetch min_bars
                bars_needed = self.min_bars
                logger.info(f"📥 Initial download: fetching {bars_needed} bars (~{bars_needed/6:.1f} days)")

                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=4 * bars_needed)

                # Fetch candles (fetch_ohlcv returns DataFrame already)
                df_new = exchange_connector.fetch_ohlcv(
                    since=start_time,
                    limit=bars_needed,
                    interval='240'  # 4H
                )

                if df_new is None or len(df_new) == 0:
                    logger.error("❌ Failed to fetch initial candles")
                    return False

                self.df_4h = df_new
                logger.info(f"✅ Initial download complete: {len(self.df_4h)} bars")

            else:
                # Incremental update: fetch candles since last timestamp
                last_timestamp = self.df_4h.index[-1]
                current_time = datetime.utcnow()
                time_diff = current_time - last_timestamp

                # Check if we need to update
                if time_diff < timedelta(hours=4):
                    logger.debug(f"⏱️ Last candle is {time_diff.total_seconds()/60:.0f}m old, no update needed yet")
                    return True

                logger.info(f"📥 Incremental update: last candle is {time_diff.total_seconds()/3600:.1f}h old")

                # Fetch new candles
                since_time = last_timestamp + timedelta(seconds=1)

                df_new = exchange_connector.fetch_ohlcv(
                    since=since_time,
                    limit=200,  # Max limit
                    interval='240'  # 4H
                )

                if df_new is None or len(df_new) == 0:
                    logger.debug("⚠️ No new candles returned")
                    return True

                # Append new candles (drop duplicates)
                self.df_4h = pd.concat([self.df_4h, df_new]).sort_index()
                self.df_4h = self.df_4h[~self.df_4h.index.duplicated(keep='last')]

                logger.info(f"✅ Added {len(df_new)} new candles, total: {len(self.df_4h)} bars")

                # Trim old data (keep only min_bars * 2 for safety)
                max_bars_to_keep = self.min_bars * 2
                if len(self.df_4h) > max_bars_to_keep:
                    self.df_4h = self.df_4h.iloc[-max_bars_to_keep:]
                    logger.info(f"🧹 Trimmed to {len(self.df_4h)} bars (keeping {max_bars_to_keep} max)")

            # Save state
            self.save_state()
            return True

        except Exception as e:
            logger.error(f"❌ Error updating historical data: {e}")
            return False

    def get_4h_data(self) -> Optional[pd.DataFrame]:
        """
        Get 4H DataFrame for indicators calculation

        Returns:
            DataFrame with OHLCV data or None
        """
        if self.df_4h is None or len(self.df_4h) < self.min_bars:
            logger.warning(f"⚠️ Insufficient data: {len(self.df_4h) if self.df_4h is not None else 0} bars (need {self.min_bars})")
            return None

        return self.df_4h.copy()

    def get_latest_candle(self) -> Optional[dict]:
        """
        Get latest 4H candle as dict

        Returns:
            Dict with OHLCV data or None
        """
        if self.df_4h is None or len(self.df_4h) == 0:
            return None

        latest = self.df_4h.iloc[-1]
        return {
            'timestamp': self.df_4h.index[-1],
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        }
