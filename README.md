# Indicators Crypto Bot - VWAP Mean Reversion v3

Trading bot for BTC/USDT using VWAP ±2σ Mean Reversion strategy with adaptive leverage (7x → 3x).

## Strategy Overview

**ST2 VWAP Mean Reversion v3** - Champion strategy with adaptive risk management:

- **Win Rate**: 97.2% (2025 data, 283 trades with TP≥0.3% filter)
- **Base Leverage**: 7x (adaptive: reduces to 3x on losing streaks)
- **Timeframe**: 4H for signals, WebSocket for TP/SL tracking
- **Entry**: Price deviation ≥ 2σ from daily VWAP
- **Exit**: Mean reversion to 0.8σ or SL at 1.8× ATR
- **TP Filter**: Skips trades with TP < 0.3% (unprofitable with 0.77% effective fees at 7x)

### Adaptive Risk Management

1. **After 2 SL in a row** → Leverage reduced to 3x
2. **After 3 SL in a row** → Trading paused for 24 hours
3. **When DD > 35%** → Leverage reduced to 5x
4. **When DD > 45%** → Circuit breaker (trading stopped)
5. **Recovery**: After 5 wins + DD < 10% → Leverage restored to 7x

### Performance (2025, with TP filter)

| Metric | Value |
|--------|-------|
| **Trades** | 283 |
| **Win Rate** | 97.2% |
| **Max Drawdown** | 27.4% |
| **Profit Factor** | 14.25 |
| **Final Capital** | $206M (from $1K, 11 months) |
| **Total Return** | +20,585,916% |
| **Fees Paid** | $1.37B (Bybit 0.110% round-trip) |

**⚠️ Note**: These are backtest results with full compounding. Realistic expectations for live trading: 3-10x per year depending on capital size.

## Setup

### Prerequisites

- Docker & Docker Compose
- Bybit API keys (Demo or Live)
- Telegram Bot (optional, for notifications)

### Installation

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   # Bybit Demo API (virtual money)
   BYBIT_DEMO_API_KEY=your_demo_key
   BYBIT_DEMO_SECRET=your_demo_secret
   BYBIT_DEMO_MODE=true

   # Telegram (optional)
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

3. **Build and start the bot**:
   ```bash
   ./rebuild.sh
   ```

## Usage

### Management Scripts

```bash
./run.sh       # Start bot
./stop.sh      # Stop bot
./restart.sh   # Restart bot
./logs.sh      # View logs (follow mode)
./status.sh    # Check bot status
./rebuild.sh   # Rebuild and restart
./clean.sh     # Stop and remove containers
```

### Configuration

Edit `config/trading_params.json`:

```json
{
  "symbol": "BTCUSDT",
  "leverage": 7,
  "position_size_pct": 1.0,
  "max_hold_hours": 64,

  "sigma_entry": 2.0,
  "sigma_exit": 0.8,
  "sl_atr_mult": 1.8,
  "min_tp_pct": 0.3,

  "enable_adaptive_leverage": true,
  "min_leverage": 3,
  "consecutive_sl_threshold": 2,
  "pause_after_sl_streak": 3,
  "pause_hours": 24
}
```

### Key Parameters

- `leverage`: Base leverage (7x recommended, adaptive)
- `position_size_pct`: % of balance per trade (1.0 = 100%)
- `max_hold_hours`: Maximum hold time (64h = 16 bars on 4H)
- `sigma_entry`: Entry threshold (2.0σ)
- `sigma_exit`: Exit threshold (0.8σ)
- `sl_atr_mult`: Stop loss multiplier (1.8× ATR)
- `min_tp_pct`: Minimum TP filter (0.3%)

### Telegram Commands

If Telegram is configured:

- `/status` - Current position and bot status
- `/balance` - Account balance
- `/close` - Force close position
- `/pause` - Pause trading
- `/resume` - Resume trading

## Data Management

The bot automatically:

1. **Caches 4H bars** in `data/btcusdt_4h.csv`
2. **Downloads minimum 7 days** (~42 bars) on first run
3. **Updates incrementally** every 15 minutes
4. **Trims old data** (keeps max 84 bars / 14 days)
5. **Uses WebSocket** for real-time TP/SL tracking

## Logs

Logs are saved in `logs/trading_vwap.log`:

```bash
# Follow logs
./logs.sh

# Or with Docker
docker logs -f indicators_crypto_bot
```

## Safety Features

### Built-in Protections

1. **TP Filter**: Skips trades with TP < 0.3% (unprofitable with fees)
2. **Adaptive Leverage**: Reduces leverage on losing streaks
3. **Circuit Breaker**: Stops trading at 45% drawdown
4. **Trading Pause**: Pauses after 3 consecutive SL
5. **WebSocket Failover**: Falls back to REST API if WebSocket fails

### Demo Mode

**Always test on demo first!**

```bash
# In .env
BYBIT_DEMO_MODE=true  # Virtual money
```

Demo trading uses Bybit's unified account demo mode (same as mainnet, but virtual funds).

### Going Live

⚠️ **DANGER: Real money!**

1. Test thoroughly on demo for at least 1 month
2. Verify all signals and exits work correctly
3. Start with small capital ($100-1000)
4. Set `.env` to live mode:
   ```bash
   BYBIT_DEMO_MODE=false
   BYBIT_LIVE_API_KEY=your_live_key
   BYBIT_LIVE_SECRET=your_live_secret
   ```

## Monitoring

### Key Metrics to Watch

1. **Consecutive SL**: Should reset after wins
2. **Current Leverage**: Should adapt based on performance
3. **Drawdown**: Should stay below 35%
4. **Position Hold Time**: Should exit within 64h

### Telegram Notifications

The bot sends notifications for:

- ✅ Positions opened/closed
- ⚠️ Trading paused
- 🚨 Circuit breaker triggered
- 📊 Weekly performance reports

## Troubleshooting

### Bot won't start

1. Check Docker logs: `./logs.sh`
2. Verify API keys in `.env`
3. Check Bybit API status

### No signals generated

1. Check if sufficient data downloaded (7 days minimum)
2. Verify VWAP calculation in logs
3. May take 4-12 hours for first signal on 4H timeframe

### WebSocket disconnects

The bot automatically reconnects. If persistent:

1. Check network connection
2. Verify Bybit WebSocket status
3. Bot will fall back to REST API

## Files Structure

```
indicators_crypto_bot/
├── trading_bot/
│   ├── main.py                       # Main bot orchestrator
│   ├── data_manager_vwap.py          # 4H data caching
│   ├── signal_generator_vwap.py      # VWAP strategy logic
│   ├── exchange_connector.py         # Bybit API wrapper
│   ├── position_tracker.py           # Position management
│   ├── telegram_notifier.py          # Telegram notifications
│   ├── websocket_manager.py          # Real-time price tracking
│   └── trade_logger.py               # Trade history
├── config/
│   └── trading_params.json           # Strategy configuration
├── data/                             # Cached data & state
├── logs/                             # Log files
├── .env                              # API credentials
├── docker-compose.yml                # Docker configuration
└── *.sh                              # Management scripts
```

## Backtest Results

See `scripts/btc_research/st2_v3_analysis_2025/` for detailed analysis:

- `README.md` - Full results and analysis
- `trades_detailed.csv` - All 283 trades with fees
- `trade_timeline_interactive.html` - Interactive chart
- `FEE_CALCULATION.md` - Fee calculation methodology

## Support

For issues or questions:

1. Check logs: `./logs.sh`
2. Review configuration in `config/trading_params.json`
3. Verify API credentials in `.env`

## Disclaimer

**This bot trades real cryptocurrency with leverage. Use at your own risk.**

- Past performance does not guarantee future results
- Cryptocurrency trading is highly risky
- Leverage amplifies both gains and losses
- Always test on demo first
- Never trade more than you can afford to lose

---

_Strategy: ST2 VWAP Mean Reversion v3_
_Created: 2025-12-10_
_Win Rate: 97.2% (2025 backtest)_
