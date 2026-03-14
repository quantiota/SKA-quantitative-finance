"""
SKA Paired Regime Trading Bot

Changelog:
  v1: Single opposite-side transition closes the position
  v2: Requires 2 consecutive opposite-side signals to close (any two)
  v3: Requires the complete opposite paired cycle to close
      - LONG closes only when full DOWN cycle confirmed: neutral→bear + bear→neutral
      - SHORT closes only when full UP cycle confirmed: neutral→bull + bull→neutral
      - UP pair must be confirmed (bull→neutral) before listening for DOWN cycle
      - DOWN pair must be confirmed (bear→neutral) before listening for UP cycle

Signal logic — full cycle:

  LONG:
    neutral→bull               (OPEN LONG — UP cycle opens)
    bull→neutral               (UP pair confirmed — now listen for DOWN cycle)
    neutral→bear               (DOWN cycle opens)
    bear→neutral               (DOWN pair confirmed — CLOSE LONG)

  SHORT:
    neutral→bear               (OPEN SHORT — DOWN cycle opens)
    bear→neutral               (DOWN pair confirmed — now listen for UP cycle)
    neutral→bull               (UP cycle opens)
    bull→neutral               (UP pair confirmed — CLOSE SHORT)

State machine per position:
  WAIT_UP_CONFIRM    → waiting for bull→neutral after OPEN LONG
  WAIT_DOWN_OPEN     → waiting for neutral→bear after UP confirmed
  WAIT_DOWN_CONFIRM  → waiting for bear→neutral to close LONG

  WAIT_DOWN_CONFIRM  → waiting for bear→neutral after OPEN SHORT
  WAIT_UP_OPEN       → waiting for neutral→bull after DOWN confirmed
  WAIT_UP_CONFIRM    → waiting for bull→neutral to close SHORT

The alpha: price follows paired regime transitions.
Only complete opposite paired cycles trigger exit.
"""

import asyncio
import csv
import logging
import asyncpg
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Exit state machine states
WAIT_PAIR_CONFIRM  = 'WAIT_PAIR_CONFIRM'   # waiting for own pair confirmation
WAIT_OPP_OPEN      = 'WAIT_OPP_OPEN'       # waiting for opposite cycle to open
WAIT_OPP_CONFIRM   = 'WAIT_OPP_CONFIRM'    # waiting for opposite cycle to confirm → EXIT


@dataclass
class Position:
    side: str               # 'LONG' or 'SHORT'
    entry_price: float
    entry_trade_id: int
    entry_time: str
    entry_transition: str
    exit_state: str = field(default=WAIT_PAIR_CONFIRM)


class SKATradingBot:
    """Paired regime transition bot v3 — full opposite cycle required to close."""

    def __init__(self, db_host='192.168.1.216', db_port=8812, symbol='XRPUSDT',
                 poll_interval=1.0, dry_run=True):
        self.db_host = db_host
        self.db_port = db_port
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.conn = None
        self.position: Optional[Position] = None
        self.last_trade_id = None

        # Results file (created at init, appended per trade)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = f'/home/coder/project/Real_Time_SKA_trading/bot_results_v3/bot_results_v3_{ts}.csv'

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.trade_log = []

    async def connect(self):
        self.conn = await asyncpg.connect(
            host=self.db_host, port=self.db_port,
            database='qdb', user='admin', password='quest'
        )
        logging.info(f"Connected to QuestDB at {self.db_host}:{self.db_port}")

    async def get_latest_transition(self):
        query = """
        WITH base_data AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy,
            LAG(price)   OVER (ORDER BY timestamp, trade_id) AS prev_price,
            LAG(entropy) OVER (ORDER BY timestamp, trade_id) AS prev_entropy
          FROM binance_trades
          WHERE symbol = $1 AND entropy IS NOT NULL
          ORDER BY timestamp DESC
          LIMIT 100
        ),
        with_regime AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy, prev_entropy,
            CASE
              WHEN price - prev_price > 0 THEN 1
              WHEN price - prev_price < 0 THEN 2
              ELSE 0
            END AS regime
          FROM base_data
          WHERE prev_price IS NOT NULL
        ),
        with_transition AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy, prev_entropy, regime,
            LAG(regime) OVER (ORDER BY timestamp, trade_id) AS prev_regime
          FROM with_regime
        ),
        with_probability AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy, regime, prev_regime,
            prev_regime * 3 + regime AS transition_code,
            CASE
              WHEN entropy != 0 AND prev_entropy IS NOT NULL
              THEN EXP(-ABS((entropy - prev_entropy) / entropy))
              ELSE NULL
            END AS prob
          FROM with_transition
          WHERE prev_regime IS NOT NULL
        )
        SELECT timestamp, trade_id, price, prob, transition_code
        FROM with_probability
        ORDER BY timestamp DESC, trade_id DESC
        LIMIT 1
        """
        try:
            row = await self.conn.fetchrow(query, self.symbol)
        except Exception:
            return None
        if row is None:
            return None

        code = int(row['transition_code'])
        names = {
            0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
            3: 'bull→neutral',    4: 'bull→bull',    5: 'bull→bear',
            6: 'bear→neutral',    7: 'bear→bull',    8: 'bear→bear'
        }
        return {
            'timestamp': row['timestamp'],
            'trade_id': int(row['trade_id']),
            'price': float(row['price']),
            'P': float(row['prob']) if row['prob'] is not None else None,
            'transition_code': code,
            'transition_name': names.get(code, 'unknown')
        }

    def process_signal(self, transition):
        """
        State machine:

        LONG exit states:
          WAIT_PAIR_CONFIRM  → bull→neutral         → WAIT_OPP_OPEN
          WAIT_OPP_OPEN      → neutral→bear          → WAIT_OPP_CONFIRM
          WAIT_OPP_CONFIRM   → bear→neutral          → CLOSE LONG

        SHORT exit states:
          WAIT_PAIR_CONFIRM  → bear→neutral          → WAIT_OPP_OPEN
          WAIT_OPP_OPEN      → neutral→bull          → WAIT_OPP_CONFIRM
          WAIT_OPP_CONFIRM   → bull→neutral          → CLOSE SHORT
        """
        trade_id = transition['trade_id']
        price    = transition['price']
        name     = transition['transition_name']
        P        = transition['P']
        ts       = transition['timestamp']

        if self.last_trade_id is not None and trade_id <= self.last_trade_id:
            return
        self.last_trade_id = trade_id

        # === NO POSITION: look for entry ===
        if self.position is None:
            if name == 'neutral→bull':
                self.position = Position(
                    side='LONG', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name,
                    exit_state=WAIT_PAIR_CONFIRM
                )
                logging.info(
                    f">>> OPEN LONG @ {price:.6f} | P={P:.4f} | trade_id={trade_id} "
                    f"| waiting: bull→neutral"
                )
                if not self.dry_run:
                    self._execute_buy(price)

            elif name == 'neutral→bear':
                self.position = Position(
                    side='SHORT', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name,
                    exit_state=WAIT_PAIR_CONFIRM
                )
                logging.info(
                    f">>> OPEN SHORT @ {price:.6f} | P={P:.4f} | trade_id={trade_id} "
                    f"| waiting: bear→neutral"
                )
                if not self.dry_run:
                    self._execute_sell(price)
            return

        # === LONG POSITION — state machine ===
        if self.position.side == 'LONG':

            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bull→neutral':
                    self.position.exit_state = WAIT_OPP_OPEN
                    logging.info(
                        f"--- UP pair confirmed (bull→neutral) @ {price:.6f} "
                        f"| waiting: neutral→bear"
                    )

            elif self.position.exit_state == WAIT_OPP_OPEN:
                if name == 'neutral→bear':
                    self.position.exit_state = WAIT_OPP_CONFIRM
                    logging.info(
                        f"--- DOWN cycle opens (neutral→bear) @ {price:.6f} "
                        f"| waiting: bear→neutral"
                    )

            elif self.position.exit_state == WAIT_OPP_CONFIRM:
                if name == 'bear→neutral':
                    pnl = price - self.position.entry_price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE LONG (DOWN pair confirmed) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    if not self.dry_run:
                        self._execute_sell(price)
                    self.position = None

                    # DOWN cycle just completed — open SHORT on the same signal
                    self.position = Position(
                        side='SHORT', entry_price=price,
                        entry_trade_id=trade_id, entry_time=str(ts),
                        entry_transition='bear→neutral',
                        exit_state=WAIT_OPP_OPEN
                    )
                    logging.info(
                        f">>> OPEN SHORT (new cycle) @ {price:.6f} "
                        f"| waiting: neutral→bull"
                    )
                    if not self.dry_run:
                        self._execute_sell(price)

        # === SHORT POSITION — state machine ===
        elif self.position.side == 'SHORT':

            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bear→neutral':
                    self.position.exit_state = WAIT_OPP_OPEN
                    logging.info(
                        f"--- DOWN pair confirmed (bear→neutral) @ {price:.6f} "
                        f"| waiting: neutral→bull"
                    )

            elif self.position.exit_state == WAIT_OPP_OPEN:
                if name == 'neutral→bull':
                    self.position.exit_state = WAIT_OPP_CONFIRM
                    logging.info(
                        f"--- UP cycle opens (neutral→bull) @ {price:.6f} "
                        f"| waiting: bull→neutral"
                    )

            elif self.position.exit_state == WAIT_OPP_CONFIRM:
                if name == 'bull→neutral':
                    pnl = self.position.entry_price - price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE SHORT (UP pair confirmed) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    if not self.dry_run:
                        self._execute_buy(price)
                    self.position = None

                    # UP cycle just completed — open LONG on the same signal
                    self.position = Position(
                        side='LONG', entry_price=price,
                        entry_trade_id=trade_id, entry_time=str(ts),
                        entry_transition='bull→neutral',
                        exit_state=WAIT_OPP_OPEN
                    )
                    logging.info(
                        f">>> OPEN LONG (new cycle) @ {price:.6f} "
                        f"| waiting: neutral→bear"
                    )
                    if not self.dry_run:
                        self._execute_buy(price)

    def _record_trade(self, pnl, pnl_pct, exit_price):
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        trade = {
            'side': self.position.side,
            'entry': self.position.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_transition': self.position.entry_transition
        }
        self.trade_log.append(trade)

        # Append to CSV immediately (survives SIGKILL)
        header = not hasattr(self, '_csv_written')
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'side', 'entry', 'exit', 'pnl', 'pnl_pct', 'entry_transition'
            ])
            if header:
                writer.writeheader()
                self._csv_written = True
            writer.writerow(trade)

    def _execute_buy(self, price):
        logging.info(f"[EXECUTE] BUY {self.symbol} @ {price:.6f}")

    def _execute_sell(self, price):
        logging.info(f"[EXECUTE] SELL {self.symbol} @ {price:.6f}")

    def print_stats(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logging.info(
            f"=== STATS === Trades: {self.total_trades} | "
            f"Win: {self.winning_trades} | Lose: {self.losing_trades} | "
            f"Win rate: {win_rate:.1f}% | Total PnL: {self.total_pnl:+.6f}"
        )

    def save_results(self):
        """Log final summary (trades saved live to self.results_file)."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logging.info(f"Results file: {self.results_file}")
        logging.info(f"Summary: {self.total_trades} trades | win_rate={win_rate:.1f}% | PnL={self.total_pnl:+.6f}")

    async def get_entropy_count(self):
        try:
            row = await self.conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM binance_trades WHERE symbol = $1 AND entropy IS NOT NULL",
                self.symbol
            )
            return int(row['cnt']) if row else 0
        except Exception:
            return 0

    async def run(self, max_trades=3500):
        await self.connect()
        logging.info(f"SKA Trading Bot v3 | symbol={self.symbol} | dry_run={self.dry_run} | auto_stop={max_trades}")
        logging.info("LONG: neutral→bull → bull→neutral → neutral→bear → bear→neutral (CLOSE)")
        logging.info("SHORT: neutral→bear → bear→neutral → neutral→bull → bull→neutral (CLOSE)")

        min_trades = 500
        logging.info(f"Waiting for {min_trades} trades with entropy before trading...")
        while True:
            count = await self.get_entropy_count()
            if count >= min_trades:
                logging.info(f"SKA ready: {count} trades with entropy. Starting signals.")
                break
            await asyncio.sleep(self.poll_interval)

        try:
            while True:
                transition = await self.get_latest_transition()
                if transition:
                    self.process_signal(transition)

                count = await self.get_entropy_count()
                if count >= max_trades:
                    logging.info(f"Auto-stop: {count} trades with entropy >= {max_trades}")
                    break

                await asyncio.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        finally:
            if self.position:
                transition = await self.get_latest_transition()
                close_price = transition['price'] if transition else self.position.entry_price
                if self.position.side == 'LONG':
                    pnl = close_price - self.position.entry_price
                else:
                    pnl = self.position.entry_price - close_price
                pnl_pct = (pnl / self.position.entry_price) * 100
                self._record_trade(pnl, pnl_pct, close_price)
                logging.info(
                    f"<<< CLOSE {self.position.side} (end of run) @ {close_price:.6f} | "
                    f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | exit_state={self.position.exit_state}"
                )
                self.position = None

            self.print_stats()
            self.save_results()
            if self.conn:
                await self.conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SKA Paired Regime Trading Bot v3')
    parser.add_argument('--symbol', default='XRPUSDT', help='Trading symbol')
    parser.add_argument('--poll', type=float, default=1.0, help='Poll interval (seconds)')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: dry run)')
    parser.add_argument('--host', default='192.168.1.216', help='QuestDB host')
    args = parser.parse_args()

    bot = SKATradingBot(
        db_host=args.host,
        symbol=args.symbol,
        poll_interval=args.poll,
        dry_run=not args.live
    )
    asyncio.run(bot.run())
