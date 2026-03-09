"""
SKA Paired Regime Trading Bot

Changelog:
  v1: Single opposite-side transition closes the position
  v2: Requires 2 consecutive opposite-side confirmations to close
      - Same-side signal resets the exit counter
      - Filters single-transition noise

Signal logic — full cycle:

  UP cycle:   neutral→bull → bull→neutral → ... → neutral→bear OR bear→neutral  (CLOSE LONG)
  DOWN cycle: neutral→bear → bear→neutral → ... → neutral→bull OR bull→neutral  (CLOSE SHORT)

  Entry: first transition of the pair (neutral→bull or neutral→bear)
  Stay:  bull→neutral or bear→neutral confirms the pair, position stays open
  Exit:  opposite side appears = cycle complete

The alpha: price follows paired regime transitions.
"""

import asyncio
import csv
import logging
import time
import asyncpg
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


@dataclass
class Position:
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_trade_id: int
    entry_time: str
    entry_transition: str


class SKATradingBot:
    """Paired regime transition bot — polls QuestDB, detects transitions, logs signals."""

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
        self.results_file = f'/home/coder/project/Real_Time_SKA_trading/bot_results_v2_{ts}.csv'

        # Cycle tracking
        self.pair_confirmed = False  # True after bull→neutral (long) or bear→neutral (short)
        self.exit_confirmations = 0  # Count consecutive opposite-side signals (need 2 to close)

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
        """
        Get the most recent regime transition from QuestDB.
        Uses the same CTE logic as grafana_probability_queries.sql.
        Returns the last row with transition info.
        """
        query = """
        WITH base_data AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy,
            LAG(price) OVER (ORDER BY timestamp, trade_id) AS prev_price,
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
        SELECT
          timestamp, trade_id, price, prob, transition_code
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

        # Decode transition
        code = int(row['transition_code'])
        names = {
            0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
            3: 'bull→neutral', 4: 'bull→bull', 5: 'bull→bear',
            6: 'bear→neutral', 7: 'bear→bull', 8: 'bear→bear'
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
        Process a transition using full cycle logic.

        UP cycle:   neutral→bull (OPEN LONG) → bull→neutral (pair confirmed, HOLD)
                    → neutral→bear OR bear→neutral (cycle complete, CLOSE LONG)

        DOWN cycle: neutral→bear (OPEN SHORT) → bear→neutral (pair confirmed, HOLD)
                    → neutral→bull OR bull→neutral (cycle complete, CLOSE SHORT)
        """
        trade_id = transition['trade_id']
        price = transition['price']
        name = transition['transition_name']
        P = transition['P']
        ts = transition['timestamp']

        # Skip if we already processed this trade
        if self.last_trade_id is not None and trade_id <= self.last_trade_id:
            return
        self.last_trade_id = trade_id

        # === NO POSITION: look for entry ===
        if self.position is None:
            if name == 'neutral→bull':
                self.position = Position(
                    side='LONG', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name
                )
                self.pair_confirmed = False
                self.exit_confirmations = 0
                logging.info(
                    f">>> OPEN LONG @ {price:.6f} | P={P:.4f} | trade_id={trade_id}"
                )
                if not self.dry_run:
                    self._execute_buy(price)

            elif name == 'neutral→bear':
                self.position = Position(
                    side='SHORT', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name
                )
                self.pair_confirmed = False
                self.exit_confirmations = 0
                logging.info(
                    f">>> OPEN SHORT @ {price:.6f} | P={P:.4f} | trade_id={trade_id}"
                )
                if not self.dry_run:
                    self._execute_sell(price)
            return

        # === LONG POSITION ===
        if self.position.side == 'LONG':
            # bull→neutral: pair confirmed, stay in position
            if name == 'bull→neutral':
                self.pair_confirmed = True
                self.exit_confirmations = 0  # reset exit count on same-side signal
                logging.info(
                    f"--- PAIR CONFIRMED (bull→neutral) @ {price:.6f} | HOLD LONG | trade_id={trade_id}"
                )

            # Opposite side appears: count confirmation
            elif name in ('neutral→bear', 'bear→neutral'):
                self.exit_confirmations += 1
                logging.info(
                    f"--- EXIT SIGNAL {self.exit_confirmations}/2 ({name}) @ {price:.6f} | trade_id={trade_id}"
                )

                if self.exit_confirmations >= 2:
                    pnl = price - self.position.entry_price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE LONG (2 confirmations) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    if not self.dry_run:
                        self._execute_sell(price)
                    self.position = None
                    self.pair_confirmed = False
                    self.exit_confirmations = 0

            # Same-side signal resets exit count
            elif name in ('neutral→bull', 'bull→bull'):
                self.exit_confirmations = 0

        # === SHORT POSITION ===
        elif self.position.side == 'SHORT':
            # bear→neutral: pair confirmed, stay in position
            if name == 'bear→neutral':
                self.pair_confirmed = True
                self.exit_confirmations = 0  # reset exit count on same-side signal
                logging.info(
                    f"--- PAIR CONFIRMED (bear→neutral) @ {price:.6f} | HOLD SHORT | trade_id={trade_id}"
                )

            # Opposite side appears: count confirmation
            elif name in ('neutral→bull', 'bull→neutral'):
                self.exit_confirmations += 1
                logging.info(
                    f"--- EXIT SIGNAL {self.exit_confirmations}/2 ({name}) @ {price:.6f} | trade_id={trade_id}"
                )

                if self.exit_confirmations >= 2:
                    pnl = self.position.entry_price - price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE SHORT (2 confirmations) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    if not self.dry_run:
                        self._execute_buy(price)
                    self.position = None
                    self.pair_confirmed = False
                    self.exit_confirmations = 0

            # Same-side signal resets exit count
            elif name in ('neutral→bear', 'bear→bear'):
                self.exit_confirmations = 0

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
        """Placeholder for Binance API buy order"""
        logging.info(f"[EXECUTE] BUY {self.symbol} @ {price:.6f}")

    def _execute_sell(self, price):
        """Placeholder for Binance API sell order"""
        logging.info(f"[EXECUTE] SELL {self.symbol} @ {price:.6f}")

    def print_stats(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logging.info(
            f"=== STATS === Trades: {self.total_trades} | "
            f"Win: {self.winning_trades} | Lose: {self.losing_trades} | "
            f"Win rate: {win_rate:.1f}% | Total PnL: {self.total_pnl:+.6f}"
        )

    def save_results(self):
        """Save trade log and stats to CSV."""
        if not self.trade_log:
            logging.info("No trades to save")
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        outfile = f'/home/coder/project/Real_Time_SKA_trading/bot_results_v1_{ts}.csv'

        with open(outfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'side', 'entry', 'exit', 'pnl', 'pnl_pct', 'entry_transition'
            ])
            writer.writeheader()
            writer.writerows(self.trade_log)

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logging.info(f"Results saved to {outfile}")
        logging.info(f"Summary: {self.total_trades} trades | win_rate={win_rate:.1f}% | PnL={self.total_pnl:+.6f}")

    async def get_entropy_count(self):
        """Count trades with entropy computed (= learning engine progress)."""
        try:
            row = await self.conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM binance_trades WHERE symbol = $1 AND entropy IS NOT NULL",
                self.symbol
            )
            return int(row['cnt']) if row else 0
        except Exception:
            # Table may not exist yet (DB was just reset)
            return 0

    async def run(self, max_trades=3500):
        """Main polling loop. Auto-stops when learning engine reaches max_trades."""
        await self.connect()
        logging.info(f"SKA Trading Bot v1 started | symbol={self.symbol} | dry_run={self.dry_run} | auto_stop={max_trades}")
        logging.info("UP cycle:   neutral→bull (OPEN) → bull→neutral (HOLD) → bear side (CLOSE)")
        logging.info("DOWN cycle: neutral→bear (OPEN) → bear→neutral (HOLD) → bull side (CLOSE)")

        # Wait for SKA to learn before trading
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

                # Auto-stop: check if learning engine finished
                count = await self.get_entropy_count()
                if count >= max_trades:
                    logging.info(f"Auto-stop: {count} trades with entropy >= {max_trades}")
                    break

                await asyncio.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        finally:
            # Close open position at last known price
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
                    f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%)"
                )
                self.position = None

            self.print_stats()
            self.save_results()
            if self.conn:
                await self.conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SKA Paired Regime Trading Bot')
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
