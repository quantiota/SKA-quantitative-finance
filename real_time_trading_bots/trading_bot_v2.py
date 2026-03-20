"""
Paired cycle trading bot v2 — entropy-derived probability regime transitions (ΔP).

Regime definition:
  P(n)   = exp(-|ΔH/H|)   where  ΔH/H = (H(n) - H(n-1)) / H(n)
  ΔP(n)  = P(n) - P(n-1)  consecutive change in probability

  ΔP(n) < -BEAR_THRESHOLD              →  regime = 2  ("bear"    — large P drop)
  -BEAR_THRESHOLD ≤ ΔP(n) < -BULL_THRESHOLD  →  regime = 1  ("bull" — moderate P drop)
  ΔP(n) ≥ -BULL_THRESHOLD              →  regime = 0  (neutral)

  BULL_THRESHOLD = 0.148  # observed ΔP gap constant within bull paired regime transitions
  BEAR_THRESHOLD = 0.221  # observed ΔP gap constant within bear paired regime transitions

  ΔP across a paired transition (opening → closing):
    bull pair : ΔP < 0  (P drifts lower — sustained entropy change)
    bear pair : ΔP > 0  (P snaps back  — brief entropy shock)

Signal logic:

  LONG:
    neutral→bull               (OPEN LONG — WAIT_PAIR)
    bull→neutral               (pair confirmed — IN_NEUTRAL)
    neutral→neutral × N        (neutral gap — stay IN_NEUTRAL)
    <first non-neutral>        (gap closes — READY)
    neutral→bull               (cycle repeats — WAIT_PAIR)
    neutral→bear               (opposite cycle opens — EXIT_WAIT)
    bear→neutral               (opposite pair confirmed — CLOSE LONG)

  SHORT:
    neutral→bear               (OPEN SHORT — WAIT_PAIR)
    bear→neutral               (pair confirmed — IN_NEUTRAL)
    neutral→neutral × N        (neutral gap — stay IN_NEUTRAL)
    <first non-neutral>        (gap closes — READY)
    neutral→bear               (cycle repeats — WAIT_PAIR)
    neutral→bull               (opposite cycle opens — EXIT_WAIT)
    bull→neutral               (opposite pair confirmed — CLOSE SHORT)

State machine per position:
  WAIT_PAIR   → waiting for own pair confirmation
  IN_NEUTRAL  → pair confirmed, counting neutral→neutral gap
  READY       → gap closed, listening for next cycle or opposite open
  EXIT_WAIT   → opposite cycle opened, waiting for opposite pair confirmation
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

WAIT_PAIR  = 'WAIT_PAIR'
IN_NEUTRAL = 'IN_NEUTRAL'
READY      = 'READY'
EXIT_WAIT  = 'EXIT_WAIT'

MIN_NN_COUNT      = 3
BULL_THRESHOLD = 0.148  # bull pair ΔP constant −10%
BEAR_THRESHOLD = 0.221  # bear pair ΔP constant −10%

EVENT = {
    'OPEN_LONG':      1,
    'OPEN_SHORT':     2,
    'CLOSE_LONG':     3,
    'CLOSE_SHORT':    4,
    'PAIR_CONFIRMED': 5,
    'NEUTRAL_GAP':    6,
    'CYCLE_REPEAT':   7,
}
STATE = {
    WAIT_PAIR:  1,
    IN_NEUTRAL: 2,
    READY:      3,
    EXIT_WAIT:  4,
}
SIDE = {
    'LONG':  1,
    'SHORT': 2,
    '':      0,
}


@dataclass
class Position:
    side: str
    entry_price: float
    entry_trade_id: int
    entry_time: str
    entry_transition: str
    exit_state: str = field(default=WAIT_PAIR)
    neutral_neutral_count: int = field(default=0)


class SKATradingBot:
    """Paired cycle trading bot v2 — entropy-based regime transitions (ΔH/H)."""

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

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = f'/home/coder/project/Real_Time_SKA_trading/bot_results_v2/bot_results_v2_{ts}.csv'

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
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ska_bot_v2 (
                timestamp            TIMESTAMP,
                trade_id             LONG,
                price                DOUBLE,
                event                INT,
                event_name           VARCHAR,
                state                INT,
                state_name           VARCHAR,
                side                 INT,
                side_name            VARCHAR,
                pnl                  DOUBLE,
                neutral_neutral_count INT
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)
        logging.info("ska_bot_v2 table ready")

    async def _log_event(self, trade_id, price, event, state, side='', pnl=None, neutral_neutral_count=None):
        try:
            await self.conn.execute(
                """INSERT INTO ska_bot_v2
                   (timestamp, trade_id, price, event, event_name, state, state_name, side, side_name, pnl, neutral_neutral_count)
                   VALUES (now(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                trade_id, price,
                EVENT[event], event,
                STATE[state], state,
                SIDE[side], side if side else None,
                pnl, neutral_neutral_count
            )
        except Exception as e:
            logging.warning(f"State log failed: {e}")

    async def get_new_transitions(self):
        last_id = self.last_trade_id if self.last_trade_id is not None else 0
        query = """
        WITH base AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy,
            LAG(entropy, 1) OVER (ORDER BY timestamp, trade_id) AS e1,
            LAG(entropy, 2) OVER (ORDER BY timestamp, trade_id) AS e2
          FROM binance_trades
          WHERE symbol = $1 AND entropy IS NOT NULL AND trade_id >= $2 - 2
          ORDER BY timestamp, trade_id
        ),
        with_P AS (
          SELECT
            timestamp, symbol, trade_id, price,
            CASE WHEN entropy != 0 AND e1 IS NOT NULL
                 THEN EXP(-ABS((entropy - e1) / entropy)) ELSE NULL END AS P,
            CASE WHEN e1 != 0 AND e2 IS NOT NULL
                 THEN EXP(-ABS((e1 - e2) / e1)) ELSE NULL END AS prev_P
          FROM base WHERE e1 IS NOT NULL
        ),
        with_regime AS (
          SELECT
            timestamp, symbol, trade_id, price, P,
            CASE
              WHEN prev_P IS NOT NULL AND (P - prev_P) < -$4 THEN 2
              WHEN prev_P IS NOT NULL AND (P - prev_P) < -$3 THEN 1
              ELSE 0
            END AS regime
          FROM with_P WHERE P IS NOT NULL AND prev_P IS NOT NULL
        ),
        with_transition AS (
          SELECT
            timestamp, symbol, trade_id, price, P, regime,
            LAG(regime) OVER (ORDER BY timestamp, trade_id) AS prev_regime
          FROM with_regime
        )
        SELECT timestamp, trade_id, price, P AS prob, prev_regime * 3 + regime AS transition_code
        FROM with_transition
        WHERE prev_regime IS NOT NULL AND trade_id > $2
        ORDER BY timestamp, trade_id
        """
        names = {
            0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
            3: 'bull→neutral',    4: 'bull→bull',    5: 'bull→bear',
            6: 'bear→neutral',    7: 'bear→bull',    8: 'bear→bear'
        }
        try:
            rows = await self.conn.fetch(query, self.symbol, last_id, BULL_THRESHOLD, BEAR_THRESHOLD)
        except Exception:
            return []
        result = []
        for row in rows:
            code = int(row['transition_code'])
            result.append({
                'timestamp': row['timestamp'],
                'trade_id': int(row['trade_id']),
                'price': float(row['price']),
                'P': float(row['prob']) if row['prob'] is not None else None,
                'transition_code': code,
                'transition_name': names.get(code, 'unknown')
            })
        return result

    async def process_signal(self, transition):
        trade_id = transition['trade_id']
        price    = transition['price']
        name     = transition['transition_name']
        P        = transition['P']
        ts       = transition['timestamp']

        if self.last_trade_id is not None and trade_id <= self.last_trade_id:
            return
        self.last_trade_id = trade_id

        p_str = f"{P:.4f}" if P is not None else "n/a"

        # === NO POSITION: look for entry ===
        if self.position is None:
            if name == 'neutral→bull':
                self.position = Position(
                    side='LONG', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name, exit_state=WAIT_PAIR
                )
                logging.info(
                    f">>> OPEN LONG @ {price:.6f} | P={p_str} | trade_id={trade_id} "
                    f"| waiting: bull→neutral"
                )
                await self._log_event(trade_id, price, 'OPEN_LONG', WAIT_PAIR, 'LONG')
                if not self.dry_run:
                    self._execute_buy(price)

            elif name == 'neutral→bear':
                self.position = Position(
                    side='SHORT', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name, exit_state=WAIT_PAIR
                )
                logging.info(
                    f">>> OPEN SHORT @ {price:.6f} | P={p_str} | trade_id={trade_id} "
                    f"| waiting: bear→neutral"
                )
                await self._log_event(trade_id, price, 'OPEN_SHORT', WAIT_PAIR, 'SHORT')
                if not self.dry_run:
                    self._execute_sell(price)
            return

        # === LONG POSITION ===
        if self.position.side == 'LONG':

            if self.position.exit_state == WAIT_PAIR:
                if name == 'bull→neutral':
                    self.position.exit_state = IN_NEUTRAL
                    logging.info(
                        f"--- UP pair confirmed (bull→neutral) @ {price:.6f} "
                        f"| IN_NEUTRAL | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'PAIR_CONFIRMED', IN_NEUTRAL, 'LONG')

            elif self.position.exit_state == IN_NEUTRAL:
                if name == 'neutral→neutral':
                    self.position.neutral_neutral_count += 1
                    logging.info(
                        f"--- Neutral gap nn_count={self.position.neutral_neutral_count} @ {price:.6f} "
                        f"| IN_NEUTRAL | trade_id={trade_id}"
                    )
                else:
                    if self.position.neutral_neutral_count >= MIN_NN_COUNT:
                        self.position.exit_state = READY
                        logging.info(
                            f"--- Neutral gap closed ({name}) @ {price:.6f} "
                            f"| READY | nn_count={self.position.neutral_neutral_count} | trade_id={trade_id}"
                        )
                        await self._log_event(trade_id, price, 'NEUTRAL_GAP', READY, 'LONG',
                                              neutral_neutral_count=self.position.neutral_neutral_count)
                    else:
                        logging.info(
                            f"--- Neutral gap too short nn_count={self.position.neutral_neutral_count} "
                            f"(min={MIN_NN_COUNT}) — reset | trade_id={trade_id}"
                        )
                        self.position.neutral_neutral_count = 0

            elif self.position.exit_state == READY:
                if name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- UP cycle repeating (neutral→bull) @ {price:.6f} "
                        f"| WAIT_PAIR | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'LONG')
                elif name == 'neutral→bear':
                    self.position.exit_state = EXIT_WAIT
                    logging.info(
                        f"--- Opposite cycle opening (neutral→bear) @ {price:.6f} "
                        f"| EXIT_WAIT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'NEUTRAL_GAP', EXIT_WAIT, 'LONG')

            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bear→neutral':
                    pnl = price - self.position.entry_price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE LONG (bear→neutral) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    await self._log_event(trade_id, price, 'CLOSE_LONG', EXIT_WAIT, 'LONG', pnl)
                    if not self.dry_run:
                        self._execute_sell(price)
                    self.position = Position(
                        side='SHORT', entry_price=price,
                        entry_trade_id=trade_id, entry_time=str(ts),
                        entry_transition='neutral→bear', exit_state=WAIT_PAIR
                    )
                    logging.info(
                        f">>> OPEN SHORT (new cycle) @ {price:.6f} "
                        f"| waiting: bear→neutral"
                    )
                    await self._log_event(trade_id, price, 'OPEN_SHORT', WAIT_PAIR, 'SHORT')
                    if not self.dry_run:
                        self._execute_sell(price)
                elif name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- Bear cycle aborted (neutral→bull) @ {price:.6f} "
                        f"| WAIT_PAIR | still LONG | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'LONG')

        # === SHORT POSITION ===
        elif self.position.side == 'SHORT':

            if self.position.exit_state == WAIT_PAIR:
                if name == 'bear→neutral':
                    self.position.exit_state = IN_NEUTRAL
                    logging.info(
                        f"--- DOWN pair confirmed (bear→neutral) @ {price:.6f} "
                        f"| IN_NEUTRAL | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'PAIR_CONFIRMED', IN_NEUTRAL, 'SHORT')

            elif self.position.exit_state == IN_NEUTRAL:
                if name == 'neutral→neutral':
                    self.position.neutral_neutral_count += 1
                    logging.info(
                        f"--- Neutral gap nn_count={self.position.neutral_neutral_count} @ {price:.6f} "
                        f"| IN_NEUTRAL | trade_id={trade_id}"
                    )
                else:
                    if self.position.neutral_neutral_count >= MIN_NN_COUNT:
                        self.position.exit_state = READY
                        logging.info(
                            f"--- Neutral gap closed ({name}) @ {price:.6f} "
                            f"| READY | nn_count={self.position.neutral_neutral_count} | trade_id={trade_id}"
                        )
                        await self._log_event(trade_id, price, 'NEUTRAL_GAP', READY, 'SHORT',
                                              neutral_neutral_count=self.position.neutral_neutral_count)
                    else:
                        logging.info(
                            f"--- Neutral gap too short nn_count={self.position.neutral_neutral_count} "
                            f"(min={MIN_NN_COUNT}) — reset | trade_id={trade_id}"
                        )
                        self.position.neutral_neutral_count = 0

            elif self.position.exit_state == READY:
                if name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- DOWN cycle repeating (neutral→bear) @ {price:.6f} "
                        f"| WAIT_PAIR | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'SHORT')
                elif name == 'neutral→bull':
                    self.position.exit_state = EXIT_WAIT
                    logging.info(
                        f"--- Opposite cycle opening (neutral→bull) @ {price:.6f} "
                        f"| EXIT_WAIT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'NEUTRAL_GAP', EXIT_WAIT, 'SHORT')

            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bull→neutral':
                    pnl = self.position.entry_price - price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE SHORT (bull→neutral) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    await self._log_event(trade_id, price, 'CLOSE_SHORT', EXIT_WAIT, 'SHORT', pnl)
                    if not self.dry_run:
                        self._execute_buy(price)
                    self.position = Position(
                        side='LONG', entry_price=price,
                        entry_trade_id=trade_id, entry_time=str(ts),
                        entry_transition='neutral→bull', exit_state=WAIT_PAIR
                    )
                    logging.info(
                        f">>> OPEN LONG (new cycle) @ {price:.6f} "
                        f"| waiting: bull→neutral"
                    )
                    await self._log_event(trade_id, price, 'OPEN_LONG', WAIT_PAIR, 'LONG')
                    if not self.dry_run:
                        self._execute_buy(price)
                elif name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- Bull cycle aborted (neutral→bear) @ {price:.6f} "
                        f"| WAIT_PAIR | still SHORT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'SHORT')

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
            'entry_transition': self.position.entry_transition
        }
        self.trade_log.append(trade)

        header = not hasattr(self, '_csv_written')
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'side', 'entry', 'exit', 'pnl', 'entry_transition'
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
        logging.info(f"SKA Trading Bot v2 | symbol={self.symbol} | dry_run={self.dry_run} | auto_stop={max_trades}")
        logging.info(f"Regime: ΔP < -{BEAR_THRESHOLD} → bear | -{BEAR_THRESHOLD} ≤ ΔP < -{BULL_THRESHOLD} → bull | else → neutral")
        logging.info("LONG:  neutral→bull → bull→neutral → neutral→neutral × N → neutral→bear → bear→neutral (CLOSE)")
        logging.info("SHORT: neutral→bear → bear→neutral → neutral→neutral × N → neutral→bull → bull→neutral (CLOSE)")

        min_trades = 0
        logging.info(f"Waiting for {min_trades} trades with entropy before trading...")
        while True:
            count = await self.get_entropy_count()
            if count >= min_trades:
                logging.info(f"SKA ready: {count} trades with entropy. Starting signals.")
                break
            await asyncio.sleep(self.poll_interval)

        try:
            while True:
                transitions = await self.get_new_transitions()
                for transition in transitions:
                    await self.process_signal(transition)

                count = await self.get_entropy_count()
                if count >= max_trades:
                    logging.info(f"Auto-stop: {count} trades with entropy >= {max_trades}")
                    break

                await asyncio.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        finally:
            if self.position:
                transitions = await self.get_new_transitions()
                close_price = transitions[-1]['price'] if transitions else self.position.entry_price
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
    parser = argparse.ArgumentParser(description='SKA Paired Regime Trading Bot v2')
    parser.add_argument('--symbol', default='XRPUSDT', help='Trading symbol')
    parser.add_argument('--poll', type=float, default=1.0, help='Poll interval (seconds)')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: dry run)')
    parser.add_argument('--host', default='192.168.1.216', help='QuestDB host')
    args = parser.parse_args()

    import os
    os.makedirs('/home/coder/project/Real_Time_SKA_trading/bot_results_v2', exist_ok=True)

    bot = SKATradingBot(
        db_host=args.host,
        symbol=args.symbol,
        poll_interval=args.poll,
        dry_run=not args.live
    )
    asyncio.run(bot.run())
