"""
Paired cycle trading bot v3 — entropy-derived probability regime transitions (ΔP tolerance bands).

Regime definition:
  P(n)   = exp(-|ΔH/H|)   where  ΔH/H = (H(n) - H(n-1)) / H(n)
  ΔP(n)  = P(n) - P(n-1)  consecutive change in probability

  |ΔP(n) − (−0.86)| ≤ K × 0.14  →  regime = 2  ("bear"    — neutral→bear, tol=0.004)
  |ΔP(n) − (−0.34)| ≤ K × 0.66  →  regime = 1  ("bull"    — neutral→bull, tol=0.020)
  else                             →  regime = 0  (neutral)

  P band positions — universal constants at convergence scale (asset-independent):
    P_NEUTRAL_NEUTRAL = 1.00
    P_NEUTRAL_BULL    = 0.66
    P_X_NEUTRAL       = 0.51   (bull→neutral = bear→neutral)
    P_NEUTRAL_BEAR    = 0.14

  BULL_THRESHOLD = P_NEUTRAL_NEUTRAL − P_NEUTRAL_BULL = 0.34
  BEAR_THRESHOLD = P_NEUTRAL_NEUTRAL − P_NEUTRAL_BEAR = 0.86

  DP_PAIR_BULL = 0.15  # P_NEUTRAL_BULL − P_X_NEUTRAL  (market constant at convergence)
  DP_PAIR_BEAR = 0.37  # P_X_NEUTRAL   − P_NEUTRAL_BEAR (market constant at convergence)

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
import base64
import math
import os
import urllib.parse
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import logging
import time
import asyncpg
import aiohttp
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

MIN_NN_COUNT    = 10
MIN_TRADES      = 50    # wait for SKA convergence before trading
ENGINE_RESET_AT = 3500  # engine resets at this entropy count
DP_PAIR_CUTOFF  = 3200  # stop recording ΔP_pair before engine reset

DB_HOST         = os.environ.get('QDB_HOST',     '192.168.1.216')
DB_PORT         = int(os.environ.get('QDB_PORT', '8812'))
DB_USER         = os.environ.get('QDB_USER',     'admin')
DB_PASSWORD     = os.environ.get('QDB_PASSWORD', 'quest')
POLL_INTERVAL   = 1.0   # seconds between DB polls
RESULTS_DIR     = '/home/coder/project/Real_Time_SKA_trading/bot_results_v3'

# Binance API — loaded lazily; validated in connect() only when dry_run=False
BINANCE_API_KEY          = os.environ.get('BINANCE_API_KEY')
BINANCE_PRIVATE_KEY_PATH = os.environ.get('BINANCE_PRIVATE_KEY_PATH')
BINANCE_BASE_URL         = 'https://api.binance.com'
ORDER_QUANTITY = {
    'XRPUSDT':  7.0,      # ~$10 at $1.43
    'XRPUSDC':  7.0,
    'BTCUSDT':  0.0001,   # ~$8.5 at $85,000
    'BTCUSDC':  0.0001,
    'ETHUSDT':  0.005,    # ~$10 at $2,000
    'ETHUSDC':  0.005,
    'SOLUSDC':  0.1,      # ~$15 at $150
}

# P band positions — universal constants at convergence scale, confirmed XRPUSDT+BTCUSDT
P_NEUTRAL_NEUTRAL = 1.00
P_NEUTRAL_BULL    = 0.66
P_X_NEUTRAL       = 0.51   # bull→neutral = bear→neutral
P_NEUTRAL_BEAR    = 0.14

# Proportional tolerance: tol per transition = K × P_curr_structural
K         = 0.03
TOL_BEAR  = K * P_NEUTRAL_BEAR   # = 0.0042  neutral→bear band
TOL_BULL  = K * P_NEUTRAL_BULL   # = 0.0198  neutral→bull band
TOL_CLOSE = K * P_X_NEUTRAL      # = 0.0153  bull→neutral = bear→neutral band

# ΔP centers for regime classification (negative — P drops from neutral)
DP_NEUTRAL_BEAR = -(P_NEUTRAL_NEUTRAL - P_NEUTRAL_BEAR)  # = -0.86
DP_NEUTRAL_BULL = -(P_NEUTRAL_NEUTRAL - P_NEUTRAL_BULL)  # = -0.34

# ΔP_pair — paired transition gap at convergence scale
DP_PAIR_BULL = P_NEUTRAL_BULL - P_X_NEUTRAL    # = 0.15
DP_PAIR_BEAR = P_X_NEUTRAL   - P_NEUTRAL_BEAR  # = 0.37

EVENT = {
    'OPEN_LONG':      1,
    'OPEN_SHORT':     2,
    'CLOSE_LONG':     3,
    'CLOSE_SHORT':    4,
    'PAIR_CONFIRMED': 5,
    'NEUTRAL_GAP':    6,
    'CYCLE_REPEAT':   7,
    'OPPOSITE_OPEN':  8,
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
    bull_pair_count: int = field(default=0)
    bear_pair_count: int = field(default=0)


class SKATradingBot:
    """Paired cycle trading bot v3 — regime classified from ΔP tolerance bands where P = exp(-|ΔH/H|).

    Execution model (spot only — no margin/futures):
      LONG open  (neutral→bull, or SHORT close → re-enter) : BUY on exchange
      LONG close (bear→neutral)                            : SELL on exchange
      SHORT                                                : synthetic only, no exchange orders
    Exchange state sequence: BUY → SELL → [flat, tracking SHORT] → BUY → SELL → ...
    """

    def __init__(self, db_host=DB_HOST, db_port=DB_PORT, symbol='XRPUSDT',
                 poll_interval=POLL_INTERVAL, dry_run=True):
        self.db_host = db_host
        self.db_port = db_port
        self.symbol = symbol
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self.conn = None
        self.position: Optional[Position] = None
        self.last_trade_id = None
        self._private_key = None

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file  = f'{RESULTS_DIR}/bot_results_v3_{ts}.csv'
        self.dp_pair_file  = f'{RESULTS_DIR}/dp_pair_v3_{ts}.csv'

        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # PnL split: LONG = real spot execution, SHORT = synthetic signal tracking
        self.spot_pnl      = 0.0   # realized from real exchange LONGs only
        self.synthetic_pnl = 0.0   # synthetic SHORTs — no exchange orders
        self.trade_log = []

        # ΔP_pair tracking
        self._last_open_name = None   # neutral→bull or neutral→bear
        self._last_open_P    = None
        self._already_long   = False  # BUY already on exchange after CLOSE_SHORT
        self._dp_pair_written = False
        self._csv_written     = False
        self._entropy_count   = 0
        self._lot_filter      = None   # populated by _fetch_lot_filter() at connect

    async def connect(self):
        self.conn = await asyncpg.connect(
            host=self.db_host, port=self.db_port,
            database='qdb', user=DB_USER, password=DB_PASSWORD
        )
        logging.info(f"Connected to QuestDB at {self.db_host}:{self.db_port}")
        if not self.dry_run:
            if not BINANCE_API_KEY:
                raise RuntimeError("BINANCE_API_KEY env var is not set")
            if not BINANCE_PRIVATE_KEY_PATH:
                raise RuntimeError("BINANCE_PRIVATE_KEY_PATH env var is not set")
            self._lot_filter = await self._fetch_lot_filter()
            self._private_key = self._load_private_key()
            logging.info("Ed25519 private key loaded")
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ska_bot_v3 (
                timestamp            TIMESTAMP,
                trade_id             DOUBLE,
                price                DOUBLE,
                P                    DOUBLE,
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
        logging.info("ska_bot_v3 table ready")

    async def _log_event(self, trade_id, price, event, state, side='', pnl=None, neutral_neutral_count=None, P=None):
        try:
            await self.conn.execute(
                """INSERT INTO ska_bot_v3
                   (timestamp, trade_id, price, P, event, event_name, state, state_name, side, side_name, pnl, neutral_neutral_count)
                   VALUES (now(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                trade_id, price, P,
                EVENT[event], event,
                STATE[state], state,
                SIDE[side], side if side else None,
                pnl, neutral_neutral_count
            )
        except Exception as e:
            logging.warning(f"State log failed: {e}")

    async def get_new_transitions(self):
        last_id = max(0, self.last_trade_id - 2) if self.last_trade_id is not None else 0
        query = """
        WITH base AS (
          SELECT
            timestamp, symbol, trade_id, price, entropy,
            LAG(entropy, 1) OVER (ORDER BY timestamp, trade_id) AS e1,
            LAG(entropy, 2) OVER (ORDER BY timestamp, trade_id) AS e2
          FROM binance_trades
          WHERE symbol = $1 AND entropy IS NOT NULL AND trade_id >= $2
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
              WHEN prev_P IS NOT NULL AND ABS((P - prev_P) - $3) <= $4 THEN 2
              WHEN prev_P IS NOT NULL AND ABS((P - prev_P) - $5) <= $6 THEN 1
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
            rows = await self.conn.fetch(query, self.symbol, last_id,
                                         DP_NEUTRAL_BEAR, TOL_BEAR,
                                         DP_NEUTRAL_BULL, TOL_BULL)
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

    def _record_dp_pair(self, pair_type, p1, p2):
        # p1 = P at opening transition (neutral→bull or neutral→bear)
        # p2 = P at closing transition (bull→neutral or bear→neutral)
        # ΔP_pair = p2 - p1 → negative for bull, positive for bear
        if p1 is None or p2 is None:
            return
        dp = p2 - p1
        row = {
            'pair_type': pair_type,
            'p1':        round(p1, 4),
            'p2':        round(p2, 4),
            'dp_pair':   round(dp, 4),
        }
        with open(self.dp_pair_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._dp_pair_written:
                writer.writeheader()
                self._dp_pair_written = True
            writer.writerow(row)
        logging.info(
            f"ΔP_pair [{pair_type}] P={p1:.4f}→{p2:.4f} ΔP={dp:+.4f}"
        )

    async def process_signal(self, transition):
        trade_id = transition['trade_id']
        price    = transition['price']
        name     = transition['transition_name']
        P        = transition['P']
        ts       = transition['timestamp']

        if self.last_trade_id is not None and trade_id <= self.last_trade_id:
            return
        self.last_trade_id = trade_id


        # ΔP_pair: gap within paired transition neutral→bull→neutral or neutral→bear→neutral
        # ΔP = P(closing) − P(opening) → negative for bull, positive for bear
        PAIR_CLOSE = {'neutral→bull': 'bull→neutral', 'neutral→bear': 'bear→neutral'}
        if name in ('neutral→bull', 'neutral→bear'):
            self._last_open_name = name
            self._last_open_P    = P
        elif (name in ('bull→neutral', 'bear→neutral') and
              self._last_open_name is not None and
              PAIR_CLOSE.get(self._last_open_name) == name and
              self._entropy_count < DP_PAIR_CUTOFF and
              P is not None and abs(P - P_X_NEUTRAL) <= TOL_CLOSE):
            pair_type = 'bull' if name == 'bull→neutral' else 'bear'
            self._record_dp_pair(pair_type, self._last_open_P, P)
            if self.position is not None:
                if pair_type == 'bull':
                    self.position.bull_pair_count += 1
                else:
                    self.position.bear_pair_count += 1
            self._last_open_name = None
            self._last_open_P    = None

        p_str = f"{P:.4f}" if P is not None else "n/a"

        # === NO POSITION: look for entry ===
        if self.position is None:
            if name == 'neutral→bull':
                if not self.dry_run and not self._already_long:
                    if not await self._execute_buy(price):
                        logging.error("[ORDER] BUY failed — LONG not opened")
                        return
                self._already_long = False
                self.position = Position(
                    side='LONG', entry_price=price,
                    entry_trade_id=trade_id, entry_time=str(ts),
                    entry_transition=name, exit_state=WAIT_PAIR
                )
                logging.info(
                    f">>> OPEN LONG @ {price:.6f} | P={p_str} | trade_id={trade_id} "
                    f"| waiting: bull→neutral"
                )
                await self._log_event(trade_id, price, 'OPEN_LONG', WAIT_PAIR, 'LONG', P=P)

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
                await self._log_event(trade_id, price, 'OPEN_SHORT', WAIT_PAIR, 'SHORT', P=P)
                # No exchange order: spot cannot short-sell — SHORT tracked synthetically only
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
                    await self._log_event(trade_id, price, 'PAIR_CONFIRMED', IN_NEUTRAL, 'LONG', P=P)

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
                                              neutral_neutral_count=self.position.neutral_neutral_count, P=P)
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
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'LONG', P=P)
                elif name == 'neutral→bear':
                    self.position.exit_state = EXIT_WAIT
                    logging.info(
                        f"--- Opposite cycle opening (neutral→bear) @ {price:.6f} "
                        f"| EXIT_WAIT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'OPPOSITE_OPEN', EXIT_WAIT, 'LONG', P=P)

            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bear→neutral' and P is not None and abs(P - P_X_NEUTRAL) <= TOL_CLOSE:
                    if not self.dry_run:
                        if not await self._execute_sell(price):
                            logging.error("[ORDER] SELL failed — LONG not closed")
                            return
                    pnl = price - self.position.entry_price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE LONG (bear→neutral) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    await self._log_event(trade_id, price, 'CLOSE_LONG', EXIT_WAIT, 'LONG', pnl, P=P)
                    self.position = None
                elif name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- Bear cycle aborted (neutral→bull) @ {price:.6f} "
                        f"| WAIT_PAIR | still LONG | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'LONG', P=P)

        # === SHORT POSITION ===
        elif self.position.side == 'SHORT':

            if self.position.exit_state == WAIT_PAIR:
                if name == 'bear→neutral':
                    self.position.exit_state = IN_NEUTRAL
                    logging.info(
                        f"--- DOWN pair confirmed (bear→neutral) @ {price:.6f} "
                        f"| IN_NEUTRAL | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'PAIR_CONFIRMED', IN_NEUTRAL, 'SHORT', P=P)

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
                                              neutral_neutral_count=self.position.neutral_neutral_count, P=P)
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
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'SHORT', P=P)
                elif name == 'neutral→bull':
                    self.position.exit_state = EXIT_WAIT
                    logging.info(
                        f"--- Opposite cycle opening (neutral→bull) @ {price:.6f} "
                        f"| EXIT_WAIT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'OPPOSITE_OPEN', EXIT_WAIT, 'SHORT', P=P)

            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bull→neutral' and P is not None and abs(P - P_X_NEUTRAL) <= TOL_CLOSE:
                    if not self.dry_run:
                        if not await self._execute_buy(price):
                            logging.error("[ORDER] BUY failed — SHORT not closed / LONG not re-entered")
                            return
                    pnl = self.position.entry_price - price
                    pnl_pct = (pnl / self.position.entry_price) * 100
                    self._record_trade(pnl, pnl_pct, price)
                    logging.info(
                        f"<<< CLOSE SHORT (bull→neutral) @ {price:.6f} | "
                        f"PnL={pnl:+.6f} ({pnl_pct:+.4f}%) | entry={self.position.entry_price:.6f}"
                    )
                    await self._log_event(trade_id, price, 'CLOSE_SHORT', EXIT_WAIT, 'SHORT', pnl, P=P)
                    self._already_long = True
                    self.position = None
                elif name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                    logging.info(
                        f"--- Bull cycle aborted (neutral→bear) @ {price:.6f} "
                        f"| WAIT_PAIR | still SHORT | trade_id={trade_id}"
                    )
                    await self._log_event(trade_id, price, 'CYCLE_REPEAT', WAIT_PAIR, 'SHORT', P=P)

    def _record_trade(self, pnl, pnl_pct, exit_price):
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Route PnL to the correct bucket
        is_real = (self.position.side == 'LONG')   # LONGs are backed by real exchange orders
        if is_real:
            self.spot_pnl += pnl
        else:
            self.synthetic_pnl += pnl

        trade = {
            'side': self.position.side,
            'real': is_real,
            'entry': self.position.entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'entry_transition': self.position.entry_transition,
            'bull_pairs': self.position.bull_pair_count,
            'bear_pairs': self.position.bear_pair_count,
        }
        self.trade_log.append(trade)

        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'side', 'real', 'entry', 'exit', 'pnl', 'entry_transition', 'bull_pairs', 'bear_pairs'
            ])
            if not self._csv_written:
                writer.writeheader()
                self._csv_written = True
            writer.writerow(trade)

    async def _fetch_lot_filter(self) -> dict:
        """Fetch LOT_SIZE and NOTIONAL filters from Binance exchangeInfo (public, no auth)."""
        url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo?symbol={self.symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"exchangeInfo returned HTTP {resp.status}")
                    data = await resp.json()
        except Exception as e:
            raise RuntimeError(f"[FILTERS] Failed to fetch exchangeInfo for {self.symbol}: {e}") from e

        symbols = data.get('symbols', [])
        if not symbols:
            raise RuntimeError(f"[FILTERS] No symbol data returned for {self.symbol}")

        filter_map = {f['filterType']: f for f in symbols[0].get('filters', [])}

        if 'LOT_SIZE' not in filter_map:
            raise RuntimeError(f"[FILTERS] LOT_SIZE filter missing for {self.symbol}")
        lot = filter_map['LOT_SIZE']

        notional_filter = filter_map.get('NOTIONAL') or filter_map.get('MIN_NOTIONAL') or {}
        result = {
            'step_size':    float(lot['stepSize']),
            'min_qty':      float(lot['minQty']),
            'max_qty':      float(lot['maxQty']),
            'min_notional': float(notional_filter.get('minNotional', 0)),
            'step_str':     lot['stepSize'],   # kept as string for precision calculation
        }
        logging.info(
            f"[FILTERS] {self.symbol} stepSize={result['step_size']} "
            f"minQty={result['min_qty']} maxQty={result['max_qty']} "
            f"minNotional={result['min_notional']}"
        )
        return result

    def _quantize_qty(self, qty: float, price: float) -> float:
        """Floor qty to stepSize, enforce minQty and minNotional. No-op in dry run."""
        if self._lot_filter is None:
            return qty
        f = self._lot_filter
        step = f['step_size']
        # floor to step size using integer arithmetic to avoid float drift
        qty = math.floor(qty / step) * step
        # round to the number of decimals in stepSize string
        step_str = f['step_str'].rstrip('0')
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        qty = round(qty, decimals)
        if qty < f['min_qty']:
            raise ValueError(
                f"{self.symbol}: qty {qty} < minQty {f['min_qty']} — increase ORDER_QUANTITY"
            )
        if qty > f['max_qty']:
            raise ValueError(
                f"{self.symbol}: qty {qty} > maxQty {f['max_qty']} — decrease ORDER_QUANTITY"
            )
        notional = qty * price
        if notional < f['min_notional']:
            raise ValueError(
                f"{self.symbol}: notional {notional:.4f} < minNotional {f['min_notional']} "
                f"— increase ORDER_QUANTITY"
            )
        return qty

    def _load_private_key(self) -> Ed25519PrivateKey:
        """Load Ed25519 private key from PEM file."""
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        with open(BINANCE_PRIVATE_KEY_PATH, 'rb') as f:
            return load_pem_private_key(f.read(), password=None)

    async def _binance_order(self, side, quantity) -> bool:
        """Place a market order on Binance using Ed25519 signature. Returns True only if status=FILLED."""
        params = {
            'symbol':    self.symbol,
            'side':      side,
            'type':      'MARKET',
            'quantity':  quantity,
            'timestamp': int(time.time() * 1000),
        }
        query = '&'.join(f"{k}={v}" for k, v in params.items())
        signature = urllib.parse.quote(
            base64.b64encode(self._private_key.sign(query.encode('ASCII'))).decode()
        )
        url = f"{BINANCE_BASE_URL}/api/v3/order?{query}&signature={signature}"
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers) as resp:
                    data = await resp.json()
                    order_status = data.get('status')
                    if resp.status == 200 and order_status == 'FILLED':
                        logging.info(
                            f"[ORDER] {side} {quantity} {self.symbol} FILLED "
                            f"orderId={data.get('orderId')} "
                            f"fills={data.get('fills')}"
                        )
                        return True
                    logging.error(
                        f"[ORDER] {side} not filled: HTTP {resp.status} status={order_status} {data}"
                    )
                    return False
        except Exception as e:
            logging.error(f"[ORDER] {side} exception: {e}")
            return False

    async def _execute_buy(self, price) -> bool:
        qty = self._quantize_qty(ORDER_QUANTITY.get(self.symbol, 1.0), price)
        logging.info(f"[EXECUTE] BUY {self.symbol} @ {price:.6f} qty={qty}")
        return await self._binance_order('BUY', qty)

    async def _execute_sell(self, price) -> bool:
        qty = self._quantize_qty(ORDER_QUANTITY.get(self.symbol, 1.0), price)
        logging.info(f"[EXECUTE] SELL {self.symbol} @ {price:.6f} qty={qty}")
        return await self._binance_order('SELL', qty)

    def print_stats(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        combined = self.spot_pnl + self.synthetic_pnl
        logging.info(
            f"=== STATS === Trades: {self.total_trades} | "
            f"Win: {self.winning_trades} | Lose: {self.losing_trades} | "
            f"Win rate: {win_rate:.1f}% | "
            f"Spot PnL (real): {self.spot_pnl:+.6f} | "
            f"Synthetic PnL: {self.synthetic_pnl:+.6f} | "
            f"Combined signal PnL: {combined:+.6f}"
        )

    def save_results(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        combined = self.spot_pnl + self.synthetic_pnl
        logging.info(f"Results file: {self.results_file}")
        logging.info(
            f"Summary: {self.total_trades} trades | win_rate={win_rate:.1f}% | "
            f"spot_pnl={self.spot_pnl:+.6f} | synthetic_pnl={self.synthetic_pnl:+.6f} | "
            f"combined={combined:+.6f}"
        )

    async def get_entropy_count(self):
        try:
            row = await self.conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM binance_trades WHERE symbol = $1 AND entropy IS NOT NULL",
                self.symbol
            )
            return int(row['cnt']) if row else 0
        except Exception:
            return 0

    async def run(self, max_trades=ENGINE_RESET_AT):
        await self.connect()
        logging.info(f"SKA Trading Bot v3 | symbol={self.symbol} | dry_run={self.dry_run} | auto_stop={max_trades} | K={K}")
        logging.info(f"Regime: |ΔP−(−0.86)|≤{TOL_BEAR:.4f} → bear | |ΔP−(−0.34)|≤{TOL_BULL:.4f} → bull | else → neutral")
        logging.info("LONG:  neutral→bull → bull→neutral → neutral→neutral × N → neutral→bear → bear→neutral (CLOSE)")
        logging.info("SHORT: neutral→bear → bear→neutral → neutral→neutral × N → neutral→bull → bull→neutral (CLOSE)")

        logging.info(f"Waiting for {MIN_TRADES} trades with entropy before trading...")
        while True:
            count = await self.get_entropy_count()
            if count >= MIN_TRADES:
                logging.info(f"SKA ready: {count} trades with entropy. Starting signals.")
                break
            await asyncio.sleep(self.poll_interval)

        try:
            while True:
                transitions = await self.get_new_transitions()
                for transition in transitions:
                    await self.process_signal(transition)

                count = await self.get_entropy_count()
                self._entropy_count = count
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

                # In live mode, only LONG holds real inventory — send a market SELL to liquidate.
                # SHORT is synthetic (no exchange position), so no order is needed.
                if not self.dry_run and self.position.side == 'LONG':
                    logging.warning("[END] Live LONG open at shutdown — sending emergency SELL")
                    sold = await self._execute_sell(close_price)
                    if not sold:
                        logging.error(
                            "[END] Emergency SELL failed — manual intervention required: "
                            f"sell {ORDER_QUANTITY.get(self.symbol, '?')} {self.symbol} on Binance"
                        )

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

    os.makedirs(RESULTS_DIR, exist_ok=True)

    bot = SKATradingBot(
        db_host=args.host,
        symbol=args.symbol,
        poll_interval=args.poll,
        dry_run=not args.live
    )
    asyncio.run(bot.run())