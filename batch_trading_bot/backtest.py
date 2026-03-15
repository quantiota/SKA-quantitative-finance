"""
SKA Paired Cycle Trading — Batch Backtest

Runs bot v1 on each CSV file in XRPUSDT/.
Saves per-file trade results to results/v1/.
Prints summary at the end.

Input CSV columns (from QuestDB export):
  symbol, trade_id, price, quantity, ..., entropy, delta_t, ..., timestamp

Usage:
  /opt/venv/bin/python3 backtest.py
"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Optional

INPUT_DIR    = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/XRPUSDT'
OUTPUT_DIR   = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/results'
SUMMARY_FILE = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/summary.csv'

WAIT_PAIR  = 'WAIT_PAIR'
IN_NEUTRAL = 'IN_NEUTRAL'
READY      = 'READY'


# ─── Transition computation ───────────────────────────────────────────────────

def compute_transitions(rows):
    transitions = []
    prev_price   = None
    prev_entropy = None
    prev_regime  = None

    for row in rows:
        price   = float(row['price'])
        try:
            entropy = float(row['entropy']) if row['entropy'] not in ('', 'None', None) else None
        except ValueError:
            entropy = None

        if prev_price is None:
            prev_price   = price
            prev_entropy = entropy
            continue

        if price - prev_price > 0:
            regime = 1
        elif price - prev_price < 0:
            regime = 2
        else:
            regime = 0

        if prev_regime is not None:
            code = prev_regime * 3 + regime
            names = {
                0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
                3: 'bull→neutral',    4: 'bull→bull',    5: 'bull→bear',
                6: 'bear→neutral',    7: 'bear→bull',    8: 'bear→bear'
            }
            P = None
            if entropy is not None and prev_entropy is not None and entropy != 0:
                P = math.exp(-abs((entropy - prev_entropy) / entropy))

            transitions.append({
                'trade_id':        int(float(row['trade_id'])),
                'price':           price,
                'timestamp':       row['timestamp'],
                'transition_code': code,
                'transition_name': names.get(code, 'unknown'),
                'P':               P,
            })

        prev_regime  = regime
        prev_price   = price
        prev_entropy = entropy

    return transitions


# ─── Position dataclass ───────────────────────────────────────────────────────

@dataclass
class Position:
    side: str
    entry_price: float
    entry_trade_id: int
    entry_time: str
    entry_transition: str
    exit_state: str = field(default=WAIT_PAIR)
    neutral_neutral_count: int = field(default=0)


# ─── Bot v1 ───────────────────────────────────────────────────────────────────

class BotV1:
    """Consecutive same-direction paired cycles.

    LONG:
      neutral→bull              (OPEN — WAIT_PAIR)
      bull→neutral              (pair confirmed — IN_NEUTRAL)
      neutral→neutral × N       (count all — stay IN_NEUTRAL)
      <first non-neutral>       (gap closes — READY)
      neutral→bull              (cycle repeats — WAIT_PAIR)
      neutral→bear / bear→neutral  (CLOSE — only from READY)

    SHORT: mirror logic.
    """

    def __init__(self):
        self.version        = 'v1'
        self.position       = None
        self.last_trade_id  = None
        self.trade_log      = []
        self.total_trades   = 0
        self.winning_trades = 0
        self.losing_trades  = 0
        self.total_pnl      = 0.0
        self.force_closes   = 0

    def _open(self, side, price, trade_id, ts, transition):
        self.position = Position(
            side=side, entry_price=price,
            entry_trade_id=trade_id, entry_time=ts,
            entry_transition=transition,
            exit_state=WAIT_PAIR
        )

    def _close(self, price, trade_id, reason):
        pos = self.position
        pnl = price - pos.entry_price if pos.side == 'LONG' else pos.entry_price - price
        pnl_pct = (pnl / pos.entry_price) * 100
        self.total_trades += 1
        self.total_pnl    += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades  += 1
        if reason == 'force':
            self.force_closes += 1
        self.trade_log.append({
            'side':             pos.side,
            'entry':            pos.entry_price,
            'exit':             price,
            'pnl':              pnl,
            'pnl_pct':          pnl_pct,
            'entry_transition': pos.entry_transition,
            'nn_count':         pos.neutral_neutral_count,
            'close_reason':     reason,
        })
        self.position = None

    def force_close(self, price):
        if self.position:
            self._close(price, -1, 'force')

    def process_signal(self, t):
        tid   = t['trade_id']
        price = t['price']
        name  = t['transition_name']
        ts    = t['timestamp']

        if self.last_trade_id is not None and tid <= self.last_trade_id:
            return
        self.last_trade_id = tid

        if self.position is None:
            if name == 'neutral→bull':
                self._open('LONG',  price, tid, ts, name)
            elif name == 'neutral→bear':
                self._open('SHORT', price, tid, ts, name)
            return

        if self.position.side == 'LONG':
            if self.position.exit_state == WAIT_PAIR:
                if name == 'bull→neutral':
                    self.position.exit_state = IN_NEUTRAL
            elif self.position.exit_state == IN_NEUTRAL:
                if name == 'neutral→neutral':
                    self.position.neutral_neutral_count += 1
                else:
                    self.position.exit_state = READY
            elif self.position.exit_state == READY:
                if name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                elif name == 'neutral→bear':
                    self._close(price, tid, name)
                    self._open('SHORT', price, tid, ts, name)
                elif name == 'bear→neutral':
                    self._close(price, tid, name)

        elif self.position.side == 'SHORT':
            if self.position.exit_state == WAIT_PAIR:
                if name == 'bear→neutral':
                    self.position.exit_state = IN_NEUTRAL
            elif self.position.exit_state == IN_NEUTRAL:
                if name == 'neutral→neutral':
                    self.position.neutral_neutral_count += 1
                else:
                    self.position.exit_state = READY
            elif self.position.exit_state == READY:
                if name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                elif name == 'neutral→bull':
                    self._close(price, tid, name)
                    self._open('LONG', price, tid, ts, name)
                elif name == 'bull→neutral':
                    self._close(price, tid, name)


# ─── Main backtest runner ─────────────────────────────────────────────────────

def run_file(filepath, bot):
    with open(filepath) as f:
        rows = list(csv.DictReader(f))
    transitions = compute_transitions(rows)
    for t in transitions:
        bot.process_signal(t)
    if transitions:
        bot.force_close(transitions[-1]['price'])


def save_trades(bot, filename):
    if not bot.trade_log:
        return
    os.makedirs(os.path.join(OUTPUT_DIR, bot.version), exist_ok=True)
    path = os.path.join(OUTPUT_DIR, bot.version, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'side', 'entry', 'exit', 'pnl', 'pnl_pct',
            'entry_transition', 'nn_count', 'close_reason'
        ])
        writer.writeheader()
        writer.writerows(bot.trade_log)


def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')])

    summary_rows = []
    agg = {'trades': 0, 'winners': 0, 'pnl': 0.0, 'force': 0}

    print(f"  {'file':<40} {'trades':>6} {'win%':>6} {'pnl':>10} {'force':>6}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*10} {'-'*6}")

    for fname in files:
        bot = BotV1()
        run_file(os.path.join(INPUT_DIR, fname), bot)
        save_trades(bot, fname)

        n  = bot.total_trades
        wr = (bot.winning_trades / n * 100) if n > 0 else 0
        print(f"  {fname:<40} {n:>6} {wr:>6.1f} {bot.total_pnl:>+10.6f} {bot.force_closes:>6}")

        summary_rows.append({
            'file': fname, 'version': 'v1', 'trades': n,
            'winners': bot.winning_trades, 'losers': bot.losing_trades,
            'force_closes': bot.force_closes, 'win_rate': round(wr, 1),
            'total_pnl': round(bot.total_pnl, 6),
            'avg_pnl': round(bot.total_pnl / n, 6) if n > 0 else 0,
        })
        agg['trades']  += n
        agg['winners'] += bot.winning_trades
        agg['pnl']     += bot.total_pnl
        agg['force']   += bot.force_closes

    wr = (agg['winners'] / agg['trades'] * 100) if agg['trades'] > 0 else 0
    avg = agg['pnl'] / agg['trades'] if agg['trades'] > 0 else 0
    print(f"\n{'='*70}")
    print(f"  TOTAL  {agg['trades']:>6} {wr:>6.1f} {agg['pnl']:>+10.6f}  avg={avg:>+.6f}  force={agg['force']}")

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'version', 'trades', 'winners', 'losers',
            'force_closes', 'win_rate', 'total_pnl', 'avg_pnl'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary saved: {SUMMARY_FILE}")


if __name__ == '__main__':
    main()
