"""
SKA Paired Cycle Trading — Batch Backtest

Runs all 4 bot versions (v1/v2/v3/v4) on each CSV file in XRPUSDT/.
Saves per-file trade results to results/vN/.
Prints summary comparison table at the end.

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

INPUT_DIR  = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/XRPUSDT'
OUTPUT_DIR = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/results'
SUMMARY_FILE = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/summary.csv'

# Bot state constants
WAIT_PAIR_CONFIRM = 'WAIT_PAIR_CONFIRM'
READY_TO_EXIT     = 'READY_TO_EXIT'
WAIT_OPP_OPEN     = 'WAIT_OPP_OPEN'
WAIT_OPP_CONFIRM  = 'WAIT_OPP_CONFIRM'


# ─── Transition computation ───────────────────────────────────────────────────

def compute_transitions(rows):
    """
    From raw CSV rows compute regime, transition_code, transition_name, P per row.
    Returns list of transition dicts ready to feed into process_signal().
    """
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

        # Regime
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

            # P = exp(-|ΔH/H|)
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
    exit_state: str = field(default=WAIT_PAIR_CONFIRM)


# ─── Bot state machines ───────────────────────────────────────────────────────

class BotBase:
    def __init__(self, version):
        self.version        = version
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
            exit_state=WAIT_PAIR_CONFIRM
        )

    def _close(self, price, trade_id, reason):
        pos = self.position
        if pos.side == 'LONG':
            pnl = price - pos.entry_price
        else:
            pnl = pos.entry_price - price
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
            'close_reason':     reason,
        })
        self.position = None

    def force_close(self, price):
        if self.position:
            self._close(price, -1, 'force')

    def process_signal(self, t):
        raise NotImplementedError


class BotV1(BotBase):
    """Single opposite signal closes."""
    def __init__(self):
        super().__init__('v1')

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
            if name in ('neutral→bear', 'bear→neutral'):
                self._close(price, tid, name)
                if name == 'neutral→bear':
                    self._open('SHORT', price, tid, ts, name)

        elif self.position.side == 'SHORT':
            if name in ('neutral→bull', 'bull→neutral'):
                self._close(price, tid, name)
                if name == 'neutral→bull':
                    self._open('LONG', price, tid, ts, name)


class BotV2(BotBase):
    """2 consecutive opposite signals close (structurally wrong)."""
    def __init__(self):
        super().__init__('v2')
        self.exit_confirmations = 0

    def _open(self, side, price, trade_id, ts, transition):
        super()._open(side, price, trade_id, ts, transition)
        self.exit_confirmations = 0

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
            if name == 'bull→neutral':
                self.exit_confirmations = 0
            elif name in ('neutral→bear', 'bear→neutral'):
                self.exit_confirmations += 1
                if self.exit_confirmations >= 2:
                    self._close(price, tid, name)
                    self.exit_confirmations = 0
            elif name in ('neutral→bull', 'bull→bull'):
                self.exit_confirmations = 0

        elif self.position.side == 'SHORT':
            if name == 'bear→neutral':
                self.exit_confirmations = 0
            elif name in ('neutral→bull', 'bull→neutral'):
                self.exit_confirmations += 1
                if self.exit_confirmations >= 2:
                    self._close(price, tid, name)
                    self.exit_confirmations = 0
            elif name in ('neutral→bear', 'bear→bear'):
                self.exit_confirmations = 0


class BotV3(BotBase):
    """Full opposite paired cycle required to close."""
    def __init__(self):
        super().__init__('v3')

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
            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bull→neutral':
                    self.position.exit_state = WAIT_OPP_OPEN
            elif self.position.exit_state == WAIT_OPP_OPEN:
                if name == 'neutral→bear':
                    self.position.exit_state = WAIT_OPP_CONFIRM
            elif self.position.exit_state == WAIT_OPP_CONFIRM:
                if name == 'bear→neutral':
                    self._close(price, tid, name)
                    self._open('SHORT', price, tid, ts, 'bear→neutral')
                    self.position.exit_state = WAIT_OPP_OPEN

        elif self.position.side == 'SHORT':
            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bear→neutral':
                    self.position.exit_state = WAIT_OPP_OPEN
            elif self.position.exit_state == WAIT_OPP_OPEN:
                if name == 'neutral→bull':
                    self.position.exit_state = WAIT_OPP_CONFIRM
            elif self.position.exit_state == WAIT_OPP_CONFIRM:
                if name == 'bull→neutral':
                    self._close(price, tid, name)
                    self._open('LONG', price, tid, ts, 'bull→neutral')
                    self.position.exit_state = WAIT_OPP_OPEN


class BotV4(BotBase):
    """Own pair confirmed first, then first opposite signal closes."""
    def __init__(self):
        super().__init__('v4')

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
            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bull→neutral':
                    self.position.exit_state = READY_TO_EXIT
            elif self.position.exit_state == READY_TO_EXIT:
                if name in ('neutral→bear', 'bear→neutral'):
                    self._close(price, tid, name)
                    if name == 'neutral→bear':
                        self._open('SHORT', price, tid, ts, name)

        elif self.position.side == 'SHORT':
            if self.position.exit_state == WAIT_PAIR_CONFIRM:
                if name == 'bear→neutral':
                    self.position.exit_state = READY_TO_EXIT
            elif self.position.exit_state == READY_TO_EXIT:
                if name in ('neutral→bull', 'bull→neutral'):
                    self._close(price, tid, name)
                    if name == 'neutral→bull':
                        self._open('LONG', price, tid, ts, name)


# ─── Main backtest runner ─────────────────────────────────────────────────────

def run_file(filepath, bots):
    with open(filepath) as f:
        rows = list(csv.DictReader(f))

    transitions = compute_transitions(rows)

    for bot in bots:
        for t in transitions:
            bot.process_signal(t)
        # Force close any open position at last price
        if transitions:
            bot.force_close(transitions[-1]['price'])


def save_trades(bot, filename):
    if not bot.trade_log:
        return
    path = os.path.join(OUTPUT_DIR, bot.version, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'side', 'entry', 'exit', 'pnl', 'pnl_pct', 'entry_transition', 'close_reason'
        ])
        writer.writeheader()
        writer.writerows(bot.trade_log)


def main():
    files = sorted([
        f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')
    ])

    summary_rows = []

    for fname in files:
        filepath = os.path.join(INPUT_DIR, fname)
        bots = [BotV1(), BotV2(), BotV3(), BotV4()]
        run_file(filepath, bots)

        for bot in bots:
            save_trades(bot, fname)
            n  = bot.total_trades
            wr = (bot.winning_trades / n * 100) if n > 0 else 0
            summary_rows.append({
                'file':         fname,
                'version':      bot.version,
                'trades':       n,
                'winners':      bot.winning_trades,
                'losers':       bot.losing_trades,
                'force_closes': bot.force_closes,
                'win_rate':     round(wr, 1),
                'total_pnl':    round(bot.total_pnl, 6),
                'avg_pnl':      round(bot.total_pnl / n, 6) if n > 0 else 0,
            })

        # Print per-file summary
        print(f"\n{fname}")
        print(f"  {'ver':<4} {'trades':>6} {'win%':>6} {'pnl':>10} {'force':>6}")
        for bot in bots:
            n  = bot.total_trades
            wr = (bot.winning_trades / n * 100) if n > 0 else 0
            print(f"  {bot.version:<4} {n:>6} {wr:>6.1f} {bot.total_pnl:>+10.6f} {bot.force_closes:>6}")

    # Save summary CSV
    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'version', 'trades', 'winners', 'losers',
            'force_closes', 'win_rate', 'total_pnl', 'avg_pnl'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print aggregated totals per version
    print(f"\n{'='*60}")
    print("AGGREGATE ACROSS ALL FILES")
    print(f"{'='*60}")
    print(f"  {'ver':<4} {'trades':>6} {'win%':>6} {'total_pnl':>12} {'avg_pnl':>10} {'force':>6}")
    for version in ['v1', 'v2', 'v3', 'v4']:
        rows = [r for r in summary_rows if r['version'] == version]
        n    = sum(r['trades'] for r in rows)
        w    = sum(r['winners'] for r in rows)
        pnl  = sum(r['total_pnl'] for r in rows)
        fc   = sum(r['force_closes'] for r in rows)
        wr   = (w / n * 100) if n > 0 else 0
        avg  = (pnl / n) if n > 0 else 0
        print(f"  {version:<4} {n:>6} {wr:>6.1f} {pnl:>+12.6f} {avg:>+10.6f} {fc:>6}")

    print(f"\nSummary saved: {SUMMARY_FILE}")


if __name__ == '__main__':
    main()
