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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE        = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR    = os.path.join(_HERE, 'XRPUSDT')
OUTPUT_DIR   = os.path.join(_HERE, 'results')
SUMMARY_FILE = os.path.join(_HERE, 'summary.csv')

WAIT_PAIR  = 'WAIT_PAIR'
IN_NEUTRAL = 'IN_NEUTRAL'
READY      = 'READY'
EXIT_WAIT  = 'EXIT_WAIT'

# ─── Filter ───────────────────────────────────────────────────────────────────
# Minimum neutral→neutral count before READY state is allowed.
# If the first non-neutral arrives before this threshold, the gap is too short:
# stay IN_NEUTRAL and reset the count (wait for the next proper gap).
MIN_NN_COUNT = 3


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
    """Consecutive same-direction paired cycles. Symmetric exit.

    LONG:
      neutral→bull              (OPEN — WAIT_PAIR)
      bull→neutral              (pair confirmed — IN_NEUTRAL)
      neutral→neutral × N       (count all — stay IN_NEUTRAL)
      <first non-neutral>       (gap closes — READY)
      neutral→bull              (cycle repeats — WAIT_PAIR)
      neutral→bear              (opposite cycle opens — EXIT_WAIT)
      bear→neutral              (opposite pair confirmed — CLOSE LONG)

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
                    if self.position.neutral_neutral_count >= MIN_NN_COUNT:
                        self.position.exit_state = READY
                    else:
                        self.position.neutral_neutral_count = 0
            elif self.position.exit_state == READY:
                if name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                elif name == 'neutral→bear':
                    self.position.exit_state = EXIT_WAIT
            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bear→neutral':
                    self._close(price, tid, name)
                    self._open('SHORT', price, tid, ts, 'neutral→bear')
                elif name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0

        elif self.position.side == 'SHORT':
            if self.position.exit_state == WAIT_PAIR:
                if name == 'bear→neutral':
                    self.position.exit_state = IN_NEUTRAL
            elif self.position.exit_state == IN_NEUTRAL:
                if name == 'neutral→neutral':
                    self.position.neutral_neutral_count += 1
                else:
                    if self.position.neutral_neutral_count >= MIN_NN_COUNT:
                        self.position.exit_state = READY
                    else:
                        self.position.neutral_neutral_count = 0
            elif self.position.exit_state == READY:
                if name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0
                elif name == 'neutral→bull':
                    self.position.exit_state = EXIT_WAIT
            elif self.position.exit_state == EXIT_WAIT:
                if name == 'bull→neutral':
                    self._close(price, tid, name)
                    self._open('LONG', price, tid, ts, 'neutral→bull')
                elif name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0


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

    plot_nn_distribution()


def plot_nn_distribution():
    """Plot nn_count distribution (winners vs losers) across all result files."""
    results_dir = os.path.join(OUTPUT_DIR, 'v1')
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]

    win_counts  = defaultdict(int)
    lose_counts = defaultdict(int)

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            for row in csv.DictReader(f):
                nn = int(row['nn_count'])
                if float(row['pnl']) > 0:
                    win_counts[nn]  += 1
                else:
                    lose_counts[nn] += 1

    max_nn = max(list(win_counts.keys()) + list(lose_counts.keys()))
    xs = list(range(max_nn + 1))
    wins  = [win_counts.get(x, 0)  for x in xs]
    loses = [lose_counts.get(x, 0) for x in xs]
    totals = [w + l for w, l in zip(wins, loses)]
    win_rates = [w / t * 100 if t > 0 else 0 for w, t in zip(wins, totals)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#FFFFFF')

    # Top: stacked bar — winners and losers
    ax1.bar(xs, wins,  color='#A8DFBC', label='Winners', alpha=0.9)
    ax1.bar(xs, loses, bottom=wins, color='#FFAAAA', label='Losers', alpha=0.9)
    ax1.axvline(MIN_NN_COUNT - 0.5, color='#CC2222', linewidth=1.5, linestyle='--',
                label=f'MIN_NN_COUNT={MIN_NN_COUNT}')
    ax1.set_ylabel('Trade count')
    ax1.set_title('Neutral Gap Distribution — Winners vs Losers', fontweight='bold')
    ax1.legend()
    ax1.set_facecolor('#F8F8F8')

    # Bottom: win rate per nn_count
    ax2.bar(xs, win_rates, color='#5599CC', alpha=0.85)
    ax2.axhline(50, color='#888888', linewidth=1, linestyle=':')
    ax2.axvline(MIN_NN_COUNT - 0.5, color='#CC2222', linewidth=1.5, linestyle='--',
                label=f'MIN_NN_COUNT={MIN_NN_COUNT}')
    ax2.set_ylabel('Win rate (%)')
    ax2.set_xlabel('nn_count (neutral→neutral transitions in neutral gap)')
    ax2.set_title('Win Rate by Neutral Gap Length', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.set_facecolor('#F8F8F8')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'nn_distribution.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")

    plot_nn_scatter()


def plot_nn_scatter():
    """Scatter plot: trade index (x) vs nn_count (y), colored by winner/loser."""
    results_dir = os.path.join(OUTPUT_DIR, 'v1')
    files = sorted(f for f in os.listdir(results_dir) if f.endswith('.csv'))

    nn_win, nn_lose, nn_flat = [], [], []
    idx = 0

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            for row in csv.DictReader(f):
                nn  = int(row['nn_count'])
                pnl = float(row['pnl'])
                if pnl > 0:
                    nn_win.append((idx, nn))
                elif pnl < 0:
                    nn_lose.append((idx, nn))
                else:
                    nn_flat.append((idx, nn))
                idx += 1

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8F8F8')

    if nn_lose:  ax.scatter(*zip(*nn_lose), color='#FF6666', s=12, alpha=0.5, label='Loser',  zorder=2)
    if nn_flat:  ax.scatter(*zip(*nn_flat), color='#AAAAAA', s=8,  alpha=0.4, label='Flat',   zorder=2)
    if nn_win:   ax.scatter(*zip(*nn_win),  color='#44AA66', s=12, alpha=0.6, label='Winner', zorder=3)

    ax.axhline(MIN_NN_COUNT, color='#CC2222', linewidth=1.5, linestyle='--',
               label=f'MIN_NN_COUNT={MIN_NN_COUNT}')

    ax.set_xlabel('Trade index')
    ax.set_ylabel('nn_count (neutral gap length)')
    ax.set_title('Neutral Gap per Trade — Winners vs Losers', fontweight='bold')
    ax.legend(markerscale=2)

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'nn_scatter.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")


if __name__ == '__main__':
    main()
