"""
SKA Paired Cycle Trading — Batch Backtest v3

Runs bot v3 logic on each CSV file in XRPUSDT/.
Saves per-file trade results to results/v3/.
Prints summary at the end.

Regime definition (matches trading_bot_v3.py exactly):
  P(n)      = exp(-|ΔH/H|)              where ΔH/H = (H(n) - H(n-1)) / H(n)
  prev_P(n) = exp(-|ΔH(n-1)/H(n-1)|)
  ΔP(n)     = P(n) - prev_P(n)

  |ΔP − (−0.86)| ≤ TOL_BEAR  →  regime = 2  (bear)
  |ΔP − (−0.34)| ≤ TOL_BULL  →  regime = 1  (bull)
  else                         →  regime = 0  (neutral)

  Transition code = prev_regime * 3 + regime  (same 9 transitions)

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

INPUT_DIR    = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/XRPUSDT'
OUTPUT_DIR   = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/results'
SUMMARY_FILE = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/summary.csv'

WAIT_PAIR  = 'WAIT_PAIR'
IN_NEUTRAL = 'IN_NEUTRAL'
READY      = 'READY'
EXIT_WAIT  = 'EXIT_WAIT'

# ─── v3 constants (match trading_bot_v3.py exactly) ──────────────────────────
MIN_NN_COUNT = 10
MIN_TRADES   = 50       # wait for SKA convergence before first trade

P_X_NEUTRAL  = 0.51    # bull→neutral = bear→neutral band
K            = 0.03
TOL_BEAR     = K * 0.14  # = 0.0042
TOL_BULL     = K * 0.66  # = 0.0198
TOL_CLOSE    = K * 0.51  # = 0.0153

BULL_THRESHOLD = -0.34  # ΔP centre for neutral→bull
BEAR_THRESHOLD = -0.86  # ΔP centre for neutral→bear

TRANSITION_NAMES = {
    0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
    3: 'bull→neutral',    4: 'bull→bull',    5: 'bull→bear',
    6: 'bear→neutral',    7: 'bear→bull',    8: 'bear→bear'
}


# ─── Transition computation (v3 regime: ΔP bands, not price direction) ────────

def compute_transitions(rows):
    """
    Requires 3 consecutive entropy values to compute P and prev_P:
      e2 (oldest) → e1 → e (current)
      P      = exp(-|e  - e1| / |e|)
      prev_P = exp(-|e1 - e2| / |e1|)
      ΔP     = P - prev_P
      regime = band(ΔP)
    Entropy-valid tick counter drives MIN_TRADES guard.
    """
    transitions  = []
    prev_regime  = None
    entropy_buf  = []   # rolling window of last 3 valid entropy values [e2, e1, e]
    entropy_count = 0   # number of ticks with valid entropy seen so far

    for row in rows:
        price = float(row['price'])
        try:
            entropy = float(row['entropy']) if row['entropy'] not in ('', 'None', None) else None
        except ValueError:
            entropy = None

        if entropy is None:
            prev_regime = None  # break in entropy stream resets regime chain
            continue

        entropy_count += 1
        entropy_buf.append(entropy)
        if len(entropy_buf) > 3:
            entropy_buf.pop(0)

        if len(entropy_buf) < 3:
            prev_regime = None
            continue

        e2, e1, e = entropy_buf

        # P and prev_P
        P      = math.exp(-abs((e  - e1) / e )) if e  != 0 else None
        prev_P = math.exp(-abs((e1 - e2) / e1)) if e1 != 0 else None

        # Regime from ΔP band
        if P is not None and prev_P is not None:
            dp = P - prev_P
            if abs(dp - BEAR_THRESHOLD) <= TOL_BEAR:
                regime = 2
            elif abs(dp - BULL_THRESHOLD) <= TOL_BULL:
                regime = 1
            else:
                regime = 0
        else:
            regime = 0

        if prev_regime is not None:
            code = prev_regime * 3 + regime
            transitions.append({
                'trade_id':        int(float(row['trade_id'])),
                'price':           price,
                'timestamp':       row['timestamp'],
                'transition_code': code,
                'transition_name': TRANSITION_NAMES.get(code, 'unknown'),
                'P':               P,
                'entropy_count':   entropy_count,
            })

        prev_regime = regime

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
    bull_pair_count: int = field(default=0)
    bear_pair_count: int = field(default=0)


# ─── Bot v3 ───────────────────────────────────────────────────────────────────

class BotV3:
    """
    Backtest replica of trading_bot_v3.py signal logic.

    Key v3 behaviours:
      - MIN_TRADES=60: no trade until 60 entropy-valid ticks processed
      - Entry at pair confirmation: open LONG on bull→neutral, SHORT on bear→neutral
      - Direct jump filter: bull→bear / bear→bull ignored (clears pending flags)
      - TOL_CLOSE filter on exit: abs(P - 0.51) <= 0.0153
      - Immediate re-entry: CLOSE_LONG → SHORT, CLOSE_SHORT → LONG (pair already confirmed)
      - bull_pair_count / bear_pair_count tracked per position
    """

    def __init__(self):
        self.version          = 'v3'
        self.position         = None
        self.last_trade_id    = None
        self.trade_log        = []
        self.total_trades     = 0
        self.winning_trades   = 0
        self.losing_trades    = 0
        self.flat_trades      = 0
        self.spot_pnl         = 0.0   # LONG only (real)
        self.synthetic_pnl    = 0.0   # SHORT only (synthetic)
        self.force_closes     = 0
        self._pending_long    = False  # neutral→bull seen, waiting for bull→neutral confirmation
        self._pending_short   = False  # neutral→bear seen, waiting for bear→neutral confirmation
        self._last_open_name  = None

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
        pnl_pips = round(pnl * 10000, 1)
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        else:
            self.flat_trades += 1
        if pos.side == 'LONG':
            self.spot_pnl += pnl
        else:
            self.synthetic_pnl += pnl
        if reason == 'force':
            self.force_closes += 1
        self.trade_log.append({
            'side':             pos.side,
            'entry':            pos.entry_price,
            'exit':             price,
            'pnl':              pnl,
            'pnl_pips':         pnl_pips,
            'pnl_pct':          pnl_pct,
            'entry_transition': pos.entry_transition,
            'nn_count':         pos.neutral_neutral_count,
            'bull_pairs':       pos.bull_pair_count,
            'bear_pairs':       pos.bear_pair_count,
            'close_reason':     reason,
        })
        self.position = None

    def force_close(self, price):
        if self.position:
            self._close(price, -1, 'force')

    def process_signal(self, t):
        tid           = t['trade_id']
        price         = t['price']
        name          = t['transition_name']
        P             = t['P']
        ts            = t['timestamp']
        entropy_count = t['entropy_count']

        # Dedup guard
        if self.last_trade_id is not None and tid <= self.last_trade_id:
            return
        self.last_trade_id = tid

        # MIN_TRADES: wait for SKA convergence
        if entropy_count < MIN_TRADES:
            return

        # Direct jump filter: localized entropy shocks — clear pending, ignore
        if name in ('bull→bear', 'bear→bull'):
            self._pending_long  = False
            self._pending_short = False
            return

        # ΔP_pair tracking for bull/bear pair counts (while position is open)
        if name in ('neutral→bull', 'neutral→bear'):
            self._last_open_name = name
        elif name == 'bull→neutral' and self._last_open_name == 'neutral→bull':
            if self.position is not None:
                self.position.bull_pair_count += 1
            self._last_open_name = None
        elif name == 'bear→neutral' and self._last_open_name == 'neutral→bear':
            if self.position is not None:
                self.position.bear_pair_count += 1
            self._last_open_name = None

        # === NO POSITION: wait for pair confirmation before entering ===
        if self.position is None:
            if name == 'neutral→bull':
                self._pending_long  = True
                self._pending_short = False
            elif name == 'neutral→bear':
                self._pending_short = True
                self._pending_long  = False
            elif name == 'bull→neutral' and self._pending_long:
                # Bull pair confirmed — enter LONG already in IN_NEUTRAL state
                self._open('LONG', price, tid, ts, 'bull→neutral')
                self.position.exit_state      = IN_NEUTRAL
                self.position.bull_pair_count = 1
                self._pending_long = False
            elif name == 'bear→neutral' and self._pending_short:
                # Bear pair confirmed — enter SHORT already in IN_NEUTRAL state
                self._open('SHORT', price, tid, ts, 'bear→neutral')
                self.position.exit_state      = IN_NEUTRAL
                self.position.bear_pair_count = 1
                self._pending_short = False
            return

        # === LONG POSITION ===
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
                if name == 'bear→neutral' and P is not None and abs(P - P_X_NEUTRAL) <= TOL_CLOSE:
                    self._close(price, tid, name)
                    # bear→neutral confirms bear pair — immediately enter SHORT
                    self._open('SHORT', price, tid, ts, 'bear→neutral')
                    self.position.exit_state      = IN_NEUTRAL
                    self.position.bear_pair_count = 1
                elif name == 'neutral→bull':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0

        # === SHORT POSITION ===
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
                if name == 'bull→neutral' and P is not None and abs(P - P_X_NEUTRAL) <= TOL_CLOSE:
                    self._close(price, tid, name)
                    # bull→neutral confirms bull pair — immediately enter LONG
                    self._open('LONG', price, tid, ts, 'bull→neutral')
                    self.position.exit_state      = IN_NEUTRAL
                    self.position.bull_pair_count = 1
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
            'side', 'entry', 'exit', 'pnl', 'pnl_pips', 'pnl_pct',
            'entry_transition', 'nn_count', 'bull_pairs', 'bear_pairs', 'close_reason'
        ])
        writer.writeheader()
        writer.writerows(bot.trade_log)


def main():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')])

    summary_rows = []
    agg = {
        'trades': 0, 'winners': 0, 'losers': 0, 'flat': 0,
        'spot_pnl': 0.0, 'synthetic_pnl': 0.0, 'force': 0
    }

    print(f"\nSKA Trading Bot v3 — Backtest")
    print(f"MIN_TRADES={MIN_TRADES} | MIN_NN_COUNT={MIN_NN_COUNT} | TOL_CLOSE={TOL_CLOSE}")
    print(f"{'='*80}")
    print(f"  {'file':<45} {'n':>4} {'W':>4} {'L':>4} {'F':>4} {'win%':>6} {'pips':>8} {'avg':>6}")
    print(f"  {'-'*45} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*8} {'-'*6}")

    for fname in files:
        bot = BotV3()
        run_file(os.path.join(INPUT_DIR, fname), bot)
        save_trades(bot, fname)

        n      = bot.total_trades
        wr     = (bot.winning_trades / n * 100) if n > 0 else 0
        pips   = round((bot.spot_pnl + bot.synthetic_pnl) * 10000, 1)
        avg    = pips / n if n > 0 else 0
        print(
            f"  {fname:<45} {n:>4} {bot.winning_trades:>4} {bot.losing_trades:>4} "
            f"{bot.flat_trades:>4} {wr:>6.1f} {pips:>+8.1f} {avg:>+6.2f}"
        )

        summary_rows.append({
            'file':          fname,
            'version':       'v3',
            'trades':        n,
            'winners':       bot.winning_trades,
            'losers':        bot.losing_trades,
            'flat':          bot.flat_trades,
            'force_closes':  bot.force_closes,
            'win_rate':      round(wr, 1),
            'spot_pnl_pips': round(bot.spot_pnl * 10000, 1),
            'synth_pnl_pips':round(bot.synthetic_pnl * 10000, 1),
            'total_pips':    round(pips, 1),
            'avg_pips':      round(avg, 2),
        })
        agg['trades']       += n
        agg['winners']      += bot.winning_trades
        agg['losers']       += bot.losing_trades
        agg['flat']         += bot.flat_trades
        agg['spot_pnl']     += bot.spot_pnl
        agg['synthetic_pnl']+= bot.synthetic_pnl
        agg['force']        += bot.force_closes

    total_pips = round((agg['spot_pnl'] + agg['synthetic_pnl']) * 10000, 1)
    wr  = (agg['winners'] / agg['trades'] * 100) if agg['trades'] > 0 else 0
    avg = total_pips / agg['trades'] if agg['trades'] > 0 else 0
    print(f"\n{'='*80}")
    print(
        f"  TOTAL {agg['trades']:>4} trades | "
        f"W={agg['winners']} L={agg['losers']} F={agg['flat']} | "
        f"win={wr:.1f}% | PnL={total_pips:+.1f} pips | avg={avg:+.2f}"
    )
    print(
        f"  LONG (spot):  {round(agg['spot_pnl']*10000,1):+.1f} pips  |  "
        f"SHORT (synth): {round(agg['synthetic_pnl']*10000,1):+.1f} pips"
    )
    print(f"  Force closes: {agg['force']}")

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'version', 'trades', 'winners', 'losers', 'flat',
            'force_closes', 'win_rate', 'spot_pnl_pips', 'synth_pnl_pips',
            'total_pips', 'avg_pips'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved: {SUMMARY_FILE}")

    plot_nn_distribution()


def plot_nn_distribution():
    """Plot nn_count distribution (winners vs losers) across all result files."""
    results_dir = os.path.join(OUTPUT_DIR, 'v3')
    if not os.path.isdir(results_dir):
        return
    files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    if not files:
        return

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

    if not win_counts and not lose_counts:
        return

    max_nn = max(list(win_counts.keys()) + list(lose_counts.keys()))
    xs = list(range(max_nn + 1))
    wins   = [win_counts.get(x, 0)  for x in xs]
    loses  = [lose_counts.get(x, 0) for x in xs]
    totals = [w + l for w, l in zip(wins, loses)]
    win_rates = [w / t * 100 if t > 0 else 0 for w, t in zip(wins, totals)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#FFFFFF')

    ax1.bar(xs, wins,  color='#A8DFBC', label='Winners', alpha=0.9)
    ax1.bar(xs, loses, bottom=wins, color='#FFAAAA', label='Losers', alpha=0.9)
    ax1.axvline(MIN_NN_COUNT - 0.5, color='#CC2222', linewidth=1.5, linestyle='--',
                label=f'MIN_NN_COUNT={MIN_NN_COUNT}')
    ax1.set_ylabel('Trade count')
    ax1.set_title('Neutral Gap Distribution — Winners vs Losers (v3)', fontweight='bold')
    ax1.legend()
    ax1.set_facecolor('#F8F8F8')

    ax2.bar(xs, win_rates, color='#5599CC', alpha=0.85)
    ax2.axhline(50, color='#888888', linewidth=1, linestyle=':')
    ax2.axvline(MIN_NN_COUNT - 0.5, color='#CC2222', linewidth=1.5, linestyle='--',
                label=f'MIN_NN_COUNT={MIN_NN_COUNT}')
    ax2.set_ylabel('Win rate (%)')
    ax2.set_xlabel('nn_count (neutral→neutral transitions in neutral gap)')
    ax2.set_title('Win Rate by Neutral Gap Length (v3)', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.set_facecolor('#F8F8F8')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'nn_distribution_v3.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")

    plot_nn_scatter()


def plot_nn_scatter():
    """Violin plot: nn_count distribution for winners vs losers."""
    results_dir = os.path.join(OUTPUT_DIR, 'v3')
    files = sorted(f for f in os.listdir(results_dir) if f.endswith('.csv'))

    nn_win, nn_lose = [], []

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            for row in csv.DictReader(f):
                nn  = int(row['nn_count'])
                pnl = float(row['pnl'])
                if pnl > 0:
                    nn_win.append(nn)
                elif pnl < 0:
                    nn_lose.append(nn)

    if not nn_win or not nn_lose:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8F8F8')

    parts = ax.violinplot([nn_lose, nn_win], positions=[1, 2], showmedians=True, showextrema=True)
    parts['bodies'][0].set_facecolor('#FF8888')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor('#66BB88')
    parts['bodies'][1].set_alpha(0.7)
    for pc in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        parts[pc].set_color('#444444')
        parts[pc].set_linewidth(1.5)

    ax.axhline(MIN_NN_COUNT, color='#CC2222', linewidth=2.0, linestyle='--',
               label=f'Threshold (N≥{MIN_NN_COUNT})')
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Losers\n(n={len(nn_lose)})', f'Winners\n(n={len(nn_win)})'], fontsize=12)
    ax.set_ylabel('nn_count (neutral gap length)', fontsize=11)
    ax.set_title('Neutral Gap Distribution — Winners vs Losers (v3)', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)

    med_lose = sorted(nn_lose)[len(nn_lose)//2]
    med_win  = sorted(nn_win)[len(nn_win)//2]
    ax.text(1, med_lose + 0.5, f'median={med_lose}', ha='center', fontsize=9, color='#AA2222')
    ax.text(2, med_win  + 0.5, f'median={med_win}',  ha='center', fontsize=9, color='#226622')

    plt.tight_layout()
    outfile = os.path.join(OUTPUT_DIR, 'nn_scatter_v3.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {outfile}")


if __name__ == '__main__':
    main()
