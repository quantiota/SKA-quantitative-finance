import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_DIR      = '/batch_trading_bot/XRPUSDT/20260406'
OUTPUT_DIR     = '/batch_trading_bot/results'
SUMMARY_FILE   = '/batch_trading_bot/summary_v4.csv'
BASELINE_FILE  = '/batch_trading_bot/summary.csv'

WAIT_PAIR  = 'WAIT_PAIR'
IN_NEUTRAL = 'IN_NEUTRAL'
READY      = 'READY'
EXIT_WAIT  = 'EXIT_WAIT'

# ─── constants (same as backtest.py / trading_bot_v3.py) ─────────────────────
MIN_NN_COUNT = 10
MIN_TRADES   = 50

P_X_NEUTRAL  = 0.51
K            = 0.03
TOL_BEAR     = K * 0.14   # 0.0042
TOL_BULL     = K * 0.66   # 0.0198
TOL_CLOSE    = K * 0.51   # 0.0153

BULL_THRESHOLD = -0.34
BEAR_THRESHOLD = -0.86

TRANSITION_NAMES = {
    0: 'neutral→neutral', 1: 'neutral→bull', 2: 'neutral→bear',
    3: 'bull→neutral',    4: 'bull→bull',    5: 'bull→bear',
    6: 'bear→neutral',    7: 'bear→bull',    8: 'bear→bear'
}


# ─── Transition computation ───────────────────────────────────────────────────

def compute_transitions(rows):
    transitions   = []
    prev_regime   = None
    entropy_buf   = []
    entropy_count = 0

    for row in rows:
        price = float(row['price'])
        try:
            entropy = float(row['entropy']) if row['entropy'] not in ('', 'None', None) else None
        except ValueError:
            entropy = None

        if entropy is None:
            prev_regime = None
            continue

        entropy_count += 1
        entropy_buf.append(entropy)
        if len(entropy_buf) > 3:
            entropy_buf.pop(0)

        if len(entropy_buf) < 3:
            prev_regime = None
            continue

        e2, e1, e = entropy_buf
        P      = math.exp(-abs((e  - e1) / e )) if e  != 0 else None
        prev_P = math.exp(-abs((e1 - e2) / e1)) if e1 != 0 else None

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


# ─── Bot v4 — challenge backtest framework ──────────────────────────────────

class BotV4:
    """
    Challenge backtest framework.

    Entry: LONG at neutral→bull (WAIT_PAIR), SHORT at neutral→bear (WAIT_PAIR).
    The false-start rule is intentionally not included in this version.
    """

    def __init__(self):
        self.version          = 'v4'
        self.position         = None
        self.last_trade_id    = None
        self.trade_log        = []
        self.total_trades     = 0
        self.winning_trades   = 0
        self.losing_trades    = 0
        self.flat_trades      = 0
        self.spot_pnl         = 0.0
        self.synthetic_pnl    = 0.0
        self.force_closes     = 0
        self._last_open_name  = None

    def _open(self, side, price, trade_id, ts, transition):
        self.position = Position(
            side=side, entry_price=price,
            entry_trade_id=trade_id, entry_time=ts,
            entry_transition=transition,
            exit_state=WAIT_PAIR,
        )

    def _close(self, price, trade_id, reason):
        pos = self.position
        pnl = price - pos.entry_price if pos.side == 'LONG' else pos.entry_price - price
        pnl_pct  = (pnl / pos.entry_price) * 100
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

        if self.last_trade_id is not None and tid <= self.last_trade_id:
            return
        self.last_trade_id = tid

        if entropy_count < MIN_TRADES:
            return

        # Direct jump filter
        if name in ('bull→bear', 'bear→bull'):
            return

        # ΔP_pair tracking
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

        # === NO POSITION: open at first signal ===
        if self.position is None:
            if name == 'neutral→bull':
                self._open('LONG', price, tid, ts, name)
            elif name == 'neutral→bear':
                self._open('SHORT', price, tid, ts, name)
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
                    # bear pair confirmed — open SHORT immediately at same price
                    if self.position is None:
                        self._open('SHORT', price, tid, ts, 'neutral→bear')
                        self.position.exit_state = IN_NEUTRAL
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
                    # bull pair confirmed — open LONG immediately at same price
                    if self.position is None:
                        self._open('LONG', price, tid, ts, 'neutral→bull')
                        self.position.exit_state = IN_NEUTRAL
                        self.position.bull_pair_count = 1
                elif name == 'neutral→bear':
                    self.position.exit_state = WAIT_PAIR
                    self.position.neutral_neutral_count = 0


# ─── Main backtest runner ─────────────────────────────────────────────────────

def compute_available_pips(rows):
    """Sum of all absolute price moves — theoretical maximum capturable profit."""
    prices = []
    for row in rows:
        try:
            prices.append(float(row['price']))
        except (ValueError, KeyError):
            pass
    return sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices))) * 10000


def run_file(filepath, bot):
    with open(filepath) as f:
        rows = list(csv.DictReader(f))
    available_pips = compute_available_pips(rows)
    transitions = compute_transitions(rows)
    for t in transitions:
        bot.process_signal(t)
    if transitions:
        bot.force_close(transitions[-1]['price'])
    return available_pips


def save_trades(bot, filename):
    if not bot.trade_log:
        return
    os.makedirs(os.path.join(OUTPUT_DIR, bot.version), exist_ok=True)
    path = os.path.join(OUTPUT_DIR, bot.version, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'side', 'entry', 'exit', 'pnl', 'pnl_pips', 'pnl_pct',
            'entry_transition', 'nn_count', 'bull_pairs', 'bear_pairs',
            'close_reason'
        ])
        writer.writeheader()
        writer.writerows(bot.trade_log)


def main():
    files = sorted([f for f in os.listdir(INPUT_DIR)
                    if f.startswith('binance_trades_') and f.endswith('.csv')])

    summary_rows = []
    agg = {
        'trades': 0, 'winners': 0, 'losers': 0, 'flat': 0,
        'spot_pnl': 0.0, 'synthetic_pnl': 0.0, 'force': 0,
        'available_pips': 0.0,
    }

    print(f"\nSKA Trading Bot v4 — Challenge Backtest")
    print(f"MIN_TRADES={MIN_TRADES} | MIN_NN_COUNT={MIN_NN_COUNT} | TOL_CLOSE={TOL_CLOSE}")
    print(f"{'='*90}")
    print(f"  {'file':<45} {'n':>4} {'W':>4} {'L':>4} {'F':>4} {'win%':>6} {'pips':>8} {'avg':>6} {'avail':>8} {'cap%':>6}")
    print(f"  {'-'*45} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*6}")

    for fname in files:
        bot = BotV4()
        available_pips = run_file(os.path.join(INPUT_DIR, fname), bot)
        save_trades(bot, fname)

        n      = bot.total_trades
        wr     = (bot.winning_trades / n * 100) if n > 0 else 0
        pips   = round((bot.spot_pnl + bot.synthetic_pnl) * 10000, 1)
        avg    = pips / n if n > 0 else 0
        avail  = round(available_pips, 1)
        cap    = (pips / avail * 100) if avail > 0 else 0
        print(
            f"  {fname:<45} {n:>4} {bot.winning_trades:>4} {bot.losing_trades:>4} "
            f"{bot.flat_trades:>4} {wr:>6.1f} {pips:>+8.1f} {avg:>+6.2f} "
            f"{avail:>8.1f} {cap:>+6.2f}"
        )

        summary_rows.append({
            'file':              fname,
            'version':           'v4',
            'trades':            n,
            'winners':           bot.winning_trades,
            'losers':            bot.losing_trades,
            'flat':              bot.flat_trades,
            'force_closes':      bot.force_closes,
            'win_rate':          round(wr, 1),
            'spot_pnl_pips':     round(bot.spot_pnl * 10000, 1),
            'synth_pnl_pips':    round(bot.synthetic_pnl * 10000, 1),
            'total_pips':        round(pips, 1),
            'avg_pips':          round(avg, 2),
            'available_pips':    round(avail, 1),
            'capture_rate_pct':  round(cap, 2),
        })
        agg['trades']          += n
        agg['winners']         += bot.winning_trades
        agg['losers']          += bot.losing_trades
        agg['flat']            += bot.flat_trades
        agg['spot_pnl']        += bot.spot_pnl
        agg['synthetic_pnl']   += bot.synthetic_pnl
        agg['force']           += bot.force_closes
        agg['available_pips']  += available_pips

    total_pips     = round((agg['spot_pnl'] + agg['synthetic_pnl']) * 10000, 1)
    total_avail    = round(agg['available_pips'], 1)
    total_cap      = (total_pips / total_avail * 100) if total_avail > 0 else 0
    wr  = (agg['winners'] / agg['trades'] * 100) if agg['trades'] > 0 else 0
    avg = total_pips / agg['trades'] if agg['trades'] > 0 else 0
    print(f"\n{'='*90}")
    print(
        f"  TOTAL {agg['trades']:>4} trades | "
        f"W={agg['winners']} L={agg['losers']} F={agg['flat']} | "
        f"win={wr:.1f}% | PnL={total_pips:+.1f} pips | avg={avg:+.2f}"
    )
    print(
        f"  LONG (spot):  {round(agg['spot_pnl']*10000,1):+.1f} pips  |  "
        f"SHORT (synth): {round(agg['synthetic_pnl']*10000,1):+.1f} pips"
    )
    print(f"  Available pips: {total_avail:.1f} | Capture rate: {total_cap:+.2f}%")
    print(f"  Force closes: {agg['force']}")

    # ── compare against v3 baseline ──────────────────────────────────────────
    if BASELINE_FILE and os.path.exists(BASELINE_FILE):
        print(f"\n{'─'*80}")
        print(f"  Comparison: v4 vs v3 baseline")
        print(f"{'─'*80}")
        v3_map = {}
        with open(BASELINE_FILE) as f:
            for row in csv.DictReader(f):
                v3_map[row['file']] = row
        delta_pips = 0.0
        for row in summary_rows:
            v3 = v3_map.get(row['file'])
            if v3 is None:
                continue
            d = row['total_pips'] - float(v3['total_pips'])
            delta_pips += d
        print(f"  Total PnL delta (v4 − v3): {delta_pips:+.1f} pips")
        v3_total = sum(float(r['total_pips']) for r in v3_map.values())
        print(f"  v3 total: {v3_total:+.1f} pips  |  v4 total: {total_pips:+.1f} pips")

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'file', 'version', 'trades', 'winners', 'losers', 'flat',
            'force_closes', 'win_rate',
            'spot_pnl_pips', 'synth_pnl_pips', 'total_pips', 'avg_pips',
            'available_pips', 'capture_rate_pct'
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved: {SUMMARY_FILE}")

    results_dir = os.path.join(OUTPUT_DIR, 'v4')
    input_files = set(os.listdir(INPUT_DIR))
    files = sorted(f for f in os.listdir(results_dir) if f.endswith('.csv') and f in input_files)
    write_trades_txt(files, results_dir)


def write_trades_txt(files, results_dir):
    """Write all trades from results/v4/ into a single human-readable txt file."""
    txt_path = os.path.join(OUTPUT_DIR, 'trades_report.txt')
    with open(txt_path, 'w') as out:
        out.write(f"{'SIDE':<6}  {'ENTRY':<8}  {'EXIT':<8}  {'PIPS':>7}  {'CLOSE_REASON'}\n")
        out.write(f"{'------':<6}  {'--------':<8}  {'--------':<8}  {'-------':>7}  {'------------'}\n")
        for fname in files:
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                for row in csv.DictReader(f):
                    pips = float(row['pnl_pips'])
                    out.write(
                        f"{row['side']:<6}  {row['entry']:<8}  {row['exit']:<8}  "
                        f"{pips:>+7.1f}  {row['close_reason']}\n"
                    )
    print(f"Trades report saved: {txt_path}")
    write_submission_csv(files, results_dir)


def write_submission_csv(files, results_dir):
    """Write all trades as submission.csv for Kaggle evaluation."""
    csv_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    row_id = 0
    with open(csv_path, 'w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['id', 'side', 'entry', 'exit', 'pnl_pips', 'close_reason'])
        writer.writeheader()
        for fname in files:
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                for row in csv.DictReader(f):
                    writer.writerow({
                        'id':           row_id,
                        'side':         row['side'],
                        'entry':        row['entry'],
                        'exit':         row['exit'],
                        'pnl_pips':     row['pnl_pips'],
                        'close_reason': row['close_reason'],
                    })
                    row_id += 1
    print(f"Submission saved:    {csv_path}")


if __name__ == '__main__':
    main()
