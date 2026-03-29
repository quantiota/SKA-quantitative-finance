"""
SKA Backtest v3 — Report from summary.csv
Usage: /opt/venv/bin/python3 report.py
"""

import csv
import os

SUMMARY_FILE = os.path.join(os.path.dirname(__file__), 'summary.csv')

with open(SUMMARY_FILE) as f:
    rows = [r for r in csv.DictReader(f) if int(r['trades']) > 0]

total_trades  = sum(int(r['trades'])   for r in rows)
total_winners = sum(int(r['winners'])  for r in rows)
total_losers  = sum(int(r['losers'])   for r in rows)
total_flat    = sum(int(r['flat'])     for r in rows)
total_pips    = sum(float(r['total_pips']) for r in rows)
spot_pips     = sum(float(r['spot_pnl_pips'])  for r in rows)
synth_pips    = sum(float(r['synth_pnl_pips']) for r in rows)
win_rate      = total_winners / total_trades * 100 if total_trades else 0
avg_pips      = total_pips / total_trades if total_trades else 0

best  = max(rows, key=lambda r: float(r['total_pips']))
worst = min(rows, key=lambda r: float(r['total_pips']))

print(f"\nSKA Trading Bot v3 — Backtest Report ({len(rows)} loops)")
print("=" * 55)
print(f"  Total trades : {total_trades}")
print(f"  Winners      : {total_winners} | Losers: {total_losers} | Flat: {total_flat}")
print(f"  Win rate     : {win_rate:.1f}%")
print(f"  Total PnL    : {total_pips:+.1f} pips")
print(f"  Avg / trade  : {avg_pips:+.2f} pips")
print(f"  LONG  (spot) : {spot_pips:+.1f} pips")
print(f"  SHORT (synth): {synth_pips:+.1f} pips")
print("-" * 55)
print(f"  Best loop  : {best['file']}")
print(f"    {best['trades']} trades | PnL={float(best['total_pips']):+.1f} pips | avg={float(best['avg_pips']):+.2f}")
print(f"  Worst loop : {worst['file']}")
print(f"    {worst['trades']} trades | PnL={float(worst['total_pips']):+.1f} pips | avg={float(worst['avg_pips']):+.2f}")
print("=" * 55)

print(f"\n  {'#':<4} {'file':<45} {'n':>4} {'W':>4} {'L':>4} {'F':>4} {'win%':>6} {'pips':>8} {'avg':>6}")
print(f"  {'-'*4} {'-'*45} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*8} {'-'*6}")
for i, r in enumerate(rows, 1):
    print(
        f"  {i:<4} {r['file']:<45} {r['trades']:>4} {r['winners']:>4} "
        f"{r['losers']:>4} {r['flat']:>4} {r['win_rate']:>6} "
        f"{float(r['total_pips']):>+8.1f} {float(r['avg_pips']):>+6.2f}"
    )
print()
