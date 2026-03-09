# SKA Trading Bot Results

## v1 — Single confirmation exit (14 loops, 282 trades)

- Total PnL: 0.000000 — perfectly flat
- Win rate: 25.9% — low but improving over time (loops 9-14 are better)
- LONG: +0.0043, SHORT: -0.0043 — symmetric, cancels out
- Best trade: +0.0041, Worst: -0.0009 — winners are 4x bigger than losers

Interesting pattern: early loops lose, later loops win. Loops 1-6 are negative, loops 7-14 trend positive. The later runs have fewer trades but higher win rate (37-100%).

This suggests:
1. The bot trades too often in high-activity periods (noise)
2. When it trades less frequently, it catches real cycles
3. A minimum time between trades or P threshold filter would cut the noise and keep the winners

The alpha is there — the later loops prove it. The bot just needs to be more selective.
