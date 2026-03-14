# SKA Trading Bot Results

## Framework

**Entropic Trading** — uses entropy dynamics as the signal axis instead of price.
The source of the alpha is the market's own learning process, not price levels or volume.

**Paired Cycle Trading (PCT)** — the specific strategy implemented here.
Entry and exit are defined by complete paired regime cycles in the TradeID Series.
The bot is structurally blind to the neutral→neutral baseline by design — it trades only
the 10% of transitions that carry directional information.

This is not HFT. It is event-driven structural trading — the signal fires on a topological
event (completion of a paired regime cycle), not on a threshold or a price level.

---

## Bot Versions

### v1 — Single opposite signal closes

```
LONG:   neutral→bull  (OPEN)
        neutral→bear OR bear→neutral  (CLOSE — first opposite signal)

SHORT:  neutral→bear  (OPEN)
        neutral→bull OR bull→neutral  (CLOSE — first opposite signal)
```

No confirmation required. Exits immediately on the first opposite-side transition.
Fast exits, small losses per trade, but low win rate — exits before own cycle completes.

---

### v2 — 2 consecutive opposite signals (structurally wrong)

```
LONG:   neutral→bull  (OPEN)
        2 consecutive opposite signals  (CLOSE)

SHORT:  neutral→bear  (OPEN)
        2 consecutive opposite signals  (CLOSE)
```

Counts any two opposite-side signals consecutively — does not require a proper paired cycle.
Fewer trades than v1, higher win rate, but the double confirmation is not structurally meaningful.
Included for comparison.

---

### v3 — Full opposite paired cycle required

```
LONG:   neutral→bull    (OPEN — UP cycle opens)
        bull→neutral    (UP pair confirmed)
        neutral→bear    (DOWN cycle opens)
        bear→neutral    (DOWN pair confirmed — CLOSE)

SHORT:  neutral→bear    (OPEN — DOWN cycle opens)
        bear→neutral    (DOWN pair confirmed)
        neutral→bull    (UP cycle opens)
        bull→neutral    (UP pair confirmed — CLOSE)
```

Theoretically correct — exit requires the complete opposite paired cycle.
Best total PnL on high liquidity data. Worst losses are small (-0.0001 to -0.0002).
On low liquidity data (current 2026 market) exits are too rare — positions force-close.

---

### v4 — Own pair confirmed, first opposite signal closes (hybrid)

```
LONG:   neutral→bull    (OPEN — WAIT_PAIR_CONFIRM)
        bull→neutral    (UP pair confirmed — READY_TO_EXIT)
        neutral→bear OR bear→neutral  (CLOSE — first opposite signal)

SHORT:  neutral→bear    (OPEN — WAIT_PAIR_CONFIRM)
        bear→neutral    (DOWN pair confirmed — READY_TO_EXIT)
        neutral→bull OR bull→neutral  (CLOSE — first opposite signal)
```

Hybrid — requires own pair confirmation before exit is active, then exits on first opposite signal.
Intended to combine v3 entry quality with v1 exit speed.
Worst performer on high liquidity data: accumulates large losses (-0.001 to -0.003) while waiting for pair confirmation on orphaned cycles.

---

## Live Trading Results

### v1 — Single confirmation exit (14 loops, 282 trades)

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

### v2 — Double confirmation exit (14 loops, 73 trades)

- Total PnL: -0.008300 — negative
- Win rate: 31.5% — improved from v1 (25.9%)
- LONG: -0.0065, SHORT: -0.0018
- Best trade: +0.0023, Worst: -0.0023 — symmetric risk/reward

v2 trades 4x less than v1 (73 vs 282) — the 2-confirmation filter reduces noise. Win rate improved. But total PnL went negative because the bot exits too late — by the time 2 opposite signals arrive, price has moved against the position.

Conclusion: v2 logic is structurally wrong. It counts any two opposite-side signals to close — but the two signals do not need to form a proper paired cycle. This means v2 can close on a mix of unrelated transitions, not on a real structural reversal. The double confirmation delays the exit without adding structural meaning, which is why PnL is worse than v1.

The correct approach is not faster or slower exits — it is requiring the **complete opposite paired cycle** as the exit condition. This is v3.

### v3 — Full opposite paired cycle exit (2 loops, 48 trades)

Signal logic:
- LONG:  neutral→bull → bull→neutral → neutral→bear → bear→neutral (CLOSE)
- SHORT: neutral→bear → bear→neutral → neutral→bull → bull→neutral (CLOSE)

Exit requires the **complete opposite paired cycle** — not just one or two opposite signals but the full structural sequence. State machine with 3 states: WAIT_PAIR_CONFIRM → WAIT_OPP_OPEN → WAIT_OPP_CONFIRM.

Rationale: v1 and v2 both close on incomplete structural events. v3 is the theoretically correct implementation — the exit is triggered only when the opposite paired regime cycle is fully confirmed, which is the structural definition of the alpha.

### Results (2 loops, 48 trades)

- Total PnL: -0.003900
- Win rate: 16.7% — lowest of all versions
- Winners: 8 | Losers: 36 | Flat: 4
- Avg PnL/trade: -0.000081
- Best trade: +0.001200 | Worst trade: -0.000600
- LONG:  24 trades | PnL=-0.001400 | win_rate=16.7%
- SHORT: 24 trades | PnL=-0.002500 | win_rate=16.7%

Per loop:
- Loop 1: 36 trades | win=16.7% | PnL=-0.002200
- Loop 2: 12 trades | win=16.7% | PnL=-0.001700

### Analysis

Win rate dropped from v1 (25.9%) → v2 (31.5%) → v3 (16.7%). Requiring the full opposite paired cycle (4 transitions) makes exits too strict:

1. The 4-transition sequence is rare within a 3500-trade window — many positions reach end-of-run without a clean exit
2. Holding through the full opposite cycle allows losses to accumulate while waiting for confirmation
3. The 4 flat trades are end-of-run force closes — positions that never found their exit signal
4. Winners (best +0.0012) are 2x larger than losers (worst -0.0006) but the ratio of losers (36) to winners (8) overwhelms the size advantage

Conclusion: the theoretically correct exit is not the practically optimal exit. The market does not always complete the full opposite cycle within the run window. A hybrid approach may be needed — require pair confirmation (bull→neutral) before listening for exit, but exit on the first opposite signal (neutral→bear OR bear→neutral) rather than waiting for the full opposite paired cycle.

### v4 — Own pair confirmed, first opposite signal closes (in progress)

Signal logic:
- LONG:  neutral→bull → bull→neutral (confirmed) → neutral→bear OR bear→neutral (CLOSE)
- SHORT: neutral→bear → bear→neutral (confirmed) → neutral→bull OR bull→neutral (CLOSE)

State machine: WAIT_PAIR_CONFIRM → READY_TO_EXIT → CLOSE on first opposite signal.

Rationale: v3 exits are too rare on low liquidity data. v4 keeps the structural entry quality (own pair must confirm) but exits on the first opposite signal rather than waiting for the full opposite cycle. Results pending.
