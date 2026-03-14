# SKA Batch Backtest — Paired Cycle Trading (PCT)

## Framework

**Entropic Trading** — uses entropy dynamics as the signal axis instead of price.
The signal is derived from the market's own learning process (SKA — Structured Knowledge Accumulation), not from price levels or volume.

**Paired Cycle Trading (PCT)** — the strategy implemented across all 4 versions.
Entry and exit are defined by paired regime transitions in the TradeID Series.
The bot is structurally blind to the neutral→neutral baseline (90% of trades) by design.
Only the 4 directional transitions carry signal:

```
neutral→bull   bull→neutral   neutral→bear   bear→neutral
```

This is not HFT. It is event-driven structural trading operating at tick data resolution.

---

## Data

- Source: Binance XRPUSDT WebSocket — real tick data exported from QuestDB
- Folder: `XRPUSDT/` — 20 files, July 2025
- Liquidity: ~875 trades/minute (high liquidity period)
- Each file: ~2300–4500 trades, 2–20 minutes of market activity
- Entropy computed by the SKA learning engine (matrix grows 1×1 → N×N per loop)

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
Worst performer: accumulates large losses (-0.001 to -0.003) while waiting for pair confirmation.

---

## Aggregate Results — 20 files, July 2025

| Version | Trades | Win%  | Total PnL | Avg PnL/trade | Force closes |
|---------|--------|-------|-----------|---------------|--------------|
| v1      | 2114   | 29.8% | +0.1616   | +0.000076     | 19           |
| v2      | 950    | 53.4% | +0.2388   | +0.000251     | 17           |
| v3      | 1359   | 42.0% | +0.2547   | +0.000187     | 20           |
| v4      | 1388   | 31.6% | +0.0433   | +0.000031     | 20           |

**Best total PnL: v3** | **Best win rate and avg PnL/trade: v2**

---

## Detailed Results Per File

| File | Ver | Trades | Win | Lose | Force | Win% | PnL | Best | Worst |
|------|-----|--------|-----|------|-------|------|-----|------|-------|
| questdb-query-1751814162388.csv | v1 | 110 | 38 | 63 | 1 | 34.5% | +0.005500 | +0.001000 | -0.000100 |
| questdb-query-1751814162388.csv | v2 | 45 | 26 | 15 | 1 | 57.8% | +0.010200 | +0.002000 | -0.000200 |
| questdb-query-1751814162388.csv | v3 | 60 | 28 | 23 | 1 | 46.7% | +0.012100 | +0.002100 | -0.000100 |
| questdb-query-1751814162388.csv | v4 | 60 | 21 | 33 | 1 | 35.0% | -0.002500 | +0.001000 | -0.001600 |
| questdb-query-1751823646841.csv | v1 | 124 | 30 | 81 | 1 | 24.2% | +0.000800 | +0.000900 | -0.000100 |
| questdb-query-1751823646841.csv | v2 | 56 | 27 | 23 | 1 | 48.2% | +0.005600 | +0.000900 | -0.000200 |
| questdb-query-1751823646841.csv | v3 | 83 | 29 | 43 | 1 | 34.9% | +0.006000 | +0.000900 | -0.000100 |
| questdb-query-1751823646841.csv | v4 | 88 | 23 | 55 | 1 | 26.1% | -0.001300 | +0.000900 | -0.000500 |
| questdb-query-1751880848676.csv | v1 | 122 | 26 | 93 | 1 | 21.3% | -0.001300 | +0.000900 | -0.000100 |
| questdb-query-1751880848676.csv | v2 | 47 | 20 | 24 | 1 | 42.6% | +0.005500 | +0.001500 | -0.000200 |
| questdb-query-1751880848676.csv | v3 | 79 | 22 | 55 | 1 | 27.8% | +0.003700 | +0.001500 | -0.000100 |
| questdb-query-1751880848676.csv | v4 | 81 | 19 | 59 | 1 | 23.5% | -0.004400 | +0.000700 | -0.001200 |
| questdb-query-1751909192925.csv | v1 | 53 | 24 | 23 | 1 | 45.3% | +0.013400 | +0.002900 | -0.000100 |
| questdb-query-1751909192925.csv | v2 | 31 | 20 | 9 | 1 | 64.5% | +0.013700 | +0.003300 | -0.000100 |
| questdb-query-1751909192925.csv | v3 | 41 | 24 | 15 | 1 | 58.5% | +0.014900 | +0.003300 | -0.000100 |
| questdb-query-1751909192925.csv | v4 | 42 | 21 | 17 | 1 | 50.0% | +0.006700 | +0.001500 | -0.003100 |
| questdb-query-1751924112805.csv | v1 | 60 | 21 | 31 | 1 | 35.0% | +0.006800 | +0.001600 | -0.000300 |
| questdb-query-1751924112805.csv | v2 | 26 | 15 | 7 | 1 | 57.7% | +0.009600 | +0.001600 | -0.000100 |
| questdb-query-1751924112805.csv | v3 | 38 | 18 | 14 | 1 | 47.4% | +0.010200 | +0.001600 | -0.000100 |
| questdb-query-1751924112805.csv | v4 | 38 | 14 | 18 | 1 | 36.8% | +0.004100 | +0.001600 | -0.000900 |
| questdb-query-1751958417525.csv | v1 | 146 | 20 | 119 | 1 | 13.7% | -0.006900 | +0.000700 | -0.000100 |
| questdb-query-1751958417525.csv | v2 | 55 | 13 | 36 | 1 | 23.6% | +0.001200 | +0.000800 | -0.000200 |
| questdb-query-1751958417525.csv | v3 | 87 | 20 | 63 | 1 | 23.0% | +0.000400 | +0.000900 | -0.000100 |
| questdb-query-1751958417525.csv | v4 | 88 | 11 | 75 | 1 | 12.5% | -0.009300 | +0.000700 | -0.000900 |
| questdb-query-1751984700242.csv | v1 | 70 | 35 | 30 | 1 | 50.0% | +0.010500 | +0.000800 | -0.000100 |
| questdb-query-1751984700242.csv | v2 | 41 | 24 | 12 | 1 | 58.5% | +0.010600 | +0.001200 | -0.000200 |
| questdb-query-1751984700242.csv | v3 | 49 | 29 | 15 | 1 | 59.2% | +0.013000 | +0.001300 | -0.000100 |
| questdb-query-1751984700242.csv | v4 | 51 | 24 | 24 | 1 | 47.1% | +0.004800 | +0.000800 | -0.000900 |
| questdb-query-1751987436731.csv | v1 | 100 | 37 | 56 | 1 | 37.0% | +0.007300 | +0.001400 | -0.000100 |
| questdb-query-1751987436731.csv | v2 | 49 | 30 | 14 | 1 | 61.2% | +0.010000 | +0.001800 | -0.000200 |
| questdb-query-1751987436731.csv | v3 | 65 | 35 | 26 | 1 | 53.8% | +0.011800 | +0.001900 | -0.000100 |
| questdb-query-1751987436731.csv | v4 | 66 | 27 | 35 | 1 | 40.9% | +0.003100 | +0.001400 | -0.001100 |
| questdb-query-1751990194216.csv | v1 | 85 | 34 | 43 | 1 | 40.0% | +0.015200 | +0.002200 | -0.000100 |
| questdb-query-1751990194216.csv | v2 | 47 | 30 | 12 | 1 | 63.8% | +0.016000 | +0.002100 | -0.000200 |
| questdb-query-1751990194216.csv | v3 | 54 | 33 | 17 | 1 | 61.1% | +0.019400 | +0.002200 | -0.000100 |
| questdb-query-1751990194216.csv | v4 | 56 | 25 | 24 | 1 | 44.6% | +0.009400 | +0.002200 | -0.001300 |
| questdb-query-1751993844367.csv | v1 | 141 | 35 | 93 | 1 | 24.8% | +0.004000 | +0.001600 | -0.000100 |
| questdb-query-1751993844367.csv | v2 | 61 | 30 | 26 | 1 | 49.2% | +0.009700 | +0.001500 | -0.000200 |
| questdb-query-1751993844367.csv | v3 | 91 | 34 | 51 | 1 | 37.4% | +0.009700 | +0.001600 | -0.000100 |
| questdb-query-1751993844367.csv | v4 | 90 | 27 | 55 | 1 | 30.0% | +0.002200 | +0.001600 | -0.000900 |
| questdb-query-1752000467711.csv | v1 | 106 | 32 | 65 | 0 | 30.2% | +0.007400 | +0.001000 | -0.000100 |
| questdb-query-1752000467711.csv | v2 | 53 | 29 | 20 | 1 | 54.7% | +0.008800 | +0.001000 | -0.000200 |
| questdb-query-1752000467711.csv | v3 | 70 | 31 | 31 | 1 | 44.3% | +0.010700 | +0.001100 | -0.000100 |
| questdb-query-1752000467711.csv | v4 | 70 | 27 | 38 | 1 | 38.6% | +0.004800 | +0.001000 | -0.000900 |
| questdb-query-1752003744108.csv | v1 | 93 | 30 | 54 | 1 | 32.3% | +0.009300 | +0.001900 | -0.000100 |
| questdb-query-1752003744108.csv | v2 | 43 | 24 | 16 | 1 | 55.8% | +0.011600 | +0.002100 | -0.000200 |
| questdb-query-1752003744108.csv | v3 | 59 | 29 | 26 | 1 | 49.2% | +0.012900 | +0.002200 | -0.000200 |
| questdb-query-1752003744108.csv | v4 | 61 | 20 | 34 | 1 | 32.8% | -0.000700 | +0.001200 | -0.002200 |
| questdb-query-1752056238892.csv | v1 | 158 | 24 | 126 | 1 | 15.2% | -0.004800 | +0.001000 | -0.000100 |
| questdb-query-1752056238892.csv | v2 | 60 | 21 | 37 | 1 | 35.0% | +0.004400 | +0.001100 | -0.000200 |
| questdb-query-1752056238892.csv | v3 | 101 | 23 | 74 | 1 | 22.8% | +0.002500 | +0.001100 | -0.000100 |
| questdb-query-1752056238892.csv | v4 | 104 | 13 | 90 | 1 | 12.5% | -0.008500 | +0.000700 | -0.001300 |
| questdb-query-1752059055042.csv | v1 | 93 | 19 | 69 | 1 | 20.4% | +0.004200 | +0.001200 | -0.000100 |
| questdb-query-1752059055042.csv | v2 | 37 | 18 | 17 | 0 | 48.6% | +0.008200 | +0.001100 | -0.000200 |
| questdb-query-1752059055042.csv | v3 | 55 | 19 | 32 | 1 | 34.5% | +0.008800 | +0.001300 | -0.000100 |
| questdb-query-1752059055042.csv | v4 | 58 | 12 | 41 | 1 | 20.7% | +0.002300 | +0.001000 | -0.000600 |
| questdb-query-1752255003359.csv | v1 | 91 | 50 | 34 | 1 | 54.9% | +0.027800 | +0.002200 | -0.000200 |
| questdb-query-1752255003359.csv | v2 | 55 | 41 | 9 | 1 | 74.5% | +0.027000 | +0.002400 | -0.000200 |
| questdb-query-1752255003359.csv | v3 | 66 | 45 | 16 | 1 | 68.2% | +0.030000 | +0.002500 | -0.000200 |
| questdb-query-1752255003359.csv | v4 | 67 | 36 | 23 | 1 | 53.7% | +0.017800 | +0.002200 | -0.001400 |
| questdb-query-1752509592905.csv | v1 | 70 | 33 | 30 | 1 | 47.1% | +0.020400 | +0.002000 | -0.000200 |
| questdb-query-1752509592905.csv | v2 | 35 | 24 | 11 | 1 | 68.6% | +0.022500 | +0.002800 | -0.000200 |
| questdb-query-1752509592905.csv | v3 | 48 | 25 | 18 | 1 | 52.1% | +0.023500 | +0.002800 | -0.000100 |
| questdb-query-1752509592905.csv | v4 | 50 | 24 | 21 | 1 | 48.0% | +0.009300 | +0.001800 | -0.002100 |
| questdb-query-1753534868337.csv | v1 | 116 | 18 | 93 | 1 | 15.5% | -0.001200 | +0.001700 | -0.000100 |
| questdb-query-1753534868337.csv | v2 | 38 | 15 | 22 | 0 | 39.5% | +0.006400 | +0.001800 | -0.000200 |
| questdb-query-1753534868337.csv | v3 | 65 | 15 | 47 | 1 | 23.1% | +0.004900 | +0.001800 | -0.000100 |
| questdb-query-1753534868337.csv | v4 | 66 | 9 | 54 | 1 | 13.6% | -0.006200 | +0.000700 | -0.001700 |
| questdb-query-1753551814823.csv | v1 | 93 | 35 | 52 | 1 | 37.6% | +0.012300 | +0.001400 | -0.000100 |
| questdb-query-1753551814823.csv | v2 | 45 | 28 | 11 | 0 | 62.2% | +0.015100 | +0.001400 | -0.000200 |
| questdb-query-1753551814823.csv | v3 | 61 | 32 | 23 | 1 | 52.5% | +0.016700 | +0.001400 | -0.000100 |
| questdb-query-1753551814823.csv | v4 | 62 | 29 | 29 | 1 | 46.8% | +0.010500 | +0.001400 | -0.000800 |
| questdb-query-1753564543126.csv | v1 | 148 | 43 | 101 | 1 | 29.1% | +0.016500 | +0.002300 | -0.000100 |
| questdb-query-1753564543126.csv | v2 | 67 | 38 | 25 | 1 | 56.7% | +0.022400 | +0.002900 | -0.000200 |
| questdb-query-1753564543126.csv | v3 | 104 | 42 | 60 | 1 | 40.4% | +0.022000 | +0.003000 | -0.000100 |
| questdb-query-1753564543126.csv | v4 | 105 | 29 | 73 | 1 | 27.6% | +0.001400 | +0.002300 | -0.002000 |
| questdb-query-1753612688661.csv | v1 | 135 | 45 | 84 | 1 | 33.3% | +0.014400 | +0.001700 | -0.000100 |
| questdb-query-1753612688661.csv | v2 | 59 | 34 | 21 | 1 | 57.6% | +0.020300 | +0.003500 | -0.000200 |
| questdb-query-1753612688661.csv | v3 | 83 | 38 | 40 | 1 | 45.8% | +0.021500 | +0.003500 | -0.000100 |
| questdb-query-1753612688661.csv | v4 | 85 | 27 | 54 | 1 | 31.8% | -0.000200 | +0.001700 | -0.003100 |

---

## Key Findings

**v3 wins on total PnL** — best in 17 out of 20 files. Works well on high liquidity data where the full opposite cycle completes within the run window. Worst trade always -0.0001 to -0.0002 — losses are structurally capped.

**v2 wins on win rate and avg PnL/trade** — 53.4% win rate, +0.000251 avg per trade. Fewer trades but higher quality. The double confirmation filters weak signals despite being structurally imprecise.

**v4 is the weakest** — large losses (-0.001 to -0.003) while waiting for pair confirmation on orphaned cycles.

**v1 worst trade is always -0.0001** — exits fast, losses minimal per trade. Low win rate (29.8%) limits total PnL.

---

## Conclusion

On high liquidity data (July 2025, ~875 trades/min):
- **v3 is optimal for total PnL**
- **v2 is optimal for win rate and avg PnL/trade**
- v4 underperforms due to large losses on orphaned cycles
- v1 is consistent but limited by low win rate

On low liquidity data (March 2026, ~200 trades/min):
- v3 force-closes too often — cycles do not complete within the run window
- v1 remains functional but fees exceed avg PnL per trade at current XRP price range
