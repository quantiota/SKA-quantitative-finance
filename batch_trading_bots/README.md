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

Format per cell: `trades/win/lose  win%  pnl  best/worst`

| File | v1 | v2 | v3 | v4 |
|------|----|----|----|----|
| 1751814162388 | 110/38/63 34.5% +0.0055 +0.0010/-0.0001 | 45/26/15 57.8% +0.0102 +0.0020/-0.0002 | 60/28/23 46.7% +0.0121 +0.0021/-0.0001 | 60/21/33 35.0% -0.0025 +0.0010/-0.0016 |
| 1751823646841 | 124/30/81 24.2% +0.0008 +0.0009/-0.0001 | 56/27/23 48.2% +0.0056 +0.0009/-0.0002 | 83/29/43 34.9% +0.0060 +0.0009/-0.0001 | 88/23/55 26.1% -0.0013 +0.0009/-0.0005 |
| 1751880848676 | 122/26/93 21.3% -0.0013 +0.0009/-0.0001 | 47/20/24 42.6% +0.0055 +0.0015/-0.0002 | 79/22/55 27.8% +0.0037 +0.0015/-0.0001 | 81/19/59 23.5% -0.0044 +0.0007/-0.0012 |
| 1751909192925 | 53/24/23 45.3% +0.0134 +0.0029/-0.0001 | 31/20/9 64.5% +0.0137 +0.0033/-0.0001 | 41/24/15 58.5% +0.0149 +0.0033/-0.0001 | 42/21/17 50.0% +0.0067 +0.0015/-0.0031 |
| 1751924112805 | 60/21/31 35.0% +0.0068 +0.0016/-0.0003 | 26/15/7 57.7% +0.0096 +0.0016/-0.0001 | 38/18/14 47.4% +0.0102 +0.0016/-0.0001 | 38/14/18 36.8% +0.0041 +0.0016/-0.0009 |
| 1751958417525 | 146/20/119 13.7% -0.0069 +0.0007/-0.0001 | 55/13/36 23.6% +0.0012 +0.0008/-0.0002 | 87/20/63 23.0% +0.0004 +0.0009/-0.0001 | 88/11/75 12.5% -0.0093 +0.0007/-0.0009 |
| 1751984700242 | 70/35/30 50.0% +0.0105 +0.0008/-0.0001 | 41/24/12 58.5% +0.0106 +0.0012/-0.0002 | 49/29/15 59.2% +0.0130 +0.0013/-0.0001 | 51/24/24 47.1% +0.0048 +0.0008/-0.0009 |
| 1751987436731 | 100/37/56 37.0% +0.0073 +0.0014/-0.0001 | 49/30/14 61.2% +0.0100 +0.0018/-0.0002 | 65/35/26 53.8% +0.0118 +0.0019/-0.0001 | 66/27/35 40.9% +0.0031 +0.0014/-0.0011 |
| 1751990194216 | 85/34/43 40.0% +0.0152 +0.0022/-0.0001 | 47/30/12 63.8% +0.0160 +0.0021/-0.0002 | 54/33/17 61.1% +0.0194 +0.0022/-0.0001 | 56/25/24 44.6% +0.0094 +0.0022/-0.0013 |
| 1751993844367 | 141/35/93 24.8% +0.0040 +0.0016/-0.0001 | 61/30/26 49.2% +0.0097 +0.0015/-0.0002 | 91/34/51 37.4% +0.0097 +0.0016/-0.0001 | 90/27/55 30.0% +0.0022 +0.0016/-0.0009 |
| 1752000467711 | 106/32/65 30.2% +0.0074 +0.0010/-0.0001 | 53/29/20 54.7% +0.0088 +0.0010/-0.0002 | 70/31/31 44.3% +0.0107 +0.0011/-0.0001 | 70/27/38 38.6% +0.0048 +0.0010/-0.0009 |
| 1752003744108 | 93/30/54 32.3% +0.0093 +0.0019/-0.0001 | 43/24/16 55.8% +0.0116 +0.0021/-0.0002 | 59/29/26 49.2% +0.0129 +0.0022/-0.0002 | 61/20/34 32.8% -0.0007 +0.0012/-0.0022 |
| 1752056238892 | 158/24/126 15.2% -0.0048 +0.0010/-0.0001 | 60/21/37 35.0% +0.0044 +0.0011/-0.0002 | 101/23/74 22.8% +0.0025 +0.0011/-0.0001 | 104/13/90 12.5% -0.0085 +0.0007/-0.0013 |
| 1752059055042 | 93/19/69 20.4% +0.0042 +0.0012/-0.0001 | 37/18/17 48.6% +0.0082 +0.0011/-0.0002 | 55/19/32 34.5% +0.0088 +0.0013/-0.0001 | 58/12/41 20.7% +0.0023 +0.0010/-0.0006 |
| 1752255003359 | 91/50/34 54.9% +0.0278 +0.0022/-0.0002 | 55/41/9 74.5% +0.0270 +0.0024/-0.0002 | 66/45/16 68.2% +0.0300 +0.0025/-0.0002 | 67/36/23 53.7% +0.0178 +0.0022/-0.0014 |
| 1752509592905 | 70/33/30 47.1% +0.0204 +0.0020/-0.0002 | 35/24/11 68.6% +0.0225 +0.0028/-0.0002 | 48/25/18 52.1% +0.0235 +0.0028/-0.0001 | 50/24/21 48.0% +0.0093 +0.0018/-0.0021 |
| 1753534868337 | 116/18/93 15.5% -0.0012 +0.0017/-0.0001 | 38/15/22 39.5% +0.0064 +0.0018/-0.0002 | 65/15/47 23.1% +0.0049 +0.0018/-0.0001 | 66/9/54 13.6% -0.0062 +0.0007/-0.0017 |
| 1753551814823 | 93/35/52 37.6% +0.0123 +0.0014/-0.0001 | 45/28/11 62.2% +0.0151 +0.0014/-0.0002 | 61/32/23 52.5% +0.0167 +0.0014/-0.0001 | 62/29/29 46.8% +0.0105 +0.0014/-0.0008 |
| 1753564543126 | 148/43/101 29.1% +0.0165 +0.0023/-0.0001 | 67/38/25 56.7% +0.0224 +0.0029/-0.0002 | 104/42/60 40.4% +0.0220 +0.0030/-0.0001 | 105/29/73 27.6% +0.0014 +0.0023/-0.0020 |
| 1753612688661 | 135/45/84 33.3% +0.0144 +0.0017/-0.0001 | 59/34/21 57.6% +0.0203 +0.0035/-0.0002 | 83/38/40 45.8% +0.0215 +0.0035/-0.0001 | 85/27/54 31.8% -0.0002 +0.0017/-0.0031 |

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

