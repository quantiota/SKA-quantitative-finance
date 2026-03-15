# SKA Batch Backtest — Paired Cycle Trading (PCT)

## Framework

**Entropic Trading** — uses entropy dynamics as the signal axis instead of price.
The signal is derived from the market's own learning process (SKA — Structured Knowledge Accumulation), not from price levels or volume.

**Paired Cycle Trading (PCT)** — entry and exit defined by paired regime transitions in the TradeID Series.
The bot is structurally blind to the neutral→neutral baseline (90% of trades) by design.
Only the 4 directional transitions carry signal:

```
neutral→bull   bull→neutral   neutral→bear   bear→neutral
```

This is not HFT. It is event-driven structural trading operating at tick data resolution.

---

## Bot v1 — Consecutive same-direction paired cycles

```
LONG:   neutral→bull              (OPEN — WAIT_PAIR)
        bull→neutral              (pair confirmed — IN_NEUTRAL)
        neutral→neutral × N       (count all — stay IN_NEUTRAL)
        <first non-neutral>       (gap closes — READY)
        neutral→bull              (cycle repeats — back to WAIT_PAIR)
        ...
        neutral→bear OR bear→neutral  (CLOSE — only from READY state)

SHORT:  neutral→bear              (OPEN — WAIT_PAIR)
        bear→neutral              (pair confirmed — IN_NEUTRAL)
        neutral→neutral × N       (count all — stay IN_NEUTRAL)
        <first non-neutral>       (gap closes — READY)
        neutral→bear              (cycle repeats — back to WAIT_PAIR)
        ...
        neutral→bull OR bull→neutral  (CLOSE — only from READY state)
```

State machine: WAIT_PAIR → IN_NEUTRAL → READY → (WAIT_PAIR loop or CLOSE).

The alpha: the market generates consecutive same-direction paired cycles.
Hold through all of them — close only when the first opposite-direction cycle opens.
The neutral gap (neutral→neutral × N) is counted per cycle as `nn_count`.

---

## Data

- Source: Binance XRPUSDT WebSocket — real tick data exported from QuestDB
- Folder: `XRPUSDT/` — 20 files, July 2025
- Liquidity: ~875 trades/minute (high liquidity period)
- Each file: ~2300–4500 trades, 2–20 minutes of market activity
- Entropy computed by the SKA learning engine (matrix grows 1×1 → N×N per loop)

---

## Backtest Results — 20 files, July 2025

| Trades | Win%  | Total PnL | Avg PnL/trade | Force closes |
|--------|-------|-----------|---------------|--------------|
| 1008   | 41.9% | +0.1223   | +0.000121     | 18           |

### Per file

| File | Trades | Win% | PnL | Force |
|------|--------|------|-----|-------|
| questdb-query-1751814162388.csv | 42 | 45.2% | +0.005300 | 1 |
| questdb-query-1751823646841.csv | 62 | 29.0% | +0.002300 | 1 |
| questdb-query-1751880848676.csv | 53 | 35.8% | +0.001600 | 1 |
| questdb-query-1751909192925.csv | 34 | 50.0% | +0.006300 | 1 |
| questdb-query-1751924112805.csv | 31 | 32.3% | +0.002600 | 1 |
| questdb-query-1751958417525.csv | 58 | 22.4% | -0.003700 | 1 |
| questdb-query-1751984700242.csv | 39 | 51.3% | +0.005600 | 1 |
| questdb-query-1751987436731.csv | 51 | 52.9% | +0.004000 | 1 |
| questdb-query-1751990194216.csv | 46 | 54.3% | +0.012500 | 1 |
| questdb-query-1751993844367.csv | 67 | 34.3% | +0.004100 | 1 |
| questdb-query-1752000467711.csv | 52 | 51.9% | +0.007800 | 1 |
| questdb-query-1752003744108.csv | 47 | 46.8% | +0.002400 | 1 |
| questdb-query-1752056238892.csv | 66 | 21.2% | -0.003400 | 1 |
| questdb-query-1752059055042.csv | 42 | 33.3% | +0.004200 | 0 |
| questdb-query-1752255003359.csv | 56 | 67.9% | +0.021600 | 1 |
| questdb-query-1752509592905.csv | 38 | 63.2% | +0.022300 | 1 |
| questdb-query-1753534868337.csv | 45 | 22.2% | -0.001900 | 0 |
| questdb-query-1753551814823.csv | 44 | 61.4% | +0.011700 | 1 |
| questdb-query-1753564543126.csv | 71 | 39.4% | +0.011000 | 1 |
| questdb-query-1753612688661.csv | 64 | 42.2% | +0.006000 | 1 |

Live results: in progress.
