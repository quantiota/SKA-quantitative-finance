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

## Bot v1 — ΔP band regime, entropy-derived probability

```
Regime definition:
  P(n)   = exp(-|ΔH/H|)         where ΔH/H = (H(n) - H(n-1)) / H(n)
  ΔP(n)  = P(n) - P(n-1)

  |ΔP − (−0.86)| ≤ 0.0042  →  regime = 2  (bear)
  |ΔP − (−0.34)| ≤ 0.0198  →  regime = 1  (bull)
  else                       →  regime = 0  (neutral)

P band positions — universal constants at convergence scale:
  P_NEUTRAL_NEUTRAL = 1.00
  P_NEUTRAL_BULL    = 0.66
  P_X_NEUTRAL       = 0.51   (bull→neutral = bear→neutral)
  P_NEUTRAL_BEAR    = 0.14

Exit filter: abs(P - 0.51) ≤ 0.0153  (TOL_CLOSE)
```

```
LONG:   neutral→bull              (OPEN — WAIT_PAIR)
        bull→neutral              (pair confirmed — IN_NEUTRAL)
        neutral→neutral × N≥3    (neutral gap — READY)
        neutral→bear              (opposite cycle opens — EXIT_WAIT)
        bear→neutral + TOL_CLOSE  (CLOSE LONG)

SHORT:  neutral→bear              (OPEN — WAIT_PAIR)
        bear→neutral              (pair confirmed — IN_NEUTRAL)
        neutral→neutral × N≥3    (neutral gap — READY)
        neutral→bull              (opposite cycle opens — EXIT_WAIT)
        bull→neutral + TOL_CLOSE  (CLOSE SHORT)
```

State machine: WAIT_PAIR → IN_NEUTRAL → READY → EXIT_WAIT → CLOSE.

Additional guards:
- `MIN_TRADES = 60` — no trade until 60 entropy-valid ticks (SKA convergence warmup)
- Direct jump filter — `bull→bear` and `bear→bull` ignored (localized entropy shocks)

---

## Signal Logic — Diagram

```mermaid
flowchart TD
    BINANCE[(Binance Tick Data)]
    ENGINE["SKA ENGINE"]
    BOT@{ shape: diamond, label: "Trading Bot" }

    BINANCE -- "symbol" --> ENGINE
    ENGINE -- "entropy" --> BOT

    BOT --> LONG
    BOT --> SHORT

    subgraph LONG["LONG"]
        direction TB
        L1["neutral→bull<br/><i>OPEN / WAIT_PAIR</i>"]
        L2["bull→neutral<br/><i>pair confirmed / IN_NEUTRAL</i>"]
        L3["neutral→neutral × N (N≥3)<br/><i>neutral gap / READY</i>"]
        L4["neutral→bear<br/><i>opp. cycle opens / EXIT_WAIT</i>"]
        L5["bear→neutral<br/><i>opp. pair confirmed / CLOSE LONG</i>"]
        L1 --> L2 --> L3 --> L4 --> L5
        L3 -. "↺ repeats" .-> L1
    end

    subgraph SHORT["SHORT"]
        direction TB
        S1["neutral→bear<br/><i>OPEN / WAIT_PAIR</i>"]
        S2["bear→neutral<br/><i>pair confirmed / IN_NEUTRAL</i>"]
        S3["neutral→neutral × N (N≥3)<br/><i>neutral gap / READY</i>"]
        S4["neutral→bull<br/><i>opp. cycle opens / EXIT_WAIT</i>"]
        S5["bull→neutral<br/><i>opp. pair confirmed / CLOSE SHORT</i>"]
        S1 --> S2 --> S3 --> S4 --> S5
        S3 -. "↺ repeats" .-> S1
    end

    classDef data      fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px;
    classDef process   fill:#E8F5E9,stroke:#43A047,stroke-width:2px;
    classDef longOpen  fill:#A8DFBC,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef longPair  fill:#C8F0A8,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef shortOpen fill:#FFAAAA,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef shortPair fill:#FFD0A0,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef neutral   fill:#E8E8E8,stroke:#AAAAAA,color:#000,stroke-width:1.5px;

    class BINANCE data;
    class API,BOT process;
    class L1 longOpen;
    class L2 longPair;
    class L3 neutral;
    class L4 shortOpen;
    class L5 shortPair;
    class S1 shortOpen;
    class S2 shortPair;
    class S3 neutral;
    class S4 longOpen;
    class S5 longPair;

```

---

## Data

- Source: Binance XRPUSDT WebSocket — real tick data exported from QuestDB
- Folder: `XRPUSDT/` — 112 files, March 28–29 2026
- Liquidity: ~700 trades/5 min (low liquidity period)
- Each file: ~3500 trades per loop
- Entropy computed by the SKA learning engine.

---

## Backtest Results

## April 6, 2026 — 133 loops, XRPUSDT, bot v4 

| Metric | Value |
|---|---|
| Loops | 133 |
| Total trades | 3,180 |
| Winners | 1,517 |
| Losers | 1,097 |
| Flat | 566 |
| Win rate | 47.7% |
| Total PnL | **+4,491 pips** |
| Avg / trade | **+1.41 pips** |
| LONG (spot) | +2,319 pips |
| SHORT (synth) | +2,172 pips |
| Available pips | 40,981 pips |
| Capture rate | **+10.96%** |
| Force closes | 133 |


---

## Usage

```bash
# Run backtest on all files in XRPUSDT/
/opt/venv/bin/python3 backtest_v4.py

```
