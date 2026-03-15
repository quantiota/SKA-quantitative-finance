# SKA Trading Bot Results


## SKA Real-Time Dashboard — XRPUSDT Live Market Structure Analysis (Youtube)

[![Watch the demo](thumbnail.png)](https://youtu.be/01qdoMPAlB4?si=5xDByNTuGZF4gare/video)

## Framework

**Entropic Trading** — uses entropy dynamics as the signal axis instead of price.
The source of the alpha is the market's own learning process, not price levels or volume.

**Paired Cycle Trading (PCT)** — the specific strategy implemented here.
Entry and exit are defined by complete paired regime cycles in the TradeID Series.
The bot is structurally blind to the neutral→neutral baseline by design — it trades only
the 10% of transitions that carry directional information.

This is not HFT. It is event-driven structural trading — the signal fires on a topological
event (completion of a paired regime cycle), not on a threshold or a price level.

## Signal Logic — Diagram

```mermaid
flowchart TB
    title["SKA Paired Cycle Trading — v1 Signal Logic"]

    note["v1 — Consecutive same-direction paired cycles<br/>Hold through repeated same-direction cycles — close only when opposite cycle opens"]

    title --> note

    subgraph LONG["LONG"]
        direction LR
        L1["neutral→bull<br/><i>OPEN / WAIT_PAIR</i>"]
        L2["bull→neutral<br/><i>pair confirmed / IN_NEUTRAL</i>"]
        L3["neutral→neutral × N<br/><i>neutral gap / READY</i>"]
        L4["neutral→bear<br/>or bear→neutral<br/><i>CLOSE LONG</i>"]

        L1 --> L2 --> L3 --> L4
        L3 -. "↺ repeats" .-> L1
    end

    subgraph SHORT["SHORT"]
        direction LR
        S1["neutral→bear<br/><i>OPEN / WAIT_PAIR</i>"]
        S2["bear→neutral<br/><i>pair confirmed / IN_NEUTRAL</i>"]
        S3["neutral→neutral × N<br/><i>neutral gap / READY</i>"]
        S4["neutral→bull<br/>or bull→neutral<br/><i>CLOSE SHORT</i>"]

        S1 --> S2 --> S3 --> S4
        S3 -.-> S1
    end

    note --> LONG
    note --> SHORT

    classDef longOpen fill:#A8DFBC,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef longPair fill:#C8F0A8,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef shortOpen fill:#FFAAAA,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef shortPair fill:#FFD0A0,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef neutral fill:#E8E8E8,stroke:#AAAAAA,color:#000,stroke-width:1.5px;
    classDef meta fill:#FFFFFF,stroke:#FFFFFF,color:#222;

    class title,note meta;
    class L1 longOpen;
    class L2 longPair;
    class L3 neutral;
    class L4 shortOpen;
    class S1 shortOpen;
    class S2 shortPair;
    class S3 neutral;
    class S4 longOpen;
```


## Bot Version

### v1 — Consecutive same-direction paired cycles (current)

```
LONG:   neutral→bull              (OPEN — WAIT_PAIR)
        bull→neutral              (UP pair confirmed — IN_NEUTRAL)
        neutral→neutral × N       (neutral gap, count all — IN_NEUTRAL)
        <first non-neutral>       (gap closes — READY)
        neutral→bull              (cycle repeats — back to WAIT_PAIR)
        ...
        neutral→bear OR bear→neutral  (CLOSE — only from READY state)

SHORT:  neutral→bear              (OPEN — WAIT_PAIR)
        bear→neutral              (DOWN pair confirmed — IN_NEUTRAL)
        neutral→neutral × N       (neutral gap, count all — IN_NEUTRAL)
        <first non-neutral>       (gap closes — READY)
        neutral→bear              (cycle repeats — back to WAIT_PAIR)
        ...
        neutral→bull OR bull→neutral  (CLOSE — only from READY state)
```

State machine: WAIT_PAIR → IN_NEUTRAL → READY → (WAIT_PAIR loop or CLOSE).

The alpha: the market generates consecutive same-direction paired cycles. Hold through
all of them — close only when the first opposite-direction cycle opens.
The neutral gap (neutral→neutral × N) is counted per cycle and logged as `neutral_neutral_count`.


Live results: in progress.