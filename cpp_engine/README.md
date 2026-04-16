# SKA C++ Engine

This folder contains the early development of the native C/C++ version of the SKA Engine.

The objective is to progressively move from a QuestDB-centered workflow to a native low-latency engine capable of processing raw tick data directly, computing entropy and structural probabilities in real time, and supporting regime transitions and trading logic without relying on an external database as the core runtime layer.

QuestDB was useful for prototyping, querying, and visualizing the structure of the market. The transition to C/C++ is the next step toward a faster and more controlled engine architecture built for real-time execution.

## Purpose

The purpose of this development folder is to build the native core of the SKA Engine in C/C++ for real-time market applications.

This work focuses on:

* direct processing of raw tick data
* real-time entropy computation
* structural probability computation
* regime classification
* transition detection
* integration with the LONG/SHORT trading logic

## Why move beyond QuestDB

QuestDB has been very useful for:

* fast experimentation
* SQL-based analysis
* transition studies
* probability visualization
* Grafana dashboards

However, for a real-time SKA Engine, the database-centered architecture becomes limiting when the goal is:

* lower latency
* direct stream processing
* fine memory control
* native execution speed
* tighter integration between signal generation and trading logic

The C/C++ implementation is intended to move the SKA Engine from an analysis-oriented environment to an execution-oriented engine.

## Development direction

The native engine is expected to evolve step by step:

1. ingest raw tick data directly
2. compute entropy online
3. compute structural probability in real time
4. classify structural regimes
5. detect regime transitions
6. support trading bot logic directly from the engine output

The goal is not simply to rewrite the old workflow in another language, but to reorganize the SKA Engine around a true real-time native core.

## Scope

This folder is intended for the development of:

* C/C++ entropy routines
* probability and transition modules
* stream-processing utilities
* engine state management
* experimental native trading logic interfaces
* performance-oriented prototypes

## Architecture

```
Binance WebSocket → Entropy (SKA) → dH/H → Regime → 4-bit word → Binary stream → Pattern match → Order API
```

---

## Encoder

Reads the entropy stream and produces a continuous binary stream of 4-bit words.

**Regime from entropy:**

```
dH_H = (H - H_prev) / H

dH_H > 0  →  bull    (1)
dH_H < 0  →  bear    (2)
dH_H = 0  →  neutral (0)
```

**Transition encoding:**

```
transition_code = prev_regime × 3 + regime
4-bit word      = transition_table[transition_code]
```

**Transition table:**

| Code | Transition       | 4-bit word |
|------|-----------------|------------|
| 0    | neutral→neutral | `0000`     |
| 1    | neutral→bull    | `0001`     |
| 2    | neutral→bear    | `0010`     |
| 3    | bull→neutral    | `0100`     |
| 4    | bull→bull       | `0101`     |
| 5    | bull→bear       | `0110`     |
| 6    | bear→neutral    | `1000`     |
| 7    | bear→bull       | `1001`     |
| 8    | bear→bear       | `1010`     |

---

## Sequence

```
S = 0000 a₁ a₂ ... aₖ 0000   =   4(k+2) bits
```

A sequence opens and closes on `0000` (neutral→neutral). The binary code is the concatenation of all 4-bit words. Two sequences are identical if and only if their binary codes are equal.

---

## Pattern Matcher

Compares the current sequence `binary_code` against the false start library.

```cpp
bool is_false_start(uint64_t code) {
    for (auto& pattern : library) {
        if (code == pattern.binary_code) return true;
    }
    return false;
}
```

Each comparison is one integer equality check — O(1) per pattern.

---

## Status

Early development.

The architecture is still under construction, and the current focus is on building the foundations of the native engine before expanding toward a full production pipeline.
