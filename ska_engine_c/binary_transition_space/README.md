# Binary Transition Space — Analysis

Converts QuestDB CSV exports to binary information flow and plots the sequence length distribution.

---

## Pipeline

```
questdb_export/*.csv → encoder → sequence detector → binary_code (uint64) → distribution plot
```



## Scripts

### `csv_to_binary_flow.py`

Reads tick data CSVs and produces:
- `*_stream.csv` — one row per tick: `trade_id, timestamp, regime, transition_code, transition_name, word, dh_h`
- `*_sequences.csv` — one row per sequence: `trade_id_open, trade_id_close, inner_length, total_length, binary_code_str, binary_code_int`

```bash
# single file
python3 csv_to_binary_flow.py --csv /path/to/file.csv

# full directory
python3 csv_to_binary_flow.py --input /path/to/questdb_export --output /path/to/output
```

### `plot_binary_flow.py`

Plots sequence length distribution: bar chart (log scale) + pie chart.

```bash
# single sequence file
python3 plot_binary_flow.py --csv /path/to/sequences.csv

# all sequence files in output directory
python3 plot_binary_flow.py
```



## Result

![binary_flow_all_log](binary_flow_all_log.png)

**1,571 CSV files — 356,528 sequences**

- 4-word sequences: 77.7%
- 5-word sequences: 8.5%
- Together: **86.2%** of all sequences
- Hard cutoff at 33 words

The distribution is **not random**. The market selects specific sequence lengths with high probability. Short sequences (4–5 words) are the natural structural unit of market information.




## Observed sequences — 1,571 CSV files

### 4-word sequences — 2 distinct

```
count: 139,507  integer: 320   delta_pips: +1
0000(neutral-neutral)  0001(neutral-bull)  0100(bull-neutral)  0000(neutral-neutral)

count: 137,537  integer: 640   delta_pips: -1
0000(neutral-neutral)  0010(neutral-bear)  1000(bear-neutral)  0000(neutral-neutral)
```

### 5-word sequences — 4 distinct

```
count: 15,812  integer: 10560  delta_pips:  0
0000(neutral-neutral)  0010(neutral-bear)  1001(bear-bull)  0100(bull-neutral)  0000(neutral-neutral)

count: 14,468  integer: 5760   delta_pips:  0
0000(neutral-neutral)  0001(neutral-bull)  0110(bull-bear)  1000(bear-neutral)  0000(neutral-neutral)

count: 106     integer: 5440   delta_pips: +2
0000(neutral-neutral)  0001(neutral-bull)  0101(bull-bull)  0100(bull-neutral)  0000(neutral-neutral)

count: 105     integer: 10880  delta_pips: -2
0000(neutral-neutral)  0010(neutral-bear)  1010(bear-bear)  1000(bear-neutral)  0000(neutral-neutral)
```

## Structural role of the 5-word reversal sequences

The two dominant 5-word sequences (integers 5760 and 10560, delta_pips=0) represent **8.5% of all sequences** across 1,571 loops:

```
neutral-bull → bull-bear → bear-neutral   (tried up, rejected, returned)
neutral-bear → bear-bull → bull-neutral   (tried down, rejected, returned)
```

These are not noise. They are the market's **directional probe** — a test with no commitment.

The market asks a binary question: "is there sustained demand in this direction?" The cross-transition is the answer: **no**. The sequence closes at delta_pips=0 — no structural direction was established, but the question had to be asked.

Without these sequences the market would only produce clean LONG and SHORT pairs. That is impossible — to find the price where supply meets demand, the market must probe both sides and retract when rejected.

| sequence type | delta_pips | role |
|---|---|---|
| 4-word (77.7%) | ±1 | commitment — direction established |
| 5-word reversal (8.5%) | 0 | rejection — direction tested and refused |

From Wheeler's "It from Bit": the reversal sequence encodes one bit — "this direction has no structural support at this moment." The sequence must exist for the market to learn that answer. They are the market's self-correction mechanism between trends.


## The integer as a structural key

Each sequence is uniquely identified by a single `uint64` integer — `binary_code_int`. This has two direct applications.

### Memory operations

Pattern matching reduces to one integer comparison:

```cpp
bool is_known(uint64_t code) {
    return library.count(code) > 0;  // O(1)
}
```

- The full sequence library (1,381 entries) fits in L1 cache as a sorted array — ~10 comparisons to locate any sequence
- No string parsing, no word-by-word iteration — one equality check replaces the entire sequence comparison

Two market events are structurally identical **if and only if their integers are equal**.

### Historical archiving

One loop produces ~3,500 ticks and ~273 sequences. At the integer level:

```
3,500 ticks  →  273 uint64 integers
```

The integer stream is lossless at the structural level: the full transition path can be reconstructed from the integer by reversing the 4-bit packing. A year of tick data becomes a compact stream of integers — queryable by pattern, not by price.
