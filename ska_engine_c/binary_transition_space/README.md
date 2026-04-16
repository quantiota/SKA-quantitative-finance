# Binary Transition Space — Analysis

Converts QuestDB CSV exports to binary information flow and plots the sequence length distribution.

---

## Pipeline

```
questdb_export/*.csv → encoder → sequence detector → binary_code (uint64) → distribution plot
```

---

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

---

## Result

![binary_flow_all_log](binary_flow_all_log.png)

**1,571 CSV files — 354,043 sequences**

- 4-word sequences: 39.6%
- 5-word sequences: 42.7%
- Together: **82.3%** of all sequences
- Hard cutoff at 33 words

The distribution is **not random**. The market selects specific sequence lengths with high probability. Short sequences (4–5 words) are the natural structural unit of market information. The binary information flow is compressible — a random source is not.
