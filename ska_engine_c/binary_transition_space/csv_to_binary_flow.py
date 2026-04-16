"""
csv_to_binary_flow.py

Converts QuestDB CSV exports to binary information flow.

Pipeline:
  entropy history → dH/H → regime → transition_code → 4-bit word → binary stream → sequences

Output per CSV file:
  - binary stream:  one 4-bit word per tick (as text: 0001, 0100, ...)
  - sequences:      one sequence per line with binary_code (uint64) and label

Usage:
  python3 csv_to_binary_flow.py [--input DIR] [--output DIR] [--csv FILE]
"""

import csv
import glob
import os
import argparse

# ── Transition table ──────────────────────────────────────────────────────────
# transition_code = prev_regime * 3 + regime
# neutral=0, bull=1, bear=2
TRANSITION_TABLE = {
    0: "0000",  # neutral→neutral
    1: "0001",  # neutral→bull
    2: "0010",  # neutral→bear
    3: "0100",  # bull→neutral
    4: "0101",  # bull→bull
    5: "0110",  # bull→bear
    6: "1000",  # bear→neutral
    7: "1001",  # bear→bull
    8: "1010",  # bear→bear
}

TRANSITION_NAMES = {
    0: "neutral→neutral",
    1: "neutral→bull",
    2: "neutral→bear",
    3: "bull→neutral",
    4: "bull→bull",
    5: "bull→bear",
    6: "bear→neutral",
    7: "bear→bull",
    8: "bear→bear",
}


def classify_regime(price, prev_price):
    if price > prev_price:
        return 1  # bull
    elif price < prev_price:
        return 2  # bear
    else:
        return 0  # neutral (price unchanged)


def pack_binary_code(words):
    """Pack list of 4-bit word strings into a single integer."""
    code = 0
    for w in words:
        code = (code << 4) | int(w, 2)
    return code


def format_binary_code(words):
    """Return space-separated 4-bit words as string."""
    return " ".join(words)


def process_csv(filepath):
    """
    Read one CSV file and return:
      - stream: list of (trade_id, timestamp, regime, transition_code, word)
      - sequences: list of dicts with sequence metadata
    """
    ticks = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if "entropy" not in (reader.fieldnames or []):
            return [], []
        for row in reader:
            if not row["entropy"]:
                continue
            ticks.append({
                "trade_id":  row["trade_id"],
                "timestamp": row["timestamp"],
                "entropy":   float(row["entropy"]),
                "price":     float(row["price"]),
            })

    if len(ticks) < 2:
        return [], []

    stream = []
    prev_entropy = None
    prev_regime  = None
    prev_price   = None

    for tick in ticks:
        H = tick["entropy"]

        if prev_entropy is None:
            prev_entropy = H
            prev_price   = tick["price"]
            continue

        dh_h   = (H - prev_entropy) / H if H != 0 else 0
        regime = classify_regime(tick["price"], prev_price)

        if prev_regime is not None:
            code = prev_regime * 3 + regime
            word = TRANSITION_TABLE[code]
            stream.append({
                "trade_id":        tick["trade_id"],
                "timestamp":       tick["timestamp"],
                "regime":          regime,
                "transition_code": code,
                "word":            word,
                "dh_h":            dh_h,
            })

        prev_regime  = regime
        prev_entropy = H
        prev_price   = tick["price"]

    # ── Sequence detection ────────────────────────────────────────────────────
    # A sequence opens when we exit 0000 and closes on the next 0000.
    sequences = []
    in_sequence   = False
    seq_words     = []
    seq_start_idx = None

    for i, entry in enumerate(stream):
        word = entry["word"]

        if not in_sequence:
            if word == "0000":
                # potential open boundary — start accumulating
                in_sequence   = True
                seq_words     = ["0000"]
                seq_start_idx = i
        else:
            if word == "0000":
                # close boundary — record only if at least 1 inner transition
                if len(seq_words) > 1:
                    seq_words.append(word)
                    binary_code_int = pack_binary_code(seq_words)
                    sequences.append({
                        "trade_id_open":  stream[seq_start_idx]["trade_id"],
                        "trade_id_close": entry["trade_id"],
                        "timestamp_open":  stream[seq_start_idx]["timestamp"],
                        "timestamp_close": entry["timestamp"],
                        "inner_length":    len(seq_words) - 2,
                        "total_length":    len(seq_words),
                        "words":           seq_words[:],
                        "binary_code_str": format_binary_code(seq_words),
                        "binary_code_int": binary_code_int,
                    })
                # restart: this 0000 is the open of the next sequence
                seq_words     = ["0000"]
                seq_start_idx = i
            else:
                seq_words.append(word)

    return stream, sequences


def write_stream(stream, outpath):
    with open(outpath, "w") as f:
        f.write("trade_id,timestamp,regime,transition_code,transition_name,word,dh_h\n")
        for entry in stream:
            f.write(
                f"{entry['trade_id']},"
                f"{entry['timestamp']},"
                f"{entry['regime']},"
                f"{entry['transition_code']},"
                f"{TRANSITION_NAMES[entry['transition_code']]},"
                f"{entry['word']},"
                f"{entry['dh_h']:.8f}\n"
            )


def write_sequences(sequences, outpath):
    with open(outpath, "w") as f:
        f.write("trade_id_open,trade_id_close,timestamp_open,timestamp_close,"
                "inner_length,total_length,binary_code_str,binary_code_int\n")
        for s in sequences:
            f.write(
                f"{s['trade_id_open']},"
                f"{s['trade_id_close']},"
                f"{s['timestamp_open']},"
                f"{s['timestamp_close']},"
                f"{s['inner_length']},"
                f"{s['total_length']},"
                f"{s['binary_code_str']},"
                f"{s['binary_code_int']}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Convert QuestDB CSV to binary information flow")
    parser.add_argument("--input",  default="/home/coder/project/Real_Time_SKA_trading/questdb_export",
                        help="Directory containing CSV files")
    parser.add_argument("--output", default="/home/coder/project/Real_Time_SKA_trading/binary_transition_space/output",
                        help="Output directory")
    parser.add_argument("--csv",    default=None,
                        help="Process a single CSV file instead of a directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.csv:
        files = [args.csv]
    else:
        files = sorted(glob.glob(os.path.join(args.input, "*.csv")))

    total_sequences = 0

    for filepath in files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        stream, sequences = process_csv(filepath)

        if not stream:
            continue

        stream_out    = os.path.join(args.output, f"{basename}_stream.csv")
        sequence_out  = os.path.join(args.output, f"{basename}_sequences.csv")

        write_stream(stream, stream_out)
        write_sequences(sequences, sequence_out)

        total_sequences += len(sequences)
        print(f"{basename}: {len(stream)} ticks → {len(sequences)} sequences")

    print(f"\nDone. Total sequences: {total_sequences}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
