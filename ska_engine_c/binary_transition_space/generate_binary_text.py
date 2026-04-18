"""
generate_binary_text.py

Generates two text files from the binary flow output CSVs:

  binary_flow_integers.txt  — space-separated binary_code_int values (one line)
                              one integer per sequence, all loops concatenated
                              e.g.: 320 640 320 5760 640 ...

  binary_flow_sample.txt    — raw 4-bit word stream (space-separated)
                              header + all words from all stream CSVs
                              e.g.: 0000 0001 0100 0000 0010 1000 ...

Usage:
  python3 generate_binary_text.py [--input DIR] [--output DIR]

Defaults:
  --input  /home/coder/project/Real_Time_SKA_trading/binary_transition_space/output
  --output /home/coder/project/Real_Time_SKA_trading/binary_transition_space
"""

import csv
import glob
import os
import argparse


INPUT_DIR  = "/SKA-quantitative-finance/ska_engine_c/binary_transition_space/output"
OUTPUT_DIR = "/SKA-quantitative-finance/ska_engine_c/binary_transition_space/output"


def generate_integers(input_dir, output_dir):
    """Extract binary_code_int from all _sequences.csv files."""
    files = sorted(glob.glob(os.path.join(input_dir, "*_sequences.csv")))
    integers = []
    for filepath in files:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    integers.append(row["binary_code_int"].strip())
                except KeyError:
                    continue

    out = os.path.join(output_dir, "binary_flow_integers.txt")
    with open(out, "w") as f:
        f.write(" ".join(integers) + "\n")

    print(f"binary_flow_integers.txt — {len(integers):,} integers from {len(files):,} files")
    print(f"Saved: {out}")


def generate_sample(input_dir, output_dir):
    """Extract 4-bit words from all _stream.csv files."""
    files = sorted(glob.glob(os.path.join(input_dir, "*_stream.csv")))
    words = []
    for filepath in files:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    words.append(row["word"].strip())
                except KeyError:
                    continue

    out = os.path.join(output_dir, "binary_flow_sample.txt")
    with open(out, "w") as f:
        f.write(f"Binary information flow — {len(files)} loops\n")
        f.write(f"{len(words):,} words\n\n")
        f.write(" ".join(words) + "\n")

    print(f"binary_flow_sample.txt  — {len(words):,} words from {len(files):,} files")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate binary flow text files")
    parser.add_argument("--input",  default=INPUT_DIR,  help="Directory containing output CSVs")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    generate_integers(args.input, args.output)
    generate_sample(args.input, args.output)


if __name__ == "__main__":
    main()
