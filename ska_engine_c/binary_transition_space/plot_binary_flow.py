"""
plot_binary_flow.py

Plot binary_code_int (log scale) from sequence CSV files.

Usage:
  python3 plot_binary_flow.py [--input DIR] [--csv FILE]
"""

import csv
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = "/home/coder/project/Real_Time_SKA_trading/binary_transition_space/output"


def load_sequences(path):
    codes = []
    lengths = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = int(row["binary_code_int"])
            if val > 0:  # skip pure neutral sequences (0)
                codes.append(val)
                lengths.append(int(row["total_length"]))
    return codes, lengths


def plot(codes, lengths, title, outpath):
    from collections import Counter
    counts = Counter(lengths)
    xs = sorted(counts.keys())
    ys = [counts[x] for x in xs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), facecolor="#0d0d0d",
                                    gridspec_kw={"height_ratios": [1, 1]})

    # ── Bar chart ─────────────────────────────────────────────────────────────
    ax1.set_facecolor("#0d0d0d")
    ax1.bar(xs, ys, width=0.6, color="#8e24aa", alpha=0.85, edgecolor="#5c0080", linewidth=0.5)
    for x, y_val in zip(xs, ys):
        ax1.text(x, y_val + 0.3, str(y_val), ha="center", va="bottom", color="white", fontsize=7)
    ax1.set_yscale("log")
    ax1.set_xlabel("number of 4-bit words (total_length)", color="white", fontsize=11)
    ax1.set_ylabel("count (log scale)", color="white", fontsize=11)
    ax1.set_title("sequence length distribution", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333333")
    ax1.grid(True, color="#222222", linewidth=0.4, axis="y")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs], color="white", fontsize=8)

    # ── Pie chart — group lengths > 8 as "long" ───────────────────────────────
    ax2.set_facecolor("#0d0d0d")
    threshold = 8
    pie_labels, pie_values = [], []
    long_count = 0
    for x, y_val in zip(xs, ys):
        if x <= threshold:
            pie_labels.append(f"{x} words")
            pie_values.append(y_val)
        else:
            long_count += y_val
    if long_count > 0:
        pie_labels.append(f"> {threshold} words")
        pie_values.append(long_count)

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(pie_values)))
    wedges, texts, autotexts = ax2.pie(
        pie_values, labels=pie_labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "#0d0d0d", "linewidth": 1.2}
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
    ax2.set_title("proportion by length", color="white", fontsize=11)

    fig.suptitle(f"binary information flow — {title}  |  n={len(codes)}",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=OUTPUT_DIR, help="Directory with sequence CSV files")
    parser.add_argument("--csv",   default=None,       help="Single sequence CSV file")
    args = parser.parse_args()

    if args.csv:
        files = [args.csv]
    else:
        files = sorted(glob.glob(os.path.join(args.input, "*_sequences.csv")))

    if not files:
        print("No sequence files found.")
        return

    # ── Single file: one plot ─────────────────────────────────────────────────
    if len(files) == 1:
        codes, lengths = load_sequences(files[0])
        basename = os.path.splitext(os.path.basename(files[0]))[0]
        outpath  = os.path.join(args.input, f"{basename}_log.png")
        plot(codes, lengths, basename, outpath)

    # ── Multiple files: merge all sequences into one plot ─────────────────────
    else:
        all_codes, all_lengths = [], []
        for f in files:
            c, l = load_sequences(f)
            all_codes.extend(c)
            all_lengths.extend(l)
        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "binary_flow_all_log.png")
        plot(all_codes, all_lengths, f"all files ({len(files)} CSVs, {len(all_codes)} sequences)", outpath)


if __name__ == "__main__":
    main()
