import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('questdb-query-1751544843847.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

print(f"Loaded {len(data)} total trades")

# --- Calculate price returns ---
data['price_return'] = data['price'].pct_change() * 100
data = data.dropna(subset=['price_return'])

# --- Classic three trajectory split ---
bear_data = data[data['price_return'] < 0].copy()
bull_data = data[data['price_return'] > 0].copy()
neutral_data = data[data['price_return'] == 0].copy()

# Normalize trade_id for plotting/exports
min_trade_id = data['trade_id'].min()
bear_data['trade_sequence'] = bear_data['trade_id'] - min_trade_id
bull_data['trade_sequence'] = bull_data['trade_id'] - min_trade_id
neutral_data['trade_sequence'] = neutral_data['trade_id'] - min_trade_id

# --- Compute transition types (SKA full regime) ---
def classify_state(r):
    if pd.isna(r):
        return None
    elif r > 0:
        return 'bull'
    elif r < 0:
        return 'bear'
    else:
        return 'neutral'

data['state'] = data['price_return'].apply(classify_state)
data['prev_state'] = data['state'].shift(1)
data['transition_type'] = data['prev_state'] + '→' + data['state']
data = data.dropna(subset=['transition_type'])

data['trade_sequence'] = data['trade_id'] - min_trade_id

# Assign shape/color per transition type
transition_palette = {
    'bull→bull':    ('#2ECC40', 'o'),
    'bull→bear':    ('#FF4136', 's'),
    'bull→neutral': ('#FFDC00', 'v'),
    'bear→bull':    ('#0074D9', '^'),
    'bear→bear':    ('#B10DC9', '<'),
    'bear→neutral': ('#FF851B', '>'),
    'neutral→bull': ('#39CCCC', 'P'),
    'neutral→bear': ('#F012BE', 'X'),
    'neutral→neutral': ('#111111', 'D')
}

# --- Calculate counts and percentages ---
transition_counts = data['transition_type'].value_counts().sort_index()
total_transitions = transition_counts.sum()
transition_percents = 100 * transition_counts / total_transitions

print("\nTransition counts and percentages:")
for ttype in transition_palette.keys():
    count = transition_counts.get(ttype, 0)
    percent = transition_percents.get(ttype, 0.0)
    print(f"{ttype:18}: {count:5}  ({percent:5.2f}%)")

# --- Export per-transition CSVs ---
for ttype in transition_palette.keys():
    subset = data[data['transition_type'] == ttype]
    outname = f'transition_{ttype.replace("→", "_")}.csv'
    subset[['trade_sequence', 'price', 'entropy']].to_csv(outname, index=False)
    print(f"Exported {len(subset)} rows to {outname}")

# --- Prepare info for box ---
asset_symbol = str(data['symbol'].iloc[0]) if 'symbol' in data.columns else "UNKNOWN"
first_trade_id = int(data['trade_id'].min())
last_trade_id = int(data['trade_id'].max())
first_time = data['timestamp'].min()
last_time = data['timestamp'].max()
first_time_str = first_time.strftime('%H:%M:%S')
last_time_str = last_time.strftime('%H:%M:%S')
date_str = first_time.strftime('%Y-%m-%d')

box_content = (
    f"{asset_symbol} Binance\n"
    f"{date_str} {first_time_str}-{last_time_str} UTC\n"
    f"Trades: {first_trade_id} - {last_trade_id}"
)

# --- Plot with percentage in the legend and info box ---
plt.figure(figsize=(15, 7))
for ttype, (color, marker) in transition_palette.items():
    mask = data['transition_type'] == ttype
    percent = transition_percents.get(ttype, 0.0)
    label = f"{ttype} ({percent:.1f}%)"
    plt.scatter(
        data.loc[mask, 'trade_sequence'],
        data.loc[mask, 'entropy'],
        label=label,
        color=color,
        s=36,
        alpha=0.85,
        marker=marker,
        edgecolors='white',
        linewidths=0.6,
    )
plt.xlabel("Trade Sequence (trade_id - min)", fontsize=12)
plt.ylabel("Entropy", fontsize=12)
plt.title("Entropy vs. Trade ID (Color + Shape by Transition Type)", fontsize=15)
plt.legend(markerscale=1.3, fontsize=11, ncol=3, loc='lower right', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Add info box at bottom right ---
plt.annotate(
    box_content,
    xy=(0.9, 0.2), xycoords='axes fraction',
    fontsize=10,
    ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', edgecolor='gray', alpha=0.95)
)

plt.savefig('entropy_vs_treade_id.png')
plt.show()

print("\nDone. All transitions exported and multi-band plot generated.")
