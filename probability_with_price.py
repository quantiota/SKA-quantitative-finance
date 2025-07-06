import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --- Define all 9 transitions and their (color, marker) ---
transition_palette = {
    'bull->bull':    ('#2ECC40', 'o'),
    'bull->bear':    ('#FF4136', 's'),
    'bull->neutral': ('#FFDC00', 'v'),
    'bear->bull':    ('#0074D9', '^'),
    'bear->bear':    ('#B10DC9', '<'),
    'bear->neutral': ('#FF851B', '>'),
    'neutral->bull': ('#39CCCC', 'P'),
    'neutral->bear': ('#F012BE', 'X'),
    'neutral->neutral': ('#111111', 'D')
}

# --- 1. Load all 9 transition CSVs ---
dfs = []
for ttype in transition_palette.keys():
    fname = f'transition_{ttype.replace("->", "_")}.csv'
    try:
        df_t = pd.read_csv(fname)
        df_t['transition'] = ttype
        dfs.append(df_t)
    except FileNotFoundError:
        print(f"Warning: File not found: {fname}")
# Only keep non-empty dataframes
dfs = [df_t for df_t in dfs if not df_t.empty]
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('trade_sequence').reset_index(drop=True)

# --- 2. Compute transition probabilities ---
df['entropy_prev'] = df['entropy'].shift(1)
df['P'] = np.exp(-np.abs((df['entropy'] - df['entropy_prev']) / df['entropy']))

# --- 3. Compute transition percentages dynamically ---
transition_counts = df['transition'].value_counts(dropna=True)
transition_percentages = (transition_counts / transition_counts.sum() * 100).round(1).astype(str) + '%'

# Guarantee all 9 transitions (fill missing with 0.0%)
all_transitions = list(transition_palette.keys())
transition_percentages = {k: transition_percentages.get(k, '0.0%') for k in all_transitions}

# --- 4. Create subplots: two rows, shared x-axis ---
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=150, sharex=True,
                               gridspec_kw={'height_ratios': [2.5, 1]})

# --- 5. Top plot: transition probabilities ---
for ttype, (color, marker) in transition_palette.items():
    mask = df['transition'] == ttype
    percent = transition_percentages.get(ttype, '0.0%')
    label = f"{ttype.replace('->', '→')} ({percent})"
    ax1.scatter(
        df.loc[mask, 'trade_sequence'],
        df.loc[mask, 'P'],
        color=color,
        label=label,
        marker=marker,
        s=48,
        alpha=0.85,
        edgecolors='k',
        linewidths=0.6,
        zorder=3
    )

ax1.set_ylabel(r'Transition Probability $P = e^{-|\Delta H/H|}$')
ax1.set_title('Transition Probability and Price by Trade Sequence and Transition Type')
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25, zorder=1)

leg = ax1.legend(
    title="Transition Type", markerscale=1.15, bbox_to_anchor=(1.01, 1.01),
    loc='upper left', frameon=True, fancybox=True, framealpha=0.92, borderpad=1
)

# --- 6. Formula legend below main legend ---
formula_text = (
    r"$P_{i} = \exp\left( -\left| \frac{H_{i} - H_{i-1}}{H_{i}} \right| \right)$"
    "\n"
    r"$P_{i}$:  trade probability"
    "\n"
    r"$H_{i}$: current entropy"
    "\n"
    r"$H_{i-1}$: previous entropy"
    "\n"
    r"$i$: trade sequence"
)
dummy = mlines.Line2D([], [], color='none')  # No visible handle
formula_leg = ax1.legend(
    [dummy], [formula_text],
    loc='upper left', bbox_to_anchor=(1.01, 0.50),
    fontsize=12, frameon=True, fancybox=True, framealpha=0.92, borderpad=1
)
ax1.add_artist(leg)

# --- 7. Bottom plot: price line ---
trade_seq = np.asarray(df['trade_sequence'])
prices = np.asarray(df['price'])
ax2.plot(trade_seq, prices, color='gray', linewidth=1.5, zorder=2)
ax2.set_xlabel('Trade Sequence')
ax2.set_ylabel('Price')
ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.36, zorder=1)

plt.tight_layout(h_pad=2.0)
plt.savefig('probability_with_price.png', dpi=300, bbox_inches='tight')
plt.savefig('probability_with_price.svg', bbox_inches='tight')
plt.show()

