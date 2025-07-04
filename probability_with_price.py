import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

# 1. Load data
bull = pd.read_csv('bull_trajectory_norm.csv')
bull['regime'] = 'bull'

bear = pd.read_csv('bear_trajectory_norm.csv')
bear['regime'] = 'bear'

neutral = pd.read_csv('neutral_trajectory_norm.csv')
neutral['regime'] = 'neutral'

# 2. Combine and sort
df = pd.concat([bull, bear, neutral], ignore_index=True)
df = df.sort_values('trade_sequence').reset_index(drop=True)

# 3. Compute regime transitions and entropy probability
df['regime_prev'] = df['regime'].shift(1)
df['transition'] = df['regime_prev'] + '->' + df['regime']
df.loc[df['regime_prev'].isna(), 'transition'] = None

df['entropy_prev'] = df['entropy'].shift(1)
df['P'] = np.exp(-np.abs((df['entropy'] - df['entropy_prev']) / df['entropy']))

transition_percentages = {
    'neutral->neutral': '',
    'neutral->bull': '',
    'neutral->bear': '',
    'bull->neutral': '',
    'bull->bull': '',
    'bull->bear': '',
    'bear->neutral': '',
    'bear->bull': '',
    'bear->bear': ''
}

# 4. Define color mapping (tab10 for accessibility)
transition_keys = [
    'neutral->neutral', 'bull->neutral', 'neutral->bull', 'neutral->bear',
    'bear->neutral', 'bear->bull', 'bull->bear', 'bear->bear', 'bull->bull'
]
cmap = cm.get_cmap('tab10', len(transition_keys))
transition_colors = {k: cmap(i) for i, k in enumerate(transition_keys)}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.title_fontsize': 13
})

# 5. Create subplots: two rows, shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=150, sharex=True,
                               gridspec_kw={'height_ratios': [2.5, 1]})

# 6. Top plot: probability transitions
for trans, color in transition_colors.items():
    mask = df['transition'] == trans
    ax1.scatter(
        df.loc[mask, 'trade_sequence'],
        df.loc[mask, 'P'],
        color=color,
        label=f"{trans} ({transition_percentages.get(trans, '')})",
        s=48,
        alpha=0.85,
        edgecolors='k',
        linewidths=0.6,
        zorder=3
    )

ax1.set_ylabel(r'Transition Probability $P = e^{-|\Delta H/H|}$')
ax1.set_title('Transition Probability and Price by Trade Sequence and Transition Type')
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.25, zorder=1)

for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(True)

leg = ax1.legend(
    title="Transition Type", markerscale=1.15, bbox_to_anchor=(1.01, 1.01),
    loc='upper left', frameon=True, fancybox=True, framealpha=0.92, borderpad=1
)

# 2. FORMULA LEGEND, styled and placed directly below
formula_text = (
    r"$P_{i} = \exp\left( -\left| \frac{H_{i} - H_{i-1}}{H_{i}} \right| \right)$"
    "\n"
    r"$P_{i}$:  trade probability"
    "\n"
    r"$H_{i}$: current entropy"
    "\n"
    r"$H_{i-1}$: previous entropy"
     "\n"
    r"$H_{i}$: trade sequence"
)
dummy = mlines.Line2D([], [], color='none')  # No visible handle

# Formula legend, just a bit below (e.g., y=0.55)
formula_leg = ax1.legend(
    [dummy], [formula_text],
    loc='upper left', bbox_to_anchor=(1.01, 0.50),
    fontsize=12, frameon=True, fancybox=True, framealpha=0.92, borderpad=1
)
ax1.add_artist(leg)


# 7. Bottom plot: price line, robust conversion to 1D arrays
trade_seq = np.asarray(df['trade_sequence'])
prices = np.asarray(df['price'])

# Optional: quick check output for debugging, can comment out later
print("trade_seq shape:", trade_seq.shape)
print("prices shape:", prices.shape)
print(prices[:5])

ax2.plot(trade_seq, prices, color='gray', linewidth=1.5, zorder=2)
ax2.set_xlabel('Trade Sequence')
ax2.set_ylabel('Price')
ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.36, zorder=1)

for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(True)

plt.tight_layout(h_pad=2.0)
plt.savefig('probability_with_price.png', dpi=300, bbox_inches='tight')
plt.savefig('probability_with_price.svg', bbox_inches='tight')
plt.show()


