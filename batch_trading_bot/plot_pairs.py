import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_DIR = '/XRPUSDT'
OUTPUT    = '/XRPUSDT/results/pairs_accumulation_questdb.png'

files = sorted([f for f in os.listdir(INPUT_DIR)
                if f.startswith('binance_trades_') and f.endswith('.csv')])

bull_acc  = 0
bear_acc  = 0
bull_series    = []
bear_series    = []
net_series     = []
rel_net_series = []
price_series   = []

for fname in files:
    data = pd.read_csv(os.path.join(INPUT_DIR, fname))

    if 'price' in data.columns:
        # Raw tick data: derive pairs from price direction transitions
        data['price_return'] = data['price'].pct_change() * 100
        data = data.dropna(subset=['price_return'])

        def classify_state(r):
            if pd.isna(r):   return None
            elif r > 0:      return 'bull'
            elif r < 0:      return 'bear'
            else:            return 'neutral'

        data['state']       = data['price_return'].apply(classify_state)
        data['prev_state']  = data['state'].shift(1)
        data['transition']  = data['prev_state'] + '→' + data['state']
        data = data.dropna(subset=['transition'])

        transitions = data['transition'].tolist()

        bull_open  = False
        bear_open  = False
        bull_pairs = 0
        bear_pairs = 0

        for t in transitions:
            if t == 'neutral→bull':
                bull_open = True
            elif t == 'bull→neutral' and bull_open:
                bull_pairs += 1
                bull_open = False
            elif t == 'neutral→bear':
                bear_open = True
            elif t == 'bear→neutral' and bear_open:
                bear_pairs += 1
                bear_open = False

        last_price = data['price'].iloc[-1]

    else:
        # Trade result data (v3 format): bull_pairs/bear_pairs per trade row
        bull_pairs = int(data['bull_pairs'].sum())
        bear_pairs = int(data['bear_pairs'].sum())
        last_price = float(data['exit'].iloc[-1])

    bull_acc += bull_pairs
    bear_acc += bear_pairs
    bull_series.append(bull_acc)
    bear_series.append(bear_acc)
    net_series.append(bull_acc - bear_acc)
    rel_net_series.append(bull_acc - bear_acc)
    price_series.append(last_price)

x = list(range(len(files)))

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

ax1.plot(x, bull_series, color='green', label='Bull pairs (cumulative)', linewidth=1.5)
ax1.plot(x, bear_series, color='red',   label='Bear pairs (cumulative)', linewidth=1.5)
ax1.set_ylabel('Accumulated pairs')
date_start = files[0].split('_')[2]
date_end   = files[-1].split('_')[2]
ax1.set_title(f'SKA — Accumulated Bull vs Bear Pairs ({date_start} → {date_end})')
ax1.legend()
ax1.grid(True, alpha=0.3)

colors = ['green' if v >= 0 else 'red' for v in net_series]
ax2.bar(x, net_series, color=colors, alpha=0.7, width=0.8)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel('Net (bull − bear)')
ax2.set_title('Net Accumulated Pairs = Bull − Bear')
ax2.grid(True, alpha=0.3)

ax3.plot(x, price_series, color='steelblue', linewidth=1.5)
ax3.set_ylabel('Price (USDT)')
ax3.set_xlabel('Loop index (3500 trades each)')
ax3.set_title('XRPUSDT Price')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
plt.savefig(OUTPUT, dpi=150)
print(f'Saved: {OUTPUT}')
print(f'Final: bull={bull_acc}  bear={bear_acc}  net={bull_acc - bear_acc}')
