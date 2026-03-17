"""
SKA Paired Cycle Trading — Bot v1 Signal Logic
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLORS = {
    'neutral→bull':  '#A8DFBC',
    'bull→neutral':  '#C8F0A8',
    'neutral→bear':  '#FFAAAA',
    'bear→neutral':  '#FFD0A0',
    'neutral→neutral': '#E8E8E8',
    'bg':            '#FFFFFF',
    'text':          '#222222',
}

def draw_transition(ax, x, y, label, color, action=None, width=1.8, height=0.55):
    box = mpatches.FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle='round,pad=0.05',
        facecolor=color, edgecolor='#AAAAAA', linewidth=1.5, alpha=0.92
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='black')
    if action:
        ax.text(x, y - height/2 - 0.18, action, ha='center', va='top',
                fontsize=8, color=COLORS['text'], fontstyle='italic')

def draw_arrow(ax, x1, x2, y, color='#888888'):
    ax.annotate('', xy=(x2 - 0.92, y), xytext=(x1 + 0.92, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))


fig, axes = plt.subplots(1, 1, figsize=(18, 6))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('SKA Paired Cycle Trading — v1 Signal Logic',
             fontsize=15, fontweight='bold', color=COLORS['text'], y=1.02)

ax = axes
ax.set_facecolor(COLORS['bg'])
ax.set_xlim(-1, 17)
ax.set_ylim(-1.2, 2.6)
ax.axis('off')

ax.text(7.5, 1.85, 'v1 — Consecutive same-direction paired cycles', ha='center', va='center',
        fontsize=13, fontweight='bold', color=COLORS['text'])
ax.text(7.5, 1.45, 'Hold through repeated same-direction cycles — close only when opposite paired cycle confirms',
        ha='center', va='center', fontsize=8.5, color='#888888', fontstyle='italic')

for seq_idx, (side, nb_color, exit_open, exit_open_color, exit_close, exit_close_color, repeat_color) in enumerate([
    ('LONG',  COLORS['neutral→bull'], 'neutral→bear', COLORS['neutral→bear'], 'bear→neutral', COLORS['bear→neutral'], COLORS['bull→neutral']),
    ('SHORT', COLORS['neutral→bear'], 'neutral→bull', COLORS['neutral→bull'], 'bull→neutral', COLORS['bull→neutral'], COLORS['bear→neutral']),
]):
    y = 0.5 - seq_idx * 1.1
    side_color = '#2E8B57' if side == 'LONG' else '#CC2222'
    ax.text(-0.5, y, side, ha='center', va='center',
            fontsize=10, fontweight='bold', color=side_color)

    open_trans = 'neutral→bull' if side == 'LONG' else 'neutral→bear'
    pair_trans = 'bull→neutral' if side == 'LONG' else 'bear→neutral'

    draw_transition(ax, 1.2, y, open_trans,       nb_color,          'OPEN\nWAIT_PAIR')
    draw_arrow(ax, 1.2, 3.8, y)
    draw_transition(ax, 3.8, y, pair_trans,        repeat_color,      'pair confirmed\nIN_NEUTRAL')
    draw_arrow(ax, 3.8, 6.4, y)
    draw_transition(ax, 6.4, y, 'neutral→neutral\n× N  (N≥3)', COLORS['neutral→neutral'], 'neutral gap\nREADY')
    # Loop arrow back
    ax.annotate('', xy=(1.2, y + 0.45), xytext=(6.4, y + 0.45),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5,
                                connectionstyle='arc3,rad=-0.4'))
    if seq_idx == 0:
        ax.text(3.8, y + 0.72, '↺  repeats', ha='center', va='center',
                fontsize=8, color='#888888', fontstyle='italic')
    draw_arrow(ax, 6.4, 9.2, y)
    draw_transition(ax, 9.2, y, exit_open,         exit_open_color,   'opp. cycle opens\nEXIT_WAIT')
    draw_arrow(ax, 9.2, 12.0, y)
    draw_transition(ax, 12.0, y, exit_close,        exit_close_color,  'opp. pair confirmed\nCLOSE ' + side)

plt.tight_layout()
outfile = '/home/coder/project/Real_Time_SKA_trading/batch_trading_bot/schema_bot_versions.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print(f'Saved: {outfile}')
