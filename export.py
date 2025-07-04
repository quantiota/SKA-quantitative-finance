import pandas as pd
import numpy as np

# Load your data
data = pd.read_csv('questdb-query-1751544843847.csv')

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

print(f"Loaded {len(data)} total trades")

# Calculate price returns
data['price_return'] = data['price'].pct_change() * 100  # Convert to percentage

# Remove the first row (NaN price return)
data = data.dropna(subset=['price_return'])

# Separate into THREE trajectories based on price returns
bear_data = data[data['price_return'] < 0].copy()    # Negative returns
bull_data = data[data['price_return'] > 0].copy()    # Positive returns  
neutral_data = data[data['price_return'] == 0].copy() # Zero returns

print(f"\nSeparated data:")
print(f"Bear trades (negative returns): {len(bear_data)}")
print(f"Bull trades (positive returns): {len(bull_data)}")
print(f"Neutral trades (zero returns): {len(neutral_data)}")
print(f"Total: {len(bear_data) + len(bull_data) + len(neutral_data)}")

# Calculate percentages
total_directional = len(data)
bear_pct = len(bear_data) / total_directional * 100
bull_pct = len(bull_data) / total_directional * 100
neutral_pct = len(neutral_data) / total_directional * 100

print(f"\nPercentages:")
print(f"Bear: {bear_pct:.1f}%")
print(f"Bull: {bull_pct:.1f}%")
print(f"Neutral: {neutral_pct:.1f}%")

# Normalize trade_id by subtracting minimum across ALL data
min_trade_id = data['trade_id'].min()

bear_data['trade_sequence'] = bear_data['trade_id'] - min_trade_id
bull_data['trade_sequence'] = bull_data['trade_id'] - min_trade_id
neutral_data['trade_sequence'] = neutral_data['trade_id'] - min_trade_id

# Extract values for TikZ
first_trade_id = int(data['trade_id'].min())
last_trade_id = int(data['trade_id'].max())
first_time = data['timestamp'].min()
last_time = data['timestamp'].max()
total_trades = len(data)
bear_count = len(bear_data)
bull_count = len(bull_data)
neutral_count = len(neutral_data)

# Format times for LaTeX (compact format)
first_time_str = first_time.strftime('%H:%M:%S')
last_time_str = last_time.strftime('%H:%M:%S')
date_str = first_time.strftime('%Y-%m-%d')

print(f"\nTimestamp Information:")
print(f"First trade: {first_time} UTC")
print(f"Last trade: {last_time} UTC")
print(f"Duration: {last_time - first_time}")

# Save ALL THREE normalized trajectories with PRICE included
bear_data[['trade_sequence', 'price', 'entropy']].to_csv('bear_trajectory_norm.csv', index=False)
bull_data[['trade_sequence', 'price', 'entropy']].to_csv('bull_trajectory_norm.csv', index=False)
neutral_data[['trade_sequence', 'price', 'entropy']].to_csv('neutral_trajectory_norm.csv', index=False)

print(f"\n✅ Files exported:")
print(f"📁 bear_trajectory_norm.csv: {len(bear_data)} records (trade_sequence, price, entropy)")
print(f"📁 bull_trajectory_norm.csv: {len(bull_data)} records (trade_sequence, price, entropy)")
print(f"📁 neutral_trajectory_norm.csv: {len(neutral_data)} records (trade_sequence, price, entropy)")

# ==========================================
# TIKZ VALUES FOR DIRECT COPY-PASTE
# ==========================================
print(f"\n" + "="*60)
print(f"📊 TIKZ VALUES FOR DIRECT COPY-PASTE:")
print(f"="*60)

print(f"\n🔢 Individual Values:")
print(f"first_trade_id = {first_trade_id}")
print(f"last_trade_id = {last_trade_id}")
print(f"date = {date_str}")
print(f"first_time = {first_time_str}")
print(f"last_time = {last_time_str}")
print(f"total_trades = {total_trades}")
print(f"bear_count = {bear_count}")
print(f"bull_count = {bull_count}")
print(f"neutral_count = {neutral_count}")

print(f"\n📋 READY-TO-USE TIKZ NODE (COMPACT):")
print(f"-"*40)

# Option 1: Compact version with three trajectories
tikz_compact = f"""% Data source box - THREE TRAJECTORIES
\\node[draw, fill=gray!10, align=center, font=\\tiny, 
      anchor=south east, inner sep=2pt] 
      at (rel axis cs:0.98,0.02) {{
	\\textbf{{XRPUSDT Binance}} \\\\
	{date_str} {first_time_str}-{last_time_str} UTC \\\\
	Trades: {first_trade_id} - {last_trade_id} \\\\
	Bear: {bear_count} | Bull: {bull_count} | Neutral: {neutral_count}
}};"""

print(tikz_compact)

print(f"\n" + "-"*40)

# Option 2: Detailed version with percentages
tikz_detailed = f"""% Detailed data source box - THREE TRAJECTORIES
\\node[draw, fill=white, align=left, font=\\scriptsize, 
      anchor=south east, inner sep=3pt] 
      at (rel axis cs:0.98,0.02) {{
	\\textbf{{Data Source:}} \\\\
	XRPUSDT Binance \\\\
	First Trade: {first_trade_id} \\\\
	Last Trade: {last_trade_id} \\\\
	Start: {date_str} {first_time_str} UTC \\\\
	End: {date_str} {last_time_str} UTC \\\\
	Total: {total_trades:,} trades \\\\
	Bear: {bear_count} ({bear_pct:.1f}\\%) \\\\
	Bull: {bull_count} ({bull_pct:.1f}\\%) \\\\
	Neutral: {neutral_count} ({neutral_pct:.1f}\\%)
}};"""

print(tikz_detailed)

print(f"\n📋 COMPLETE LATEX CODE FOR THREE TRAJECTORIES (WITH PRICE):")
print(f"-"*50)

latex_code = f"""\\documentclass{{standalone}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\usepackage{{pgfplotstable}}
\\begin{{document}}
	\\begin{{tikzpicture}}
		\\begin{{axis}}[
			width=10cm, height=7cm,
			xlabel={{\\textbf{{Trade ID}}}},
			ylabel={{\\textbf{{Entropy}}}},
			title={{\\textbf{{Entropy Trajectories by Price Return}}}},
			legend style={{at={{(0.98,0.98)}},anchor=north east, draw=none, font=\\small}},
			tick label style={{font=\\small}},
			label style={{font=\\bfseries\\small}},
			title style={{font=\\bfseries\\small}},
			every axis plot/.append style={{thick}},
			grid=major,
			% Axis limits removed for automatic scaling
			]
			\\addplot+[only marks, mark=*, mark size=0.05pt, color=red, opacity=0.7]
			table [x=trade_sequence, y=entropy, col sep=comma] {{bear_trajectory_norm.csv}};
			\\addlegendentry{{Price return negative (bear, red)}}
			
			\\addplot+[only marks, mark=*, mark size=0.05pt, color=green!70!black, opacity=0.7]
			table [x=trade_sequence, y=entropy, col sep=comma] {{bull_trajectory_norm.csv}};
			\\addlegendentry{{Price return positive (bull, green)}}
			
			\\addplot+[only marks, mark=*, mark size=0.05pt, color=blue, opacity=0.7]
			table [x=trade_sequence, y=entropy, col sep=comma] {{neutral_trajectory_norm.csv}};
			\\addlegendentry{{Price return zero (neutral, blue)}}
			
			% Data source box - THREE TRAJECTORIES
			\\node[draw, fill=gray!10, align=center, font=\\tiny, 
			anchor=south east, inner sep=2pt] 
			at (rel axis cs:0.98,0.02) {{
				\\textbf{{XRPUSDT Binance}} \\\\
				{date_str} {first_time_str}-{last_time_str} UTC \\\\
				Trades: {first_trade_id} - {last_trade_id} \\\\
				Bear: {bear_count} | Bull: {bull_count} | Neutral: {neutral_count}
			}};
		\\end{{axis}}
	\\end{{tikzpicture}}
\\end{{document}}

% Alternative plot using price data:
\\begin{{tikzpicture}}
	\\begin{{axis}}[
		width=10cm, height=7cm,
		xlabel={{\\textbf{{Trade ID}}}},
		ylabel={{\\textbf{{Price (USDT)}}}},
		title={{\\textbf{{Price Trajectories by Return Type}}}},
		legend style={{at={{(0.98,0.98)}},anchor=north east, draw=none, font=\\small}},
		]
		\\addplot+[only marks, mark=*, mark size=0.05pt, color=red, opacity=0.7]
		table [x=trade_sequence, y=price, col sep=comma] {{bear_trajectory_norm.csv}};
		\\addlegendentry{{Bear trades}}
		
		\\addplot+[only marks, mark=*, mark size=0.05pt, color=green!70!black, opacity=0.7]
		table [x=trade_sequence, y=price, col sep=comma] {{bull_trajectory_norm.csv}};
		\\addlegendentry{{Bull trades}}
		
		\\addplot+[only marks, mark=*, mark size=0.05pt, color=blue, opacity=0.7]
		table [x=trade_sequence, y=price, col sep=comma] {{neutral_trajectory_norm.csv}};
		\\addlegendentry{{Neutral trades}}
	\\end{{axis}}
\\end{{tikzpicture}}"""

print(latex_code)

print(f"\n" + "="*60)
print(f"🚀 THREE ENTROPY UNIVERSES WITH PRICE DATA DISCOVERED!")
print(f"="*60)

# Advanced entropy analysis
print(f"\n🧬 ENTROPY UNIVERSE ANALYSIS:")
print(f"Bear Universe: {bear_count} trades ({bear_pct:.1f}%) - Negative price movement")
print(f"Bull Universe: {bull_count} trades ({bull_pct:.1f}%) - Positive price movement") 
print(f"Neutral Universe: {neutral_count} trades ({neutral_pct:.1f}%) - Price consolidation")

print(f"\n📊 MARKET MICROSTRUCTURE INSIGHTS:")
print(f"• Directional movement: {bear_count + bull_count} trades ({(bear_pct + bull_pct):.1f}%)")
print(f"• Price consolidation: {neutral_count} trades ({neutral_pct:.1f}%)")
print(f"• Consolidation dominance: {neutral_pct / (bear_pct + bull_pct):.1f}x more than directional")

# Show sample data for verification
print(f"\n🔍 Sample bear data:")
print(bear_data[['trade_id', 'trade_sequence', 'price', 'price_return', 'entropy', 'timestamp']].head())

print(f"\n🔍 Sample bull data:")
print(bull_data[['trade_id', 'trade_sequence', 'price', 'price_return', 'entropy', 'timestamp']].head())

print(f"\n🔍 Sample neutral data:")
print(neutral_data[['trade_id', 'trade_sequence', 'price', 'price_return', 'entropy', 'timestamp']].head())

# Entropy statistics
print(f"\n📈 ENTROPY STATISTICS:")
print(f"Bear entropy range: {bear_data['entropy'].min():.6f} to {bear_data['entropy'].max():.6f}")
print(f"Bull entropy range: {bull_data['entropy'].min():.6f} to {bull_data['entropy'].max():.6f}")
print(f"Neutral entropy range: {neutral_data['entropy'].min():.6f} to {neutral_data['entropy'].max():.6f}")

print(f"\nBear entropy mean: {bear_data['entropy'].mean():.6f}")
print(f"Bull entropy mean: {bull_data['entropy'].mean():.6f}")
print(f"Neutral entropy mean: {neutral_data['entropy'].mean():.6f}")

# Price statistics
print(f"\n💰 PRICE STATISTICS:")
print(f"Bear price range: ${bear_data['price'].min():.6f} to ${bear_data['price'].max():.6f}")
print(f"Bull price range: ${bull_data['price'].min():.6f} to ${bull_data['price'].max():.6f}")
print(f"Neutral price range: ${neutral_data['price'].min():.6f} to ${neutral_data['price'].max():.6f}")

print(f"\nBear price mean: ${bear_data['price'].mean():.6f}")
print(f"Bull price mean: ${bull_data['price'].mean():.6f}")
print(f"Neutral price mean: ${neutral_data['price'].mean():.6f}")