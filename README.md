
# SKA-quant

## Project Description

**SKA-quant** is a real-time quantitative research project focused on high-frequency cryptocurrency trading data, using a **Stochastic Kinetic Analysis (SKA)** framework for insights at tick-level resolution. It processes raw trade-level (tick-by-tick) data to compute **entropy trajectories** – a measure of uncertainty or disorder in price movements – and dynamically classifies these into three directional market states: **bear**, **bull**, or **neutral**. By applying a dynamic entropy-based regime classification, SKA-quant provides granular insight into market microstructure, revealing how market behavior oscillates between trending and consolidating phases in real time.

## Key Features

* **Real-Time Data Ingestion & Preprocessing**: Handles high-frequency streaming or historical tick data, cleaning and normalizing it in real time for analysis. This includes converting timestamps and calculating immediate price returns for each trade.
* **Entropy Computation & Regime Segmentation**: Calculates an entropy value for each trade (representing market disorder at that instant) and segments the trade stream into regimes: *bull* (positive price returns), *bear* (negative price returns), and *neutral* (zero price change). This dynamic, entropy-based classification uncovers subtle shifts in market regime that traditional indicators might miss.
* **Export of Normalized Trajectories**: Outputs normalized entropy trajectory data for each regime. After processing, the tool exports three CSV files containing sequences of trades with their entropy values and prices, separated by regime (bull, bear, neutral) for further analysis or validation.
* **Entropy Transition Probability Modeling**: Analyzes the likelihood of transitioning from one regime to another using entropy-based probabilities. For each consecutive trade, it computes a transition probability $P = \exp\!\Big(-\big|\frac{H_i - H_{i-1}}{H_i}\big|\Big)$ (where $H_i$ is the current trade entropy) to quantify how sharply entropy changes. This helps model the stability of regimes and the abruptness of market shifts.
* **Visualization Tools**: Provides built-in support for visual analysis. One tool generates high-quality **entropy trajectory plots** (via LaTeX **TikZ/PGFPlots** for publication-ready PDFs) showing entropy against trade sequence for each regime. Another visualization overlays **transition probabilities** on a price chart, producing scatter plots of entropy transition probability colored by regime change type, with the underlying price trend for context.

## Installation

Before using SKA-quant, ensure you have the required environment:

* **Python 3.x** and the following Python libraries:

  * `pandas` (for data handling)
  * `numpy` (for numerical computations)
  * `matplotlib` (for plotting)
* *(Optional)* **LaTeX** distribution (with TikZ/PGFPlots packages) if you plan to compile the entropy plot from the LaTeX code.

You can install the Python dependencies with pip:

```bash
pip install pandas numpy matplotlib
```

If you intend to use the LaTeX plotting feature, make sure a LaTeX engine (like TeX Live or MikTeX) is installed and accessible (the code will output a `.tex` snippet for the entropy plot which you can compile manually).

## Usage

Follow these steps to run the analysis and generate outputs:

1. **Prepare Data**: Obtain your high-frequency trade data. The input should be a CSV or database export with columns such as `trade_id`, `timestamp`, `price`, and `entropy`. In our example, data is fetched via a QuestDB query (producing a file like `questdb-query-...csv`). Ensure this file is present or adjust the data loading in the scripts.

2. **Compute Entropy Trajectories**: Run the `export.py` script to ingest data and perform entropy trajectory computation and segmentation:

   ```bash
   python export.py
   ```

   This script will:

   * Load the raw trade data and convert timestamps to datetime objects.
   * Compute the price return for each trade (percentage change from the previous trade's price).
   * Categorize each trade into one of three trajectories based on price return:

     * **Bear** trajectory for trades with negative returns (price drops).
     * **Bull** trajectory for trades with positive returns (price rises).
     * **Neutral** trajectory for trades with zero price change (price unchanged).
   * **Normalize trade sequences** by re-indexing `trade_id` to a relative sequence (so all three trajectories share a common timeline axis).
   * Save three CSV files, one for each regime, containing the normalized trade sequence, price, and entropy values:

     * `bear_trajectory_norm.csv` – trades in bear regime (price down moves)
     * `bull_trajectory_norm.csv` – trades in bull regime (price up moves)
     * `neutral_trajectory_norm.csv` – trades in neutral regime (no price change)
   * Print out summary statistics, including the count and percentage of trades in each category (e.g., the percentage of trades classified as neutral, bull, or bear).

3. **Analyze Transition Probabilities**: After obtaining the normalized trajectories, run the transition probability visualization:

   ```bash
   python probability_with_price.py
   ```

   This script will:

   * Combine the three CSV trajectories into a single time-sequenced dataset.
   * Determine the regime transition for each consecutive trade (e.g., `neutral->neutral`, `neutral->bull`, `bull->bear`, etc.).
   * Compute an entropy-based transition probability for each trade using the formula mentioned above, which yields values close to 1 for small entropy changes (smooth regime continuation) and lower values for large entropy jumps (regime shifts).
   * Generate a plot of these transition probabilities and overlay the price sequence:

     * The **top subplot** is a scatter plot of transition probabilities vs. trade sequence, with each point colored by the type of regime transition. A legend indicates transition types (for example, a large cluster of `neutral->neutral` points shows the frequency of consecutive neutral trades, whereas fewer points might indicate `bull->bear` transitions).
     * The **bottom subplot** is the price chart over the same trade sequence, plotted in grayscale for reference, so you can correlate high or low probability events with price movements.
   * Save the combined figure as `probability_with_price.png` (and an SVG version for scalability).

4. **(Optional) Generate Entropy Plot via LaTeX**: The `export.py` script also outputs a LaTeX **TikZ** code snippet for plotting the entropy trajectories in a single figure. To use it:

   * Copy the LaTeX code printed by `export.py` (it includes embedding of the CSV data and styling).
   * Paste it into a `.tex` file with a proper LaTeX document wrapper (if not already included).
   * Compile the LaTeX file (e.g., run `pdflatex yourfile.tex`). If all goes well, this produces an `entropy_trajectories.pdf` showing three scatter plots (red for bear, green for bull, blue for neutral) of entropy values over the trade sequence, with a legend and summary statistics box. This PDF is suitable for high-resolution viewing or inclusion in reports.

## Output Files

After running the above steps, you will obtain the following output files:

* **Normalized Trajectory CSVs**: These files contain the processed data points for each regime, with columns for the normalized `trade_sequence`, `price`, and `entropy`. They can be used for further custom analysis or plotting.

  * `bear_trajectory_norm.csv` – Entropy trajectory for all bear-regime trades (price decline instances).
  * `bull_trajectory_norm.csv` – Entropy trajectory for all bull-regime trades (price increase instances).
  * `neutral_trajectory_norm.csv` – Entropy trajectory for all neutral-regime trades (no price change).
* **Visualizations**:

  * `entropy_trajectories.pdf` – A PDF plot generated via LaTeX/TikZ, illustrating entropy vs. trade sequence for each regime in a single chart. This high-quality figure is *TikZ-ready* for academic or publication use, offering precise rendering of thousands of trade points and a compact summary of the data source (trading pair, time range, trade counts, etc.).
  * `probability_with_price.png` – A PNG image of the transition probability visualization. It contains two aligned subplots: the top one showing entropy transition probabilities for each trade (color-coded by transition type), and the bottom one showing the asset price over the same sequence for context.

## Sample Insights

Using SKA-quant on high-frequency crypto data can reveal several interesting insights into market dynamics:

* **Dominance of Neutral Regimes**: Markets spend a significant amount of time in a *neutral* entropy state. In our sample analysis, about \~78% of sequential trade transitions remained in the neutral regime (little to no price change between trades), underscoring that consolidation or sideway markets are the prevalent condition. This dominance of neutral entropy regimes indicates that sharp bull or bear moves are comparatively infrequent, occurring in the remaining \~22% of cases.
* **Transition Hotspots & Volatility Clustering**: The transition probability scatter plot helps visually identify *hotspots* where regime changes cluster. For instance, a series of low-probability points (signaling large entropy jumps) might highlight a flurry of **bull→bear** or **bear→bull** transitions within a short time – often aligning with news events or breakout volatility. Such clusters confirm the phenomenon of volatility clustering: periods of high turbulence follow each other, which SKA-quant captures through successive entropy spikes.
* **Entropy-Price Relationship**: By overlaying price with transition probabilities, one can observe how price trends correspond to entropy stability. Stable prices often coincide with high transition probabilities (points near 1.0, usually neutral→neutral transitions), whereas price breakouts or reversals correspond to sudden drops in transition probability (indicating a regime shift). This provides an intuitive visualization of market microstructure shifts – for example, a steep price rally might show a sequence of **neutral→bull** transitions with declining $P$ values as entropy rises.
* **Publication-Quality Outputs**: Researchers and analysts benefit from the built-in LaTeX export. The entropy trajectories PDF can be directly used in papers or presentations, showing a clear separation of market regimes in entropy space. The combination of Python for computation and LaTeX for visualization means the results are reproducible and of high quality. The TikZ code includes annotations (like data source, time range, and percentage breakdown of regimes) making it easy to communicate the conditions of the experiment.

 *Example output from `probability_with_price.py`: Transition probability scatter plot overlaid with price. The **top panel** shows entropy-based transition probabilities $P$ for each trade, color-coded by transition type (see legend with percentages of each transition). The **bottom panel** shows the corresponding price trajectory for the trading period. We can see that **neutral→neutral** transitions (blue points) dominate the plot (reflecting the \~78.7% persistence in a neutral state), while infrequent regime changes like **bull→bear** or **bear→bull** (other colors) appear as isolated points, often coinciding with notable price moves.*

## License and Contributions

SKA-quant is an open-source project (released under the MIT License) intended for research and educational purposes. We welcome contributions from the quant research and data science community to enhance this framework. Collaboration and feedback are encouraged – together we can refine the SKA approach to better understand market microstructure through entropy analytics.

