"""
Scatterplot of RMSE vs. Mean Inference Time for LLMs (Temperature: 0.1)
Reads data from results/benchmark-eval/model_vs_human_stats_0.1.md and saves a plot.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_utils import PlotUtils as PU


def parse_markdown_table(md_path: Path) -> pd.DataFrame:
    """Parse a markdown table into a DataFrame."""
    with md_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    # Find header
    for i, line in enumerate(lines):
        if line.startswith('|') and 'RMSE' in line:
            header = [h.strip() for h in line.strip('|').split('|')]
            start = i + 2  # skip header and separator
            break
    else:
        raise ValueError('No markdown table header found.')
    rows = []
    for line in lines[start:]:
        if not line.startswith('|'):
            continue
        row = [cell.strip().replace('**','') for cell in line.strip('|').split('|')]
        if len(row) == len(header):
            rows.append(row)
    df = pd.DataFrame(rows, columns=header)
    # Convert numeric columns
    for col in ['RMSE', 'Correlation', 'Mean Time Sec', 'Stdev Time Sec']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_rmse_vs_time(df: pd.DataFrame, outpath: Path, temperature: float = 0.1, light: bool = True) -> None:
    PU.init_plot_style(light=light, large=True)
    palette = sns.color_palette(PU.get_palette(), n_colors=len(df))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['Mean Time Sec'], df['RMSE'], s=280, c=palette, alpha=0.85)
    # Annotate each point
    for _, row in df.iterrows():
        ax.text(row['Mean Time Sec'] + 0.5, row['RMSE'], row['Model'], fontsize=18, va='center')
    # No trendline or background range
    ax.set_xlabel('Mean Inference Time (seconds)', fontsize=18)
    ax.set_ylabel('RMSE (lower is better)', fontsize=18)
    # No title
    ax.set_xlim(0, 50)
    ax.set_ylim(0.25, 0.55)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    print(f"Saved plot to {outpath}")
    plt.close(fig)

def main() -> None:
    path = Path('results') / 'benchmark-eval'
    md_path = path / 'model_vs_human_stats_0.1.md'
    outpath = path / 'rmse_vs_time_scatterplot.png'
    df = parse_markdown_table(md_path)
    plot_rmse_vs_time(df, outpath, temperature=0.1, light=True)

if __name__ == '__main__':
    main()
