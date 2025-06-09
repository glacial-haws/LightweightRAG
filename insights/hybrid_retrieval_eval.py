"""
Plots Recall and MRR@5/@8 for hybrid retrieval evaluation.
- Reads Excel data as in chunking_eval.py
- Plots a 2x2 grid: Recall@5, Recall@8, MRR@5, MRR@8
- Series: vector, bm25, combined (for recall and mrr)
- Y axis: questions question_00 to question_19 (sorted by combined_recall@5 desc)
- Uses PlotUtils for style/palette/labeling
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cbfk.log.log_config import LogConfig as LC
from insights.plot_utils import PlotUtils as PU


def plot_token_count_distributions(
    df: pd.DataFrame,
    results_dir: Path,
    column: str = "index.token_count_distribution",
    title: str = "Token Count Distributions",
    figsize: tuple[int, int] = (20, 6),
    alpha: float = 0.6,
    lw: float = 1.5,     ) -> None:
    """
    Plots each row of the specified column (containing JSON strings of lists) as a line on a wide plot.

    Args:
        df: DataFrame containing the column.
        column: Name of the column with JSON string of token counts.
        title: Plot title.
        figsize: Figure size (width, height).
        alpha: Line transparency.
        lw: Line width.
    """
    import json

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    for i, cell in enumerate(df[column]):
        try:
            values = json.loads(cell)
            values = values[::10]  # downsample
            plt.plot(values, label=f"Row {i}", alpha=alpha, lw=lw)
        except Exception as e:
            print(f"Row {i} could not be plotted: {e}")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Token Count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plot_path = results_dir / "token_count_distributions.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved token count distributions plot: {plot_path}")


def plot_hybrid_eval(
    dfs: dict[str, pd.DataFrame],
    results_dir: Path,
    light: bool = False ) -> None:
    """
    Plots all hybrid retrieval metric DataFrames in a single PNG grid.
    Each subplot shows all engines (bm25, vector, combined) for a metric@k, with questions on the x-axis.
    """
    palette = sns.color_palette(PU.get_palette(), n_colors=3)
    engines = ["bm25", "vector", "combined"]
    #metric_keys = sorted(dfs.keys()) 
    metric_keys = sorted(dfs.keys(), key=lambda x: (x.split('@')[0], int(x.split('@')[1])))
    n = len(metric_keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(8*ncols, 5*nrows), squeeze=False)

    for idx, metric_k in enumerate(metric_keys):
        row, col = divmod(idx, ncols)
        ax = axs[row][col]
        df = dfs[metric_k]
        # Sort questions by combined if available, else by question
        sort_col = "combined" if "combined" in df.columns else df.columns[0]
        df_sorted = df.sort_values(by=sort_col, ascending=False)
        x_pos = list(range(len(df_sorted)))
        for i, eng in enumerate(engines):
            if eng in df_sorted.columns:
                y = df_sorted[eng]
                markers = ['.', '+', '*', 'o', 'X']
                ax.scatter(x_pos, y, color=palette[i], s=60, marker=markers[i])  # Scatter for points, no label
                ax.plot(x_pos, y, label=PU.title_label(f"{eng}"), color=palette[i], linewidth=2, marker=markers[i], markersize=8)  # Plot for line and legend entry
        ax.set_title(metric_k, fontsize=20)
        ax.set_ylabel(metric_k.split('@')[0])
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_sorted["question"], rotation=45)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('black' if not light else 'white')

    # Remove empty subplots if any
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axs[row][col])

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=18, bbox_to_anchor=(0.5, 1.03), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "hybrid_retrieval_eval.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved hybrid retrieval plot: {plot_path}")


def prepare_hybrid_retrieval_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Prepare data for hybrid retrieval evaluation.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary where each key is a metric@k string (e.g., 'mrr@5', 'recall@13'),
            and each value is a DataFrame with rows as questions (zero-padded strings)
            and columns for each engine ('bm25', 'vector', 'combined').
            Each DataFrame contains the metric values for all questions and engines
            for that metric@k combination, suitable for direct plotting or analysis.
    """
    # Remove rows that have no deduplicate, no rerank, no is_valid_config
    #df = df[~(((df['deduplicate'] == False) | (df['is_valid_config'] == False) | (df['rerank'] == False)))]   noqa: E712 avoid equality comparison
    
    # Keep only columns that match the regex
    regex = re.compile(r"^question_(\d+)\.(bm25|vector|combined)_(recall|rr)@(\d+)$")
    df = df.loc[:, [col for col in df.columns if regex.match(col)]]

    # df.columns
    #     Index(['question_00.bm25_recall@13', 'question_00.bm25_recall@5',
    #         'question_00.bm25_recall@8', 'question_00.combined_recall@13',
    #         'question_00.combined_recall@5', 'question_00.combined_recall@8',
    #         'question_00.vector_recall@13', 'question_00.vector_recall@5',
    #         'question_00.vector_recall@8', 'question_01.bm25_recall@13',
    #         ...
    #         'question_18.vector_recall@8', 'question_19.bm25_recall@13',
    #         'question_19.bm25_recall@5', 'question_19.bm25_recall@8',
    #         'question_19.combined_recall@13', 'question_19.combined_recall@5',
    #         'question_19.combined_recall@8', 'question_19.vector_recall@13',
    #         'question_19.vector_recall@5', 'question_19.vector_recall@8'],
    #         dtype='object', length=180)

    # Reshape df to have rows per question number and columns for the metrics (bm25, vector, combined) and k (5, 8, 13)
    melted = df.melt(var_name="col", value_name="value")
    extracted = melted["col"].str.extract(r"^question_(\d+)\.(bm25|vector|combined)_(recall|rr)@(\d+)$")
    extracted.columns = ["question", "engine", "metric", "k"]
    melted = pd.concat([melted, extracted], axis=1)
    melted = melted.dropna(subset=["question", "engine", "metric", "k"])
    melted["question"] = melted["question"].astype(str).str.zfill(2)
    melted["metric_k"] = melted["metric"] + "@" + melted["k"]

    # Get all unique metric@k combinations
    metric_k_set = sorted(melted["metric_k"].unique())
    engines = ["bm25", "vector", "combined"]
    result: dict[str, pd.DataFrame] = {}
    for metric_k in metric_k_set:
        subset = melted[melted["metric_k"] == metric_k]
        # Pivot: rows=question, columns=engine, values=value
        pivot = subset.pivot_table(index="question", columns="engine", values="value")
        # Ensure all engines are present as columns
        for eng in engines:
            if eng not in pivot.columns:
                pivot[eng] = pd.NA
        pivot = pivot[engines]  # consistent column order
        pivot = pivot.sort_index(axis=0)
        pivot.index.name = "question"
        result[metric_k] = pivot.reset_index()
    return result


def main() -> None:
    args = PU.parse_args()
    if not args.wandb_project and not args.sweep_id and not args.files:
        PU.print_help()
        print("Please provide wandb_project and sweep_id to evaluate sweep or files to plot.")
        print("For plots, provide _post_eval_<temp>.xlsx and _human_eval.xlsx file in this order.")
        return
    if args.files:
        results_dir = Path(args.files[0]).parent
        xlsx_path = Path(args.files[0])
    else:
        results_dir = Path(f"results/{args.wandb_project}_sweep_{args.sweep_id}")
        xlsx_path = results_dir / f"sweep_{args.sweep_id}.xlsx"
    print(f"Using Excel file {LC.HI}{xlsx_path}{LC.RESET}")
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
    except Exception as e:
        print(f"Error reading Excel file in {results_dir}: {e}")
        return
    if df.empty:
        print(f"Empty Excel file in {results_dir}")
        return

    PU.init_plot_style(getattr(args, 'light', False))

    # plot hybrid retrieaval line plots
    hybrid_dfs = prepare_hybrid_retrieval_data(df)
    plot_hybrid_eval(hybrid_dfs, results_dir, getattr(args, 'light', False))

    tcdf = df[~df["index.token_count_distribution"].isna()].copy()
    plot_token_count_distributions(tcdf, results_dir)


if __name__ == "__main__":
    main()
