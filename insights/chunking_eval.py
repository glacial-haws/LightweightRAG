import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import style as mplstyle
from matplotlib.axes import Axes

from cbfk.log.log_config import LogConfig as LC
from insights.plot_utils import PlotUtils as PU


def load_sweeps(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:

    def load_excel(excel_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
            if df.empty:
                print(f"Empty Excel file in {LC.RED}{folder}{LC.RESET}")
                return None
            print(f"Loaded {LC.HI}{df.shape[0]}{LC.RESET} runs from {LC.HI}{excel_path.stem}{LC.RESET}")
            return df
        except Exception as e:
            print(f"Error reading Excel file in {LC.RED}{folder}{LC.RESET}: {e}")
            return None 

    df = pd.DataFrame()

    if len(args.files) > 0:
        folder = Path(args.files[0]).parent
        xlsx_files = args.files
        print(f"Output directory: {LC.HI}{folder}{LC.RESET}")
        df = pd.concat([load_excel(Path(f)) for f in xlsx_files], ignore_index=True)
    else:
        print("Please provide either results_dir or both sweep_id and wandb_project.")
        return None
    if not folder.exists() or not folder.is_dir():
        print(f"Results directory not found: {LC.RED}{folder}{LC.RESET}")
        return None

    if df.shape[0] == 0:
        # folder names provided, not excel files, collect sweeps in folders
        for folder in args.folders:
            folder = Path(folder)
            # Try to find the xlsx file (first .xlsx file)
            xlsx_files = list(folder.glob("sweep_*.xlsx"))
            if not xlsx_files:
                print(f"No xlsx file found in {folder}")
                continue
            # Take only the first xlsx file found
            excel_path = xlsx_files[0]
            df_single = load_excel(excel_path)
            if df_single is None:
                continue
            df = pd.concat([df, df_single], ignore_index=True)

    # Postprocess data
    df = df.copy()  # defragment
    df["splitter_type"] = df["splitter_type"]\
                                .apply(lambda x: x.lower())    # lowercase splitter_type
    df['time.index_mins'] = df['time.index_sec'] / 60          # add index_time_mins column from index_time sec
    df['using_cached_index'] = df['index.using_cached_index']\
                                .fillna(True)\
                                .infer_objects(copy=False)\
                                .astype(bool)                  # convert using_cached_index to bool, fill missing with True
    df = df.copy()  # defragment

    print("Evaluation dataset is ready with ", LC.HI, df.shape[0], LC.RESET, "runs, ", LC.HI, df.shape[1], LC.RESET, "columns.")
    print(f"Memory usage: {LC.HI}{df.memory_usage(deep=True).sum() / 10**6:.1f}{LC.RESET} MB")

    return df, folder


def clean_label(label: str) -> str:
    """Extract substring between '/' and '@' if both are present, else return label as-is."""
    if isinstance(label, str) and '/' in label and '@' in label:
        start = label.find('/') + 1
        end = label.find('@')
        if start < end:
            return label[start:end]
    return label


def strip_prefix(label: str) -> str:
    """Return the substring after the last '/' in label, or the label itself if '/' not present."""
    return label.rsplit('/', 1)[-1]


def __violinplot_param_vs_metric(
    ax: Axes,
    x: pd.Series,
    y: pd.Series,
    param: str,
    metric: str,
    *,    
    title_fontsize: int = 18,
    label_fontsize: int = 16, ) -> None:
    """
    Plot a single param-metric pair on the given axis, handling categorical/numeric/constant/missing data.
    """
    
    ax.set_facecolor('black')
    # Only plot if both x and y are not all NaN
    if x.isnull().all() or y.isnull().all():
        ax.set_title(f"{strip_prefix(metric)} (no data)")
        ax.axis('off')
        return
    # Categorical or constant
    if not np.issubdtype(x.dtype, np.number):
        x_labels = [clean_label(val) for val in x]
    else:
        x_labels = [clean_label(val) for val in x]
    hue_labels = x_labels
    sns.violinplot(x=x_labels, y=y, ax=ax, inner="point", cut=0, hue=hue_labels, palette=PU.get_palette(), legend=False)
    ax.tick_params(axis='x', rotation=45)
    ax.set_title(f"{strip_prefix(param)} vs {strip_prefix(metric)}", fontsize=title_fontsize)
    # ax.set_xlabel(strip_prefix(param), fontsize=label_fontsize)
    # ax.set_ylabel(strip_prefix(metric), fontsize=label_fontsize)


def plot_all_hyperparams_grid(
    df: pd.DataFrame,
    results_dir: Path,
    hyperparams: tuple[str, ...],
    metrics: tuple[str, ...], ) -> None:
    """
    Create a single figure with a row for each hyperparameter, each row containing subplots (one per metric),
    and a line for each hyperparameter value. Each subplot is a metric vs. hyperparameter value plot.
    """
    n_params = len(hyperparams)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_params, n_metrics, figsize=(8 * n_metrics, 5 * n_params), squeeze=False)
    fig.patch.set_facecolor('black')
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    for row, param in enumerate(hyperparams):
        if param not in df.columns:
            for col in range(n_metrics):
                axes[row, col].set_visible(False)
            continue
        x = df[param]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            if metric not in df.columns:
                ax.set_title(f"{strip_prefix(metric)} (not found)")
                ax.axis('off')
                continue
            y = df[metric]
            __violinplot_param_vs_metric(
                ax, x, y, param, metric,
                title_fontsize=18, label_fontsize=16
            )
    plt.tight_layout()
    plot_path = results_dir / "all_hyperparams_grid.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved all-hyperparams grid plot: {LC.HI}{plot_path}{LC.RESET}")


def plot_accuracy_vs_chunk_size_by_embedding_model(
    df: pd.DataFrame,
    results_dir: Path, ) -> None:
    """
    Plot accuracy vs. chunk_size with separate lines for each embedding_model.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    mplstyle.use('dark_background')
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 24,
    })
    # Filter out rows with missing required columns
    if not {'chunk_size', 'graded_accuracy', 'embedding_model'}.issubset(df.columns):
        print("Required columns for accuracy vs chunk_size plot not found.")
        return
    plot_df = df.dropna(subset=['chunk_size', 'graded_accuracy', 'embedding_model'])
    if plot_df.empty:
        print("No data available for accuracy vs chunk_size plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('black')
    # Ensure chunk_size is numeric for plotting
    plot_df = plot_df.copy()
    plot_df['chunk_size'] = pd.to_numeric(plot_df['chunk_size'], errors='coerce')
    plot_df = plot_df.dropna(subset=['chunk_size'])
    # Sort by chunk_size for line plotting
    plot_df = plot_df.sort_values(by='chunk_size')
    # Use Oranges palette for embedding_model lines
    embedding_models = list(plot_df['embedding_model'].dropna().unique())
    palette = sns.color_palette(PU.get_palette(), n_colors=len(embedding_models))
    for idx, (embedding_model, group) in enumerate(plot_df.groupby('embedding_model')):
        agg = group.groupby('chunk_size')['graded_accuracy'].agg(['mean', 'std']).reset_index()
        color = palette[idx]
        ax.plot(agg['chunk_size'], agg['mean'], label=embedding_model, color=color, marker='o', linewidth=2)
        ax.scatter(agg['chunk_size'], agg['mean'], color=color, s=60)
        ax.fill_between(
            agg['chunk_size'],
            agg['mean'] - agg['std'],
            agg['mean'] + agg['std'],
            alpha=0.2,
            color=color
        )
    ax.set_xlabel('chunk_size', fontsize=20)
    ax.set_ylabel('graded_accuracy', fontsize=20)
    ax.set_title('Graded Accuracy vs. Chunk Size by Embedding Model', fontsize=22)
    ax.legend(title='embedding_model', fontsize=16, title_fontsize=16)
    ax.set_facecolor('black')
    plt.tight_layout()
    plot_path = results_dir / "graded_accuracy_vs_chunk_size_by_embedding_model.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved graded accuracy vs chunk_size plot: {LC.HI}{plot_path}{LC.RESET}")


def plot_score_over_questions(
    df: pd.DataFrame,
    results_dir: Path,
    hyperparameter: str,
    scores: list[str], ) -> None:
    """
    Plot scores over all questions as scatter plots with lines, one subplot per score, one line for each hyperparameter value.
    Args:
        df: DataFrame with columns like 'question_01/graded_accuracy', 'question_02/graded_accuracy', etc.
        results_dir: Directory to save the plot
        hyperparameter: Column name in df to group by (e.g., 'llm_model').
        scores: List of score suffixes to look for (e.g., ['graded_accuracy', 'recall']).
    """
    n_scores = len(scores)
    fig, axes = plt.subplots(1, n_scores, figsize=(max(8, n_scores*7), 6), squeeze=False)
    palette = sns.color_palette(PU.get_palette(), n_colors=df[hyperparameter].nunique())
    for col_idx, score in enumerate(scores):
        pattern = re.compile(rf"question_(\d{{2}})\..*{re.escape(score)}.*")
        question_cols = [col for col in df.columns if pattern.match(col)]
        if not question_cols:
            print(f"No columns found for score '{score}' with question pattern.")
            continue
        # Sort columns by average score (ascending)
        avg_scores = df[question_cols].mean(axis=0, skipna=True)
        question_cols_sorted = avg_scores.sort_values().index.tolist()
        question_indices = list(range(len(question_cols_sorted)))
        ax = axes[0, col_idx]
        for idx, (hp_value, group) in enumerate(df.groupby(hyperparameter)):
            y_mean = group[question_cols_sorted].mean(axis=0, skipna=True).values
            y_std = group[question_cols_sorted].std(axis=0, skipna=True).values
            ax.plot(question_indices, y_mean, label=str(hp_value), color=palette[idx], marker='o', linewidth=2)
            ax.scatter(question_indices, y_mean, color=palette[idx], s=60)
            # Draw band if >1 row per (hyperparameter, question)
            if len(group) > 1:
                ax.fill_between(question_indices, y_mean - y_std, y_mean + y_std, alpha=0.2, color=palette[idx])
        #ax.set_xticks(question_indices)
        #ax.set_xticklabels([col.split('/')[0] for col in question_cols_sorted], rotation=45)
        ax.set_xlabel('Question')

        ax.set_title(f"{score.replace('_', ' ').capitalize()} by {hyperparameter}", pad=18)
        ax.set_facecolor('black')
        ax.set_ylim(0.0, 1.0)
    # Place a single legend below all subplots
    handles, labels = axes[0,0].get_legend_handles_labels()
    if labels:
        fig.legend(
            handles,
            labels,
            title=hyperparameter,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.08),
            ncol=max(1, min(len(labels), 4)),
            frameon=False
        )
    else:
        print(f"{LC.AMBER}[plot_score_over_questions] No data was plotted, skipping legend{LC.RESET}")
        print(f"hyperparameter: {LC.HI}{hyperparameter}{LC.RESET}")
        print(f"scores: {LC.HI}{', '.join(scores)}{LC.RESET}")
        #print(f"df.columns: {', '.join(df.columns)}")
    plt.subplots_adjust(bottom=0.22)
    plot_path = results_dir / f"score_over_questions_{hyperparameter}_{'_'.join(scores)}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved score-over-questions plot: {LC.HI}{plot_path}{LC.RESET}")


def plot_score_over_param(
    df: pd.DataFrame,
    results_dir: Path,
    series_param: str,
    x_axis_param: str,
    scores: list[str],
    light: bool = False , ) -> None:
    """
    Plot scores over ascending values of param as scatter plots with lines, one subplot per score, one line for each hyperparameter value.
    Args:
        df: DataFrame with columns including param and hyperparameter.
        results_dir: Directory to save the plot.
        hyperparameter: Column name in df to group by (e.g., 'llm_model').
        param: Column name in df to plot on the x-axis (e.g., 'chunk_size').
        scores: List of score column names to plot (e.g., ['graded_accuracy', 'recall']).
    """
    n_scores = len(scores)
    fig, axes = plt.subplots(1, n_scores, figsize=(max(8, n_scores*7), 6), squeeze=False)
    palette = sns.color_palette(PU.get_palette(), n_colors=df[series_param].nunique())

    x_values = sorted(df[x_axis_param].dropna().unique())
    x_labels = [str(x) for x in x_values]
    title_series_param = series_param.split(".")[-1]
    fig.suptitle(f"{PU.title_label(title_series_param)} and {PU.title_label(x_axis_param)}", fontsize=32, y=1.15)
    for col_idx, score in enumerate(scores):
        ax = axes[0, col_idx]
        all_y_values = []  # Collect all individual y values for this subplot
        plot_score_name = score
        # First, collect all values for this score (in seconds)
        for _, (_, group) in enumerate(df.groupby(series_param)):
            for x in x_values:
                vals = group.loc[group[x_axis_param] == x, score]
                all_y_values.extend([v for v in vals if pd.notna(v)])
        # Decide on conversion
        for idx, (hp_value, group) in enumerate(df.groupby(series_param)):
            y_means = []
            y_stds = []
            for x in x_values:
                vals = group.loc[group[x_axis_param] == x, score]
                y_means.append(vals.mean(skipna=True))
                y_stds.append(vals.std(skipna=True))
            ax.plot(x_labels, y_means, label=str(hp_value), color=palette[idx], marker='o', linewidth=2)
            ax.scatter(x_labels, y_means, color=palette[idx], s=60)
            if len(group) > 1:
                y_means_arr = np.array(y_means)
                y_stds_arr = np.array(y_stds)
                ax.fill_between(x_labels, y_means_arr - y_stds_arr, y_means_arr + y_stds_arr, alpha=0.2, color=palette[idx])

        # For y-axis scaling, recompute all_y_values if converted
        y_scale_values = [v for v in all_y_values]
        ax.set_title(f"{PU.title_label(plot_score_name)}", fontsize=18, pad=18)

        # Determine y-axis upper limit from all individual values
        max_y = max(y_scale_values) if y_scale_values else 1.0
        if max_y > 1.0:
            ax.set_ylim(0.0, max_y * 1.05)
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels, rotation=33, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        #ax.set_xlabel(fix_label(x_axis_param), fontsize=18, labelpad=12)
        #ax.set_ylabel(fix_label(plot_score_name), fontsize=18, labelpad=12)

    handles, labels = axes[0,0].get_legend_handles_labels()
    #fig.legend(handles, labels, title=series_param, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=min(len(labels), 4), frameon=False)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=min(len(labels), 4), frameon=False, fontsize=18, title_fontsize=18)
    plt.subplots_adjust(top=0.85)
    plot_path = results_dir / f"score_over_{x_axis_param}_{title_series_param}_{'_'.join(scores)}.png"

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"score-over-param: {LC.HI}{plot_path}{LC.RESET}")


def write_scientific_table_markdown(
    df: pd.DataFrame,
    series_params: list[str],
    scores: list[str],
    md_path: Path,
    fix_label: callable = lambda x: x, ) -> None:
    """
    Write a scientific-style table as a markdown file, summarizing mean ± std for each unique combination of series_params, for each score.
    The columns are: all series_params (in order), then the scores.
    """
    # Get all unique combinations of series_params
    unique_combos = df.drop_duplicates(subset=series_params)[series_params]
    table_rows = []
    for _, combo in unique_combos.iterrows():
        mask = (df[series_params] == combo.values).all(axis=1)
        row = [str(combo[param]) for param in series_params]
        for score in scores:
            vals = df.loc[mask, score]
            mean = vals.mean(skipna=True)
            std = vals.std(skipna=True)
            if pd.isna(mean):
                cell = "-"
            else:
                cell = f"{mean:.2g} ± {std:.2g}" if not pd.isna(std) else f"{mean:.2g}"
            row.append(cell)
        table_rows.append(row)
    table_cols = [PU.title_label(p) for p in series_params] + [PU.title_label(s) for s in scores]
    # Markdown table header
    md_lines = ["| " + " | ".join(table_cols) + " |", "|" + "|".join(["---"] * len(table_cols)) + "|"]
    for row in table_rows:
        md_lines.append("| " + " | ".join(row) + " |")
    md_content = "\n".join(md_lines)
    md_path.write_text(md_content)
    print(f"Wrote markdown table to {LC.HI}{md_path}{LC.RESET}")



def pareto_front(df: pd.DataFrame, results_dir: Path, prefix: str, light: bool) -> None:

    def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)
                is_efficient[i] = True
        return is_efficient

    def plot_pareto(df: pd.DataFrame, pareto_mask: np.ndarray, results_dir: Path, light: bool) -> None:
        pareto_df = df[pareto_mask]
        non_pareto_df = df[~pareto_mask]

        fig, axs = plt.subplots(1, 3, figsize=(21, 6), squeeze = False)
        fig.suptitle("Pareto Front Analysis", fontsize=32)

        def plot_one(ax, x, y, x_label, y_label, light: bool):
            color = 'gray' if light else 'lightgray'
            ax.scatter(non_pareto_df[x], non_pareto_df[y], color=color, alpha=0.4, label='Other Models')
            ax.scatter(pareto_df[x], pareto_df[y], color='orange', edgecolor='black', s=80, label='Pareto-optimal')
            for _, row in pareto_df.iterrows():
                ax.text(row[x], row[y], row["index.embedding_model"], fontsize=10, color=color, ha='left', va='bottom')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    

        plot_one(axs[0,0], 'time.index_mins', 'recall', 'Index Time (s)', 'Recall', light)
        plot_one(axs[0,1], 'time.query_sec', 'graded_accuracy', 'Query Time (s)', 'Graded Accuracy', light)
        plot_one(axs[0,2], 'time.query_sec', 'mrr', 'Query Time (s)', 'MRR', light)

        handles, labels = axs[0,0].get_legend_handles_labels()
        # Place legend between suptitle and plots
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=22, bbox_to_anchor=(0.5, 0.86), borderaxespad=0.5, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plot_path = results_dir / f"{prefix}_pareto_front.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"pareto front: {LC.HI}{plot_path}{LC.RESET}")
    
    required_cols = ['embedding_model', 'splitter_type', 'chunk_size', 'recall', 'mrr', 'graded_accuracy', 'index.indexing_time_sec', 'time.query_sec']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data = df[required_cols].copy()
    data['-index.indexing_time_sec'] = -data['index.indexing_time_sec']
    data['-time.query_sec'] = -data['time.query_sec']
    pareto_metrics = ['recall', 'mrr', 'graded_accuracy', '-index.indexing_time_sec', '-time.query_sec']
    values = data[pareto_metrics].values
    pareto_mask = is_pareto_efficient(values)

    pareto_df = df[pareto_mask].copy().sort_values(by=["recall", "graded_accuracy", "mrr"], ascending=False)

    print("\nPareto-optimal models:")
    print(pareto_df[['embedding_model', 'splitter_type', 'chunk_size', 'recall', 'mrr', 'graded_accuracy', 'time.index_sec', 'time.query_sec']])

    if results_dir:
        # Only write required columns to pareto_front.csv
        pareto_df[required_cols].to_csv(results_dir / "pareto_front.csv", index=False)
        pareto_df[required_cols].to_excel(results_dir / "pareto_front.xlsx", index=False)
        write_scientific_table_markdown(
            df=pareto_df,
            series_params=["index.embedding_model", "splitter_type", "chunk_size"],
            scores=["recall", "mrr", "graded_accuracy", "time.index_sec", "time.query_sec"],
            md_path=results_dir / "pareto_front.md",
        )
        print(f"\nSaved Pareto front to: {results_dir / 'pareto_front.csv'} and {results_dir / 'pareto_front.xlsx'}")

    plot_pareto(df, pareto_mask, results_dir, light)


def main() -> None:

    args = PU.parse_args()
    PU.init_plot_style(args.light)
    df, results_dir = load_sweeps(args)
    if df is None: return

    # remove rows with false or NA is_valid_config
    df_plot = df[df['is_valid_config'].fillna(False)]  

    # hyperparams = ( "chunk_size","chunk_overlap_pct", "chunk_overlap", "splitter_type", "index.embedding_model", 
    #               "similarity_top_k", "llm_model", "llm_temperature", "llm_max_tokens" )
    # plot_all_hyperparams_grid(df, results_dir, 
    #     hyperparams, ("graded_accuracy", "recall", ))
    # plot_accuracy_vs_chunk_size_by_embedding_model(df, results_dir)
    # plot_score_over_questions(df, results_dir, "llm_model", ["recall", "graded_accuracy"])


    # create splitter table
    splitter_df = df_plot[df_plot['chunk_size'] == 256].copy()
    splitter_df = splitter_df.groupby(["index.embedding_model", "splitter_type"]).agg({
        "graded_accuracy": "mean",
        "recall": "mean",
        "mrr": "mean",
        "time.index_mins": "mean",
        "time.query_sec": "mean",
    }).reset_index()
    #plot_score_over_param(splitter_df, results_dir, "index.embedding_model", "splitter_type", ["graded_accuracy", "recall", "mrr", "time.index_mins", "time.query_sec"], light=args.light)
    write_scientific_table_markdown(
        df=splitter_df,
        #series_params=["index.embedding_model", "splitter_type"],
        series_params=["splitter_type"],
        scores=["graded_accuracy", "recall", "mrr", "time.index_mins", "time.query_sec"],
        md_path=results_dir / "splitter_table.md",
    )

    # remove runs using cached index and runs not using SENTENCE splitter
    df_plot = df_plot[~df_plot['using_cached_index']]       
    df_plot = df_plot[df_plot['splitter_type'] == 'sentence']

    # plot embedding model vs chunk size
    scores = ["recall", "mrr", "graded_accuracy", "time.index_mins", "time.query_sec"]
    plot_score_over_param(df_plot, results_dir, "index.embedding_model", "chunk_size", scores, light=args.light)
    write_scientific_table_markdown(
        df=df_plot,
        series_params=["index.embedding_model", "splitter_type"],
        scores=scores,
        md_path=results_dir / "score_over_chunk_size_embedding_model_splitter_type.md",
    )
    pareto_front(df_plot, results_dir, "judge", args.light)


    # plot LLM vs chunk size
    scores = ["recall", "mrr", "graded_accuracy", "time.prompt_sec"]
    plot_score_over_param(df_plot, results_dir, "llm_model", "chunk_size", scores, light=args.light)
    write_scientific_table_markdown(
        df=df_plot,
        series_params=["llm_model", "chunk_size"],
        scores=scores,
        md_path=results_dir / "score_over_chunk_size_llm_model.md",
    )
    pareto_front(df_plot, results_dir, "LLM",args.light)




if __name__ == "__main__":
    main()

