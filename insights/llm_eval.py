import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from cbfk.log.log_config import LogConfig as LC
from insights.plot_utils import PlotUtils as PU


def load_excel(excel_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(excel_path, engine='openpyxl')
        if df.empty:
            print(f"Empty Excel file {LC.RED}{excel_path}{LC.RESET}")
            return None
        print(f"Loaded {LC.HI}{df.shape[0]}{LC.RESET} runs from {LC.HI}{excel_path.stem}{LC.RESET}")
        return df
    except Exception as e:
        print(f"Error reading Excel file {LC.RED}{excel_path}{LC.RESET}: {e}")
        return None 

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
            #std = vals.std(skipna=True)
            if pd.isna(mean):
                cell = "-"
            else:
                #cell = f"{mean:.3g} ± {std:.3g}" if not pd.isna(std) else f"{mean:.3g}"
                cell = f"{mean:.3g}"
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


def write_scientific_table_pandoc(
    df: pd.DataFrame,
    series_params: list[str],
    scores: list[str],
    md_path: Path, ) -> None:
    """
    Write a Pandoc-style multi-header markdown table where the first column is llm_model, and for each embedding_model, there are subcolumns for each score.
    Each row corresponds to an llm_model. Each (embedding_model, llm_model) pair summarizes mean ± std for each score.
    """

    # Defensive: ensure required columns
    if not ("embedding_model" in df.columns and "llm_model" in df.columns):
        raise ValueError("DataFrame must contain 'embedding_model' and 'llm_model' columns.")

    # Unique values
    embedding_models = [strip_prefix(em) for em in df["embedding_model"].dropna().unique()]
    llm_models = [strip_prefix(lm) for lm in df["llm_model"].dropna().unique()]

    # Map for quick lookup
    df = df.copy()
    df.loc[:, "embedding_model"] = df["embedding_model"].map(strip_prefix)
    df.loc[:, "llm_model"] = df["llm_model"].map(strip_prefix)

    # Build a lookup: (llm_model, embedding_model) -> dict of {score: (mean, std)}
    lookup: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for lm in llm_models:
        for em in embedding_models:
            mask = (df["llm_model"] == lm) & (df["embedding_model"] == em)
            vals = df.loc[mask]
            score_stats: dict[str, str] = {}
            for score in scores:
                score_vals = vals[score]
                mean = score_vals.mean(skipna=True)
                std = score_vals.std(skipna=True)
                if pd.isna(mean):
                    cell = "-"
                else:
                    cell = f"{mean:.3g} ± {std:.3g}" if not pd.isna(std) and std > 0 else f"{mean:.3g}"
                score_stats[score] = cell
            lookup[(lm, em)] = score_stats

    # Build Pandoc multi-header
    # First header: blank, then embedding_model (spans len(scores)), ...
    header_1 = ["LLM Model"]
    for em in embedding_models:
        header_1.extend([em] + [""] * (len(scores) - 1))
    # Second header: blank, then scores repeated for each embedding_model
    header_2 = [" "]
    for _ in embedding_models:
        header_2.extend(scores)

    # Markdown: Pandoc multi-header (just repeat as two header rows)
    md_lines = [
        "+---------------" + "+---------------" * (len(embedding_models) * len(scores)) + "+",
        "|               " + "|               " * (len(embedding_models) * len(scores)) + "|",
        "| " + " | ".join(header_1) + " |",
        "| " + " | ".join(header_2) + " |",
        "+=" + "=---------------" * (len(embedding_models) * len(scores)) + "+",
    ]

    # Data rows
    for lm in llm_models:
        row = [lm]
        for em in embedding_models:
            stats = lookup.get((lm, em), {})
            for score in scores:
                row.append(stats.get(score, "-"))
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("+---------------" + "+---------------" * (len(embedding_models) * len(scores)) + "+")
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

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), squeeze = False)

        def plot_one(ax, x, y, x_label, y_label, light: bool):
            color = 'gray' if light else 'lightgray'
            ax.scatter(non_pareto_df[x], non_pareto_df[y], color=color, alpha=0.4, label='Other Models')
            ax.scatter(pareto_df[x], pareto_df[y], color='orange', edgecolor='black', s=80, label='Pareto-optimal')
            for _, row in pareto_df.iterrows():
                ax.text(row[x], row[y], row["display_name"], fontsize=10, color=color, ha='left', va='bottom')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    
        plot_one(axs[0,0], 'time.prompt_sec', 'graded_accuracy', 'Prompt Time (s)', 'Graded Accuracy', light)
        #plot_one(axs[0,1], 'time.total_sec', 'graded_accuracy', 'Total Time (s)', 'Graded Accuracy', light)

        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=22, bbox_to_anchor=(0.5, 0.93), borderaxespad=0.5, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.82])
        plot_path = results_dir / f"{prefix}_pareto_front.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        print(f"pareto front: {LC.HI}{plot_path}{LC.RESET}")
    

    required_cols = ['display_name', 'graded_accuracy', 'time.prompt_sec', 'time.total_sec']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    data = df[required_cols].copy()
    data['-time.total_sec'] = -data['time.total_sec']
    pareto_metrics = ['graded_accuracy', '-time.total_sec']
    values = data[pareto_metrics].values
    pareto_mask = is_pareto_efficient(values)

    pareto_df = df[pareto_mask].copy().sort_values(by=["graded_accuracy"], ascending=False)

    print("\nPareto-optimal models (average times per prompt):")
    print(pareto_df[['display_name', 'graded_accuracy', 'time.prompt_sec', 'time.total_sec']])

    if results_dir:
        # Only write required columns to pareto_front.csv
        pareto_df[required_cols].to_csv(results_dir / "pareto_front.csv", index=False)
        pareto_df[required_cols].to_excel(results_dir / "pareto_front.xlsx", index=False)
        write_scientific_table_markdown(
            df=pareto_df,
            series_params=["embedding_model", "llm_model"],
            scores=["graded_accuracy", "time.total_sec"],
            md_path=results_dir / "llm_pareto_front.md",
        )
        print(f"\nSaved Pareto front to: {results_dir / 'llm_pareto_front.csv'} and {results_dir / 'llm_pareto_front.xlsx'}")

    plot_pareto(df, pareto_mask, results_dir, light)


def main() -> None:

    args = PU.parse_args()
    PU.init_plot_style(args.light)
    df = load_excel(Path(args.files[0]))
    if df is None: return
    results_dir = Path(args.files[0]).parent

    # remove rows with false or NA is_valid_config
    df_plot = df[df['is_valid_config'].fillna(False)]  

    # remove runs using cached index and runs not using SENTENCE splitter
    df_rewritten = df_plot[df_plot['query_rewrite']]   

    # convert to time per prompt for index query and llm response   
    number_of_prompts = 20
    df_rewritten['time.prompt_sec'] = df_rewritten['time.prompt_sec'] / number_of_prompts    
    df_rewritten['time.query_sec'] = df_rewritten['time.query_sec'] / number_of_prompts    
    df_rewritten['time.total_sec'] = df_rewritten['time.prompt_sec'] + df_rewritten['time.query_sec']

    df_rewritten['display_name'] = df_rewritten['llm_model'] + "/" + df_rewritten['embedding_model'].apply(lambda x: x[:5] if len(x) > 5 else x)

    write_scientific_table_pandoc(
        df=df_rewritten,
        series_params=[ "embedding_model", "llm_model"],
        scores=["graded_accuracy", "recall", "mrr", "time.prompt_sec", "time.total_sec"],
        md_path=results_dir / "llm_table.md",
    )

    write_scientific_table_markdown(
        df=df_rewritten,
        series_params=[ "embedding_model"],
        scores=[ "recall@5", "recall@8",  "mrr@5", "mrr@8","time.query_sec"],
        md_path=results_dir / "embedding_time_table.md",
    )

    pareto_front(df_rewritten, results_dir, "llm", args.light)



if __name__ == "__main__":
    main()

