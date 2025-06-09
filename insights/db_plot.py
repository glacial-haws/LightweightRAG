import logging
import sqlite3
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import PlotUtils as PU

from cbfk.log.log_config import LogConfig as LC

LC.setup_logging()
logger = logging.getLogger(__name__)


db_path = "results/_db/runs.db"
output_dir = Path("results/_db/plots")
output_dir.mkdir(parents=True, exist_ok=True)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(val):
        if isinstance(val, bool):
            return str(val)  # Convert boolean to 'True' or 'False' string
        try:
            fval = float(val)
            return f"{fval:.3f}"
        except (ValueError, TypeError):
            return str(val)

    headers = df.columns.tolist()
    aligns = [":" + "---".rjust(3) + ":" if pd.api.types.is_numeric_dtype(df[col]) else ":---" for col in headers]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(aligns) + "|"]
    for _, row in df.iterrows():
        values = [fmt(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines))
    print(f"Saved table: {LC.HI}{path}{LC.RESET}")


def load(db_path: str, query: str) -> pd.DataFrame:
    df = pd.read_sql_query(query, sqlite3.connect(db_path))
    if df.empty:
        logger.error(f"No data for query: {query} in {db_path}")
    return df


def plot_chunk_size(
    df: pd.DataFrame,
    out_path: Path,
    metric_cols: list[str],
    group_by: str,
    artifact_name: str  ) -> None:
    """Plot all metrics as subplots in a single row and save as one PNG."""
    PU.init_plot_style(light=True, large=True)
    palette = sns.color_palette(PU.get_palette(), n_colors=df[group_by].nunique())
    n_metrics = len(metric_cols)
    # Calculate rows and columns for a more square layout (2x2 for 4 metrics)
    n_rows = 2
    n_cols = (n_metrics + 1) // 2  # Ceiling division to ensure all metrics fit
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows), squeeze=False)
    chunk_labels = [256, 384, 512, 768, 1024]
    chunk_label_str = [str(cs) for cs in chunk_labels]
    chunk_label_map = {v: i for i, v in enumerate(chunk_labels)}

    for col_idx, metric in enumerate(metric_cols):
        if metric not in df.columns:
            continue
        row_idx = col_idx // n_cols
        col = col_idx % n_cols
        ax = axes[row_idx, col]
        for idx, (model, group) in enumerate(df.groupby(group_by)):
            group = group.dropna(subset=['chunk_size', metric])
            group['chunk_size'] = pd.to_numeric(group['chunk_size'], errors='coerce')
            group[metric] = pd.to_numeric(group[metric], errors='coerce')
            agg = group.groupby('chunk_size', as_index=False)[metric].mean()
            agg = agg[agg['chunk_size'].isin(chunk_labels)]
            agg = agg.sort_values(by='chunk_size')
            x_categorical = [chunk_label_map.get(cs, None) for cs in agg['chunk_size']]
            color = palette[idx]
            ax.plot(x_categorical, agg[metric], label=model, marker='o', color=color)
            ax.scatter(x_categorical, agg[metric], color=color, s=60)
        ax.set_xticks(range(len(chunk_labels)))
        ax.set_xticklabels(chunk_label_str)
        ax.set_title(f"{PU.title_label(metric)}", fontsize=28)
        ax.set_xlabel("Chunk Size", fontsize=26)
        
        # Dynamically set y-axis for time fields
        if metric.endswith('_sec') or 'time_sec' in metric:
            # Compute min/max ignoring NaNs and inf
            vals = pd.to_numeric(df[metric], errors='coerce').dropna()
            if not vals.empty:
                ymin = vals.min()
                ymax = vals.max()
                pad = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
                ax.set_ylim(max(0, ymin - pad), ymax + pad)
                ax.yaxis.set_label_position('left')
            else:
                ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])
            ax.yaxis.set_label_position('left')
        if col_idx == n_metrics - 1:
            ax.legend(title=None)
    fig.tight_layout()
    png_path = out_path / f"{artifact_name}_all_metrics.png"
    plt.savefig(png_path)
    plt.close(fig)
    print(f"Saved all-metrics scatterplot {LC.HI}{png_path}{LC.RESET}")
    md_path = out_path / f"{artifact_name}.md"
    save_markdown_table(df, md_path)


def plot_mrr_comparison(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """
    Plots MRR comparison for different embedding models and @k values.

    Args:
        df: Pandas DataFrame with columns:
            - question_nr
            - embedding_model
            - vector_rr_5_no_aug, vector_rr_5_aug
            - vector_rr_8_no_aug, vector_rr_8_aug
            - vector_rr_13_no_aug, vector_rr_13_aug
        output_path: Optional Path object to save the figure. If None, shows plot.
    """
    # PU, LC, pd, plt, np, Path are expected to be globally imported.

    PU.init_plot_style(light=True, large=True)
    
    embedding_models = df['embedding_model'].unique()
    k_values = [5, 8, 13]
    
    n_models = len(embedding_models)
    n_k = len(k_values)
    
    fig, axes = plt.subplots(n_models, n_k, figsize=(6 * n_k, 5 * n_models), squeeze=False)
    
    # Correctly get color palette using seaborn
    try:
        palette_name = PU.get_palette() # Assuming this returns a string like "Orange" or a list of colors
        if isinstance(palette_name, str):
            actual_colors = sns.color_palette(palette_name, n_colors=2)
        elif isinstance(palette_name, list) and len(palette_name) >= 2: # If PU.get_palette() already returns a list of colors
            actual_colors = palette_name
        else: # Fallback if palette_name is not a string or a sufficient list
            logger.warning(f"PU.get_palette() returned an unexpected value: {palette_name}. Using default fallback colors.")
            actual_colors = sns.color_palette(n_colors=2) # Seaborn default
            
        color_no_aug = actual_colors[0]
        color_aug = actual_colors[1]
    except Exception as e:
        logger.warning(f"Failed to get colors from PU.get_palette() or seaborn: {e}. Using default fallback colors.")
        # Fallback colors in case of any error
        color_no_aug = 'blue' 
        color_aug = 'green'

    for i, model in enumerate(embedding_models):
        model_df_orig = df[df['embedding_model'] == model]
        model_df = model_df_orig.sort_values('question_nr').copy()
        
        model_df['question_nr'] = pd.to_numeric(model_df['question_nr'], errors='coerce')

        for j, k_val in enumerate(k_values):
            ax = axes[i, j]
            
            col_no_aug = f'vector_rr_{k_val}_no_aug'
            col_aug = f'vector_rr_{k_val}_aug'
            
            if col_no_aug in model_df.columns and col_aug in model_df.columns:
                # Iterate through each question to plot points and connecting lines
                for _idx, row in model_df.iterrows(): # _idx to avoid lint if not used
                    q_nr = row['question_nr']
                    mrr_no_aug_val = pd.to_numeric(row[col_no_aug], errors='coerce')
                    mrr_aug_val = pd.to_numeric(row[col_aug], errors='coerce')

                    if pd.notna(q_nr) and pd.notna(mrr_no_aug_val) and pd.notna(mrr_aug_val):
                        # Plot points
                        ax.scatter([q_nr], [mrr_no_aug_val], color=color_no_aug, marker='o', s=30, zorder=3)
                        ax.scatter([q_nr], [mrr_aug_val], color=color_aug, marker='x', s=30, zorder=3)
                        # Plot connecting line
                        ax.plot([q_nr, q_nr], [mrr_no_aug_val, mrr_aug_val], color='grey', linestyle='-', linewidth=0.8, zorder=2)
            
            title = f"{model} - MRR@{k_val}"
            ax.set_title(PU.title_label(title), fontsize=15) 
            ax.set_xlabel("Question Number", fontsize=12) 
            ax.set_ylabel("MRR", fontsize=12) 
            
            # Create custom legend
            legend_handles = [
                mlines.Line2D([], [], color=color_no_aug, marker='o', linestyle='None', markersize=7, label=f'No Aug MRR@{k_val}'),
                mlines.Line2D([], [], color=color_aug, marker='x', linestyle='None', markersize=7, label=f'Aug MRR@{k_val}')
            ]
            if legend_handles: # Only add legend if handles were created (i.e., if there was data)
                ax.legend(handles=legend_handles, fontsize=9)
            
            ax.tick_params(axis='both', which='major', labelsize=9) 

            ax.set_ylim(0, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            
            # Adjust x-axis ticks based on actual question numbers present
            unique_x_qnr_for_ticks = model_df['question_nr'].dropna().unique()
            if len(unique_x_qnr_for_ticks) > 0:
                if len(unique_x_qnr_for_ticks) > 15:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7, integer=True))
                else:
                    ax.set_xticks(sorted(list(unique_x_qnr_for_ticks)))
                    if len(unique_x_qnr_for_ticks) > 5: # Rotate if many ticks and close together
                        ax.tick_params(axis='x', rotation=45)
            # If no valid x data, x-ticks will be default/empty

    fig.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.0)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        try:
            print(f"Saved MRR comparison plot to {LC.HI}{output_path}{LC.RESET}")
        except NameError: 
            print(f"Saved MRR comparison plot to {output_path}")
    else:
        plt.show()


def find_pareto_frontier(df: pd.DataFrame, time_col: str, metric_col: str) -> pd.DataFrame:
    """Find the Pareto frontier points in a dataframe.
    
    A point is on the Pareto frontier if no other point has both better time (lower)
    and better metric value (higher). Adds a 'is_pareto' column to the dataframe.
    
    Args:
        df: DataFrame with time and metric columns
        time_col: Name of the time column (lower is better)
        metric_col: Name of the metric column (higher is better)
        
    Returns:
        DataFrame with an additional 'is_pareto' boolean column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Convert columns to numeric if they aren't already
    result_df[time_col] = pd.to_numeric(result_df[time_col], errors='coerce')
    result_df[metric_col] = pd.to_numeric(result_df[metric_col], errors='coerce')
    
    # Drop rows with NaN values in relevant columns
    result_df = result_df.dropna(subset=[time_col, metric_col, 'identifier'])
    
    # Initialize all points as being on the Pareto frontier
    result_df['is_pareto'] = True
    
    # For each point, check if it's dominated by any other point
    for i, row_i in result_df.iterrows():
        # Skip if already marked as not on Pareto frontier
        if not result_df.at[i, 'is_pareto']:
            continue
            
        for j, row_j in result_df.iterrows():
            if i != j:
                # If j dominates i (j has better or equal time AND better metric)
                if (row_j[time_col] <= row_i[time_col] and 
                    row_j[metric_col] > row_i[metric_col]):
                    result_df.at[i, 'is_pareto'] = False
                    break
    
    return result_df


def plot_vs_time(df: pd.DataFrame, metric_cols: list[str], time_col: str, outpath: Path) -> None:
    """Plot a scatter plot of metric vs time with annotations for each point.
    
    Points on the Pareto frontier are highlighted in orange with labels.
    All other points are shown in gray without labels.
    """
    PU.init_plot_style(light=True, large=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    work_df = df.copy()
    
    work_df[time_col] = pd.to_numeric(work_df[time_col], errors='coerce')
    work_df[metric_cols[0]] = pd.to_numeric(work_df[metric_cols[0]], errors='coerce')
    
    plot_df = find_pareto_frontier(work_df, time_col, metric_cols[0])
    
    if plot_df.empty:
        print("Warning: No valid data points to plot after filtering NaNs")
        plt.close(fig)
        return
    
    # Plot all points in gray first
    ax.scatter(
        plot_df[time_col],
        plot_df[metric_cols[0]],
        s=180,
        color='lightgray',
        marker='o',
        alpha=0.6,
        edgecolor='darkgray',
        linewidth=1
    )
    
    # Highlight Pareto frontier points in orange with labels
    pareto_df = plot_df[plot_df['is_pareto']]
    
    # Sort Pareto points by x-coordinate for better label placement
    pareto_df = pareto_df.sort_values(by=time_col)
    
    for i, (_, row) in enumerate(pareto_df.iterrows()):
        # Plot Pareto points in orange
        ax.scatter(
            row[time_col],
            row[metric_cols[0]],
            s=180,
            color='darkorange',
            marker='o',
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        
        # Alternate annotation offset to reduce overlap
        x_offset = 20 if i % 2 == 0 else -50
        y_offset = 15 if i % 3 == 0 else -10

        ax.annotate(
            row['identifier'],
            xy=(row[time_col], row[metric_cols[0]]),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            fontsize=13,
            va='center',
            ha='left' if x_offset > 0 else 'right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7)
        )

    ax.set_xlabel('Answer Time per Question (seconds)', fontsize=14)
    ax.set_ylabel('Graded Accuracy', fontsize=14)

    x_min, x_max = plot_df[time_col].min(), plot_df[time_col].max()
    y_min, y_max = plot_df[metric_cols[0]].min(), plot_df[metric_cols[0]].max()

    x_padding = (x_max - x_min) * 0.3 if x_max > x_min else 5.0
    y_padding = (y_max - y_min) * 0.15 if y_max > y_min else 0.05

    ax.set_xlim(max(0, x_min - x_padding * 0.2), x_max + x_padding)
    ax.set_ylim(max(0, y_min - y_padding), min(1.05, y_max + y_padding))

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout(pad=2.0)
    fig.savefig(outpath, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {outpath}")
    plt.close(fig)




def splitter_chunksize_plot():
    query = """
        SELECT
            main.chunk_size,
            main.splitter_type,
            main.recall,
            main.mrr,
            index_fields.chunking_time_sec,
            time_fields.query_sec
        FROM main
        LEFT JOIN index_fields ON main.id = index_fields.id
        LEFT JOIN time_fields ON main.id = time_fields.id
        WHERE main.is_valid_config = 'true'
            AND main.chunk_size IS NOT NULL
            AND main.splitter_type IS NOT NULL
            AND main._timestamp IS NOT NULL
            AND main._project = 'cbfk-all-embeddings'
    """
    df = load(db_path, query)
    df = df.rename(columns={'query_sec': 'query_time_sec'}) # this becomes a label, needs to be expressive

    plot_chunk_size(
        df,
        output_dir,
        ['recall', 'mrr', 'query_time_sec', 'chunking_time_sec'],
        "splitter_type",
        "splitter_chunksize"
    )


def embedding_chunksize_plot():
    query = """
        SELECT
            main.chunk_size,
            main.embedding_model,
            main.recall,
            main.mrr, 
            main.graded_accuracy,
            index_fields.chunking_time_sec,
            time_fields.query_sec
        FROM main
        LEFT JOIN index_fields ON main.id = index_fields.id
        LEFT JOIN time_fields ON main.id = time_fields.id
        WHERE main.is_valid_config = 'true'
            AND main.chunk_size IS NOT NULL
            AND main.embedding_model IS NOT NULL
            AND main._timestamp IS NOT NULL
    """
    df = load(db_path, query)
    df = df.rename(columns={'query_sec': 'query_time_sec'}) # this becomes a label, needs to be  expressive 

    # some chunk size 256 runs had a bug and reported mrr 0.0, so set those to NaN
    df.loc[(df['chunk_size'] == "256") & (df['mrr'] == "0.0"), 'mrr'] = pd.NA
    
    plot_chunk_size(
        df,
        output_dir,
        ['recall', 'mrr', 'graded_accuracy', 'query_time_sec'],
        "embedding_model",
        "embedding_chunksize"
    )


def augment_chunks_plot():

    # Compare augmented runs with non-augmented runs that have the exact same config (embedding_model, splitter_type, chunk_size, llm_model).1
    # 1. identify the configurations used in augmented runs.
    # 2. use those configs to filter both augmented and non-augmented runs.
    # 3. compute averages grouped by those configs."""
    query = """
        WITH augmented_configs AS (
            SELECT DISTINCT
                embedding_model,
                splitter_type,
                chunk_size,
                llm_model
            FROM main
            WHERE augment_chunks IN ('1', 'true', 'True')
        ),

        filtered_runs AS (
            SELECT 
                m.augment_chunks,
                m.embedding_model,
                m.splitter_type,
                m.chunk_size,
                m.llm_model,
                CAST(m.recall AS REAL) AS recall,
                CAST(m.mrr AS REAL) AS mrr,
                CAST(m.graded_accuracy AS REAL) AS graded_accuracy
            FROM main m
            INNER JOIN augmented_configs ac
                ON m.embedding_model = ac.embedding_model
                AND m.splitter_type = ac.splitter_type
                AND m.chunk_size = ac.chunk_size
                AND m.llm_model = ac.llm_model
            WHERE m.augment_chunks IN ('1', 'true', 'True', '0', 'false', 'False')
        )

        SELECT 
            embedding_model,
            splitter_type,
            chunk_size,
            llm_model,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN recall END) AS avg_recall_non_augmented,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN recall END) AS avg_recall_augmented,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN mrr END) AS avg_mrr_non_augmented,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN mrr END) AS avg_mrr_augmented,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN graded_accuracy END) AS avg_graded_accuracy_non_augmented,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN graded_accuracy END) AS avg_graded_accuracy_augmented

        FROM filtered_runs
        GROUP BY 
            embedding_model,
            splitter_type,
            chunk_size,
            llm_model;
    """
    df_runs = load(db_path, query)
    save_markdown_table(df_runs, output_dir / "augment_chunks_runs.md")

    # Get the per question results for the above
    query = """
        WITH augmented_configs AS (
            SELECT DISTINCT
                embedding_model,
                splitter_type,
                chunk_size,
                llm_model
            FROM main
            WHERE augment_chunks IN ('1', 'true', 'True')
        ),

        matching_runs AS (
            SELECT 
                m.id,
                m.augment_chunks,
                m.embedding_model,
                m.splitter_type,
                m.chunk_size,
                m.llm_model
            FROM main m
            INNER JOIN augmented_configs ac
                ON m.embedding_model = ac.embedding_model
                AND m.splitter_type = ac.splitter_type
                AND m.chunk_size = ac.chunk_size
                AND m.llm_model = ac.llm_model
            WHERE m.augment_chunks IN ('1', 'true', 'True', '0', 'false', 'False')
        ),

        question_data AS (
            SELECT 
                q.question_nr,
                mr.embedding_model,
                mr.augment_chunks,
                CAST(q."vector_rr@5" AS REAL) AS vector_rr_5,
                CAST(q."vector_rr@8" AS REAL) AS vector_rr_8,
                CAST(q."vector_rr@13" AS REAL) AS vector_rr_13
            FROM questions q
            JOIN matching_runs mr ON q.id = mr.id
        )

        SELECT 
            question_nr,
            embedding_model,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN vector_rr_5 END) AS vector_rr_5_no_aug,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN vector_rr_5 END) AS vector_rr_5_aug,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN vector_rr_8 END) AS vector_rr_8_no_aug,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN vector_rr_8 END) AS vector_rr_8_aug,

            AVG(CASE WHEN augment_chunks IN ('0', 'false', 'False') THEN vector_rr_13 END) AS vector_rr_13_no_aug,
            AVG(CASE WHEN augment_chunks IN ('1', 'true', 'True') THEN vector_rr_13 END) AS vector_rr_13_aug

        FROM question_data
        GROUP BY 
            question_nr,
            embedding_model
        ORDER BY 
            embedding_model, question_nr;
    """
    df_questions = load(db_path, query)
    save_markdown_table(df_questions, output_dir / "augment_chunks_questions.md")
    plot_mrr_comparison(df_questions, output_dir / "augment_chunks_questions.png")


def plot_graded_accuracy_comparison(df: pd.DataFrame, output_path: Path | None = None) -> None:
    """
    Plots graded accuracy comparison for runs with and without query rewriter.
    Creates a separate plot for each embedding model, arranged in a single row.

    Args:
        df: Pandas DataFrame with columns:
            - question_nr
            - embedding_model
            - graded_accuracy_no_rewrite, graded_accuracy_rewrite
        output_path: Optional Path object to save the figure. If None, shows plot.
    """
    # Set up plot style
    PU.init_plot_style(light=True, large=True)
    
    # Get unique embedding models
    embedding_models = df['embedding_model'].unique()
    n_models = len(embedding_models)
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), squeeze=False)
    
    # Correctly get color palette using seaborn
    try:
        palette_name = PU.get_palette()  # Assuming this returns a string like "Orange" or a list of colors
        if isinstance(palette_name, str):
            actual_colors = sns.color_palette(palette_name, n_colors=2)
        elif isinstance(palette_name, list) and len(palette_name) >= 2:  # If PU.get_palette() already returns a list of colors
            actual_colors = palette_name
        else:  # Fallback if palette_name is not a string or a sufficient list
            logger.warning(f"PU.get_palette() returned an unexpected value: {palette_name}. Using default fallback colors.")
            actual_colors = sns.color_palette(n_colors=2)  # Seaborn default
            
        color_no_rewrite = actual_colors[0]
        color_rewrite = actual_colors[1]
    except Exception as e:
        logger.warning(f"Failed to get colors from PU.get_palette() or seaborn: {e}. Using default fallback colors.")
        # Fallback colors in case of any error
        color_no_rewrite = 'blue' 
        color_rewrite = 'green'
    
    # Ensure question_nr is numeric
    df['question_nr'] = pd.to_numeric(df['question_nr'], errors='coerce')
    
    # Plot each embedding model in its own subplot
    for i, model in enumerate(embedding_models):
        ax = axes[0, i]  # Get the appropriate axis from the 1xn grid
        
        # Filter data for this model and sort by question number
        model_df = df[df['embedding_model'] == model].sort_values('question_nr')
        
        # Iterate through each question to plot points and connecting lines
        for _idx, row in model_df.iterrows():  # _idx to avoid lint if not used
            q_nr = row['question_nr']
            ga_no_rewrite = pd.to_numeric(row['graded_accuracy_no_rewrite'], errors='coerce')
            ga_rewrite = pd.to_numeric(row['graded_accuracy_rewrite'], errors='coerce')
            
            if pd.notna(q_nr) and pd.notna(ga_no_rewrite) and pd.notna(ga_rewrite):
                # Plot points
                ax.scatter([q_nr], [ga_no_rewrite], color=color_no_rewrite, marker='o', s=50, zorder=3)
                ax.scatter([q_nr], [ga_rewrite], color=color_rewrite, marker='x', s=50, zorder=3)
                # Plot connecting line
                ax.plot([q_nr, q_nr], [ga_no_rewrite, ga_rewrite], color='grey', linestyle='-', linewidth=1, zorder=2)
        
        # Set titles and labels
        ax.set_title(PU.title_label(f"{model}"), fontsize=15)
        ax.set_xlabel("Question Number", fontsize=12)
        
        # Only add y-label to the leftmost plot
        if i == 0:
            ax.set_ylabel("Graded Accuracy", fontsize=12)
        
        # Create custom legend (only for the first subplot to avoid redundancy)
        if i == 0:
            legend_handles = [
                mlines.Line2D([], [], color=color_no_rewrite, marker='o', linestyle='None', markersize=8, label='Without Query Rewriter'),
                mlines.Line2D([], [], color=color_rewrite, marker='x', linestyle='None', markersize=8, label='With Query Rewriter')
            ]
            ax.legend(handles=legend_handles, fontsize=10, loc='lower right')
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Set y-axis limits and ticks
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        # Adjust x-axis ticks based on actual question numbers present
        unique_x_qnr_for_ticks = model_df['question_nr'].dropna().unique()
        if len(unique_x_qnr_for_ticks) > 0:
            if len(unique_x_qnr_for_ticks) > 15:
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7, integer=True))
            else:
                ax.set_xticks(sorted(list(unique_x_qnr_for_ticks)))
                if len(unique_x_qnr_for_ticks) > 5:  # Rotate if many ticks and close together
                    ax.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    #plt.suptitle(PU.title_label("Graded Accuracy with vs. without Query Rewriter"), fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save or show
    if output_path:
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved figure: {LC.HI}{output_path}{LC.RESET}")
    else:
        plt.show()


def query_rewriter_plot():
    """
    Compare runs with query rewriter with runs without query rewriter that have the exact same
    configurations (embedding_model, splitter_type, chunk_size, llm_model).
    
    Creates a figure showing graded_accuracy per question comparing the two approaches.
    """
    # Query to get overall comparison between runs with and without query rewriter
    query = """
        WITH rewriter_configs AS (
            SELECT DISTINCT
                embedding_model,
                splitter_type,
                chunk_size,
                llm_model
            FROM main
            WHERE query_rewrite IN ('1', 'true', 'True')
            AND _project like 'cbfk-rewrite'
        ),

        filtered_runs AS (
            SELECT 
                m.query_rewrite,
                m.embedding_model,
                m.splitter_type,
                m.chunk_size,
                m.llm_model,
                CAST(m.recall AS REAL) AS recall,
                CAST(m.mrr AS REAL) AS mrr,
                CAST(m.graded_accuracy AS REAL) AS graded_accuracy
            FROM main m
            INNER JOIN rewriter_configs rc
                ON m.embedding_model = rc.embedding_model
                AND m.splitter_type = rc.splitter_type
                AND m.chunk_size = rc.chunk_size
                AND m.llm_model = rc.llm_model
            WHERE m.query_rewrite IN ('1', 'true', 'True', '0', 'false', 'False')
            AND m._project like 'cbfk-rewrite'
        )

        SELECT 
            embedding_model,
            COUNT(DISTINCT splitter_type || '-' || chunk_size || '-' || llm_model) AS config_count,

            AVG(CASE WHEN query_rewrite IN ('0', 'false', 'False') THEN recall END) AS avg_recall_no_rewrite,
            AVG(CASE WHEN query_rewrite IN ('1', 'true', 'True') THEN recall END) AS avg_recall_rewrite,
            
            AVG(CASE WHEN query_rewrite IN ('0', 'false', 'False') THEN mrr END) AS avg_mrr_no_rewrite,
            AVG(CASE WHEN query_rewrite IN ('1', 'true', 'True') THEN mrr END) AS avg_mrr_rewrite,

            AVG(CASE WHEN query_rewrite IN ('0', 'false', 'False') THEN graded_accuracy END) AS avg_graded_accuracy_no_rewrite,
            AVG(CASE WHEN query_rewrite IN ('1', 'true', 'True') THEN graded_accuracy END) AS avg_graded_accuracy_rewrite

        FROM filtered_runs
        GROUP BY 
            embedding_model;
    """
    df_runs = load(db_path, query)
    save_markdown_table(df_runs, output_dir / "query_rewriter_runs.md")

    # Query to get per-question results for graded accuracy comparison
    query = """
        WITH rewriter_configs AS (
            SELECT DISTINCT
                embedding_model,
                splitter_type,
                chunk_size,
                llm_model
            FROM main
            WHERE query_rewrite IN ('1', 'true', 'True')
            AND _project like 'cbfk-rewrite'
        ),

        matching_runs AS (
            SELECT 
                m.id,
                m.query_rewrite,
                m.embedding_model,
                m.splitter_type,
                m.chunk_size,
                m.llm_model
            FROM main m
            INNER JOIN rewriter_configs rc
                ON m.embedding_model = rc.embedding_model
                AND m.splitter_type = rc.splitter_type
                AND m.chunk_size = rc.chunk_size
                AND m.llm_model = rc.llm_model
            WHERE m.query_rewrite IN ('1', 'true', 'True', '0', 'false', 'False')
            AND m._project like 'cbfk-rewrite'
        ),

        question_data AS (
            SELECT 
                q.question_nr,
                mr.embedding_model,
                mr.query_rewrite,
                CAST(q.graded_accuracy AS REAL) AS graded_accuracy
            FROM questions q
            JOIN matching_runs mr ON q.id = mr.id
        )

        SELECT 
            question_nr,
            embedding_model,
            AVG(CASE WHEN query_rewrite IN ('0', 'false', 'False') THEN graded_accuracy END) AS graded_accuracy_no_rewrite,
            AVG(CASE WHEN query_rewrite IN ('1', 'true', 'True') THEN graded_accuracy END) AS graded_accuracy_rewrite
        FROM question_data
        GROUP BY 
            question_nr,
            embedding_model
        ORDER BY 
            embedding_model, question_nr;
    """
    df_questions = load(db_path, query)
    save_markdown_table(df_questions, output_dir / "query_rewriter_questions.md")
    plot_graded_accuracy_comparison(df_questions, output_dir / "query_rewriter_questions.png")


def generation_time_plot() -> None:
    query = """
        SELECT 
            main.embedding_model, 
            main.llm_model, 
            main.mrr, 
            main.graded_accuracy, 
            main.augment_chunks, 
            main.query_rewrite,
            time_fields.query_sec, 
            time_fields.prompt_sec, 
            CAST(time_fields.query_sec AS FLOAT) + CAST(time_fields.prompt_sec AS FLOAT) AS total_time_sec,
            (CAST(time_fields.query_sec AS FLOAT) + CAST(time_fields.prompt_sec AS FLOAT)) / 20.0 AS one_time_sec,
            main.embedding_model || ' + ' || main.llm_model AS identifier
        FROM main
        JOIN time_fields ON main.id = time_fields.id
        WHERE 
            main.graded_accuracy IS NOT 'nan'
            AND main.is_valid_config IN ('1', 'true', 'True')
            AND main.embedding_model IN ('mxbai-large', 'nomic-moe')
            AND main.chunk_size IS NOT '256'
        GROUP BY embedding_model, llm_model, augment_chunks, query_rewrite
        ORDER BY CAST(main.graded_accuracy AS FLOAT) DESC;
    """
    df = load(db_path, query)
    df['graded_accuracy'] = df['graded_accuracy'].apply(lambda x: float(x) if x != 'nan' else float('nan'))
    df = df[df['graded_accuracy'] >= 0.4]   

    # Find Pareto frontier points
    df = find_pareto_frontier(df, 'one_time_sec', 'graded_accuracy')
    
    save_markdown_table(df[df['is_pareto']][[
        'identifier', 'mrr', 'graded_accuracy', 'augment_chunks', 'query_rewrite', 'one_time_sec'
    ]], output_dir / "generation_time.md")
    plot_vs_time(df, ['graded_accuracy'], 'one_time_sec', output_dir / "generation_time.png")


def main() -> None:
    # splitter_chunksize_plot() # 5.1 Dense Retrieval Results
    # embedding_chunksize_plot() # 5.1 Dense Retrieval Results
    # augment_chunks_plot() # 5.3 Chunk Augmentation Results
    # query_rewriter_plot() # 5.4 Query Rewriting Results
    generation_time_plot() # 5.5 Generation Results

if __name__ == "__main__":
    main()
