"""
Post-sweep evaluation script 

- Re-evaluates sweep dumps with different LLMs, BLEU/ROUGE, and creates new Excel for analysis and human evaluation.
- Picks answers with most deviating rankings and allows manual ranking. Plots this vs time to evaluate and pick one evaluation LLM.

Usage:
    uv run cbfk/post_sweep_evaluation.py --wandb-project <project> --sweep-id <sweep_id>

Input:
    CSV at results/subfolder <wandb-project>-<sweep-id>/sweep_<sweep-id>.csv
Output:
    Excel at results/subfolder <wandb-project>-<sweep-id>/sweep_<sweep-id>_post_eval.xlsx
"""

import math
import re
from pathlib import Path
from typing import Tuple

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_utils import PlotUtils as PU
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

import cbfk.evaluate
import cbfk.sweep_config
from cbfk.log.log_config import LogConfig as LC

absl.logging.set_verbosity(absl.logging.FATAL)

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer
except ImportError:
    sentence_bleu = None
    rouge_scorer = None


def calc_bleu(reference: str | float, hypothesis: str | float) -> float:
    """Calculate BLEU score between reference and hypothesis.
    
    Args:
        reference: Reference text or NaN
        hypothesis: Hypothesis text or NaN
        
    Returns:
        float: BLEU score or NaN if inputs are invalid
    """
    if sentence_bleu is None:
        return float('nan')
    
    # Handle NaN or non-string inputs
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return float('nan')
    
    # Handle empty strings
    if not reference.strip() or not hypothesis.strip():
        return float('nan')
    
    ref = [reference.split()]
    hyp = hypothesis.split()
    return float(sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1))


def calc_rouge(reference: str | float, hypothesis: str | float) -> float:
    """Calculate ROUGE-L score between reference and hypothesis.
    
    Args:
        reference: Reference text or NaN
        hypothesis: Hypothesis text or NaN
        
    Returns:
        float: ROUGE-L score or NaN if inputs are invalid
    """
    if rouge_scorer is None:
        return float('nan')
    
    # Handle NaN or non-string inputs
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return float('nan')
    
    # Handle empty strings
    if not reference.strip() or not hypothesis.strip():
        return float('nan')
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return float(score['rougeL'].fmeasure)


def save_excel(df: pd.DataFrame, xlsx_out: Path, postfix: str = "", debug: bool = False) -> None:
    xlsx_out = xlsx_out.with_stem(xlsx_out.stem + postfix)
    if debug:
        xlsx_out = xlsx_out.with_stem(xlsx_out.stem + "_debug")
    print(f"Saving dataframe {df.shape} to {xlsx_out}")
    with pd.ExcelWriter(xlsx_out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

        # Format main DataFrame sheet
        worksheet1 = writer.sheets["Sheet1"]
        (max_row, max_col) = df.shape
        worksheet1.add_table(0, 0, max_row, max_col-1, {
            'columns': [{'header': col} for col in df.columns],
            'name': 'MainTable',
            'style': 'Table Style Medium 9'
        })
        for col_idx in range(max_col):
            worksheet1.set_column(col_idx, col_idx, 40)


def evaluate_response(model: str, temperature: float, question: str, truth: str, 
                     llm_response: str, recall: float) -> Tuple[float, str, float]:
    try:
        score, response, elapsed_time = cbfk.evaluate.evaluate_llm_response(
            model=model, 
            question=question, 
            truth=truth, 
            llm_response=llm_response, 
            recall=recall, 
            temperature=temperature, 
            max_tokens=8192)
        # print(f"{model}: {score:0.5f} ({elapsed_time:0.2f}s)")
        # print(f'--- {response}\n')
        return score, response, elapsed_time
    except Exception as e:
        print(f"Error evaluating response with {model}: {e.__class__.__name__}: {e}")
        raise e


def plot_violin_deviation(
    deviation_df: pd.DataFrame,
    temperature: float,
    palette: str,
    output_dir: Path ) -> None:
    """Plot violin plot of deviations from human scores."""
    plt.figure(figsize=(18, 6))
    sns.violinplot(x='model', y='deviation', hue='model', data=deviation_df, inner='box', palette=palette, legend=False)
    plt.title("Absolute Deviation from Human Score per Model")
    plt.ylabel("Absolute Deviation")
    plt.xlabel("")  # Set x-label to empty string to remove it completely
    plt.ylim(0, 1)  # Clamp y-axis to [0, 1] to avoid visual artifacts
    plt.xticks(rotation=30, ha='right')  # Rotate x-labels by 30 degrees
    plt.tight_layout()
    plotname = output_dir / f"violin_deviation_per_model_{temperature}.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def plot_box_deviation(
    deviation_df: pd.DataFrame,
    palette: str,
    output_dir: Path ) -> None:
    """Plot box plot of deviations from human scores."""
    plt.figure(figsize=(18, 6))
    sns.boxplot(x='model', y='deviation', hue='model', data=deviation_df, palette=palette, dodge=False, showfliers=True)
    plt.title("Absolute Deviation from Human Score per Model (Box Plot)")
    plt.ylabel("Absolute Deviation")
    plt.xlabel("Model")
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plotname = output_dir / "box_deviation_per_model.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def plot_barchart_rmse_and_time(
    rmse_scores: dict[str, float],
    time_stats: dict[str, dict[str, float]],
    output_dir: Path ) -> None:
    """Plot RMSE per model with time overlay."""
    models_sorted = sorted(rmse_scores, key=rmse_scores.get)
    rmses_sorted = [rmse_scores[m] for m in models_sorted]
    times_sorted = [time_stats[m]['mean_time'] for m in models_sorted]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.barh(models_sorted, rmses_sorted, color='skyblue', label='RMSE')
    ax1.set_title('RMSE and Inference Time per Model', pad=20)
    ax1.invert_yaxis()
    
    # Rotate y-axis tick labels by 30 degrees to avoid overlap
    plt.yticks(rotation=30, ha='right')

    # Secondary axis for time (ensure diamonds are visible)
    ax2 = ax1.twiny()
    min_time, max_time = min(times_sorted), max(times_sorted)
    pad = (max_time - min_time) * 0.1 if max_time > min_time else 1.0
    ax2.set_xlim(min_time - pad, max_time + pad)
    ax2.set_ylim(ax1.get_ylim())  # align y-axes

    # Overlay diamonds for mean_time_sec
    for i, (bar, t) in enumerate(zip(bars, times_sorted, strict=False)):
        y = bar.get_y() + bar.get_height()/2
        ax2.plot(t, y, marker='D', color='orange', markersize=16, markeredgecolor='black', 
                 label='Mean Inference Time' if i == 0 else "")
        ax2.text(t, y, f'{t:.2f}s', va='center', ha='left', color='orange', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', colors='orange')
    ax2.yaxis.set_visible(False)

    # Handle legend (only one entry for diamonds)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    by_label = dict(zip(labels, handles, strict=False))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plotname = output_dir / "rmse_per_model.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def plot_violin_inference_time(
    df: pd.DataFrame,
    models: list[str],
    temperature: float,
    palette: str,
    output_dir: Path ) -> None:
    """Plot violin plot of inference times."""
    # Collect all timing values per model
    all_times = []
    for model in models:
        time_col = f"{model}_time_sec"
        if time_col in df:
            for t in df[time_col]:
                all_times.append({'model': model, 'time_sec': t})
    all_times_df = pd.DataFrame(all_times)

    plt.figure(figsize=(18, 6))
    sns.violinplot(x='model', y='time_sec', hue='model', data=all_times_df, inner='box', palette=palette, legend=False)
    plt.title("Inference Time Statistics per Model")
    plt.ylabel("Time (seconds)")
    plt.xlabel("")  # Set x-label to empty string to remove it completely
    plt.xticks(rotation=30, ha='right')  # Rotate x-labels by 30 degrees
    plt.tight_layout()
    plotname = output_dir / f"violin_time_per_model_{temperature}.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def plot_correlation_scatterplots(
    df: pd.DataFrame,
    correlation_scores: dict[str, float],
    model_to_color: dict[str, tuple],
    output_dir: Path ) -> None:
    """Plot correlation scatter plots with trendlines."""
    n_models = len(correlation_scores)
    cols = 4
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (model, corr) in enumerate(correlation_scores.items()):
        score_col = f"{model}_score"
        ax = axes[idx]
        color = model_to_color.get(model, None)
        sns.regplot(x='human_score', y=score_col, data=df, ax=ax, scatter_kws={'s': 10}, color=color)
        ax.set_title(f"{model} (r = {corr:.2f})")
        ax.set_xlabel("Human Score")
        ax.set_ylabel("Model Score")

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plotname = output_dir / "model_vs_human_scatterplots.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def plot_scatterplot_rmse_vs_time(
    rmse_scores: dict[str, float],
    time_stats: dict[str, dict[str, float]],
    model_to_color: dict[str, tuple],
    temperature: float,
    output_dir: Path ) -> None:
    """Create a scatterplot of RMSE vs inference time per model.
    
    This plot helps visualize the trade-off between model accuracy (RMSE)
    and performance (inference time).
    """
    # Extract data for plotting
    models = list(rmse_scores.keys())
    rmses = [rmse_scores[m] for m in models]
    times = [time_stats[m]['mean_time'] for m in models]
    colors = [model_to_color.get(m, (0.5, 0.5, 0.5)) for m in models]
    
    # Create the scatterplot
    plt.figure(figsize=(12, 7))
    
    # Plot points
    for i, model in enumerate(models):
        plt.scatter(times[i], rmses[i], color=colors[i], s=400, label=model)
        plt.text(times[i] * 1.02, rmses[i], model, fontsize=16, va='top', ha='left')
    
    # Add trend line (optional)
    if len(models) > 2:
        z = np.polyfit(times, rmses, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(times) * 0.9, max(times) * 1.1, 100)
        plt.plot(x_trend, p(x_trend), "--", color='gray', alpha=0.6)
    
    # Add quadrants with median values
    median_rmse = np.median(rmses)
    median_time = np.median(times)
    plt.axhline(median_rmse, color='lightgray', linestyle='--', alpha=0.5)
    plt.axvline(median_time, color='lightgray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    plt.text(max(times) * 0.9, min(rmses) * 1.1, "Slow & Accurate", 
             fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(min(times) * 1.1, min(rmses) * 1.1, "Fast & Accurate", 
             fontsize=10, ha='left', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(max(times) * 0.9, max(rmses) * 0.9, "Slow & Inaccurate", 
             fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.5))
    plt.text(min(times) * 1.1, max(rmses) * 0.9, "Fast & Inaccurate", 
             fontsize=10, ha='left', bbox=dict(facecolor='white', alpha=0.5))
    
    # Set labels and title
    plt.xlabel("Mean Inference Time (seconds, lower is better)")
    plt.ylabel("RMSE (lower is better)")
    plt.title(f"Model Accuracy vs. Speed Trade-off (Temperature: {temperature})")
    
    # Ensure axes start near zero but include all data points with some padding
    plt.xlim(min(times) * 0.9, max(times) * 1.1)
    plt.ylim(min(rmses) * 0.9, max(rmses) * 1.1)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plotname = output_dir / f"rmse_vs_time_scatterplot_{temperature}.png"
    plt.savefig(plotname)
    print(f"Saved {LC.HI}{plotname}{LC.RESET}")
    plt.close()


def export_stats_markdown(
    correlation_scores: dict[str, float],
    rmse_scores: dict[str, float],
    time_stats: dict[str, dict[str, float]],
    temperature: float,
    output_dir: Path) -> None:
    """Export markdown table with model statistics and temperature."""
    md_table = ""
    md_table += "| Model | RMSE | Correlation | Mean Time Sec | Stdev Time Sec |\n"
    md_table += "|-------|------|-------------|---------------|----------------|\n"
    for model in correlation_scores:
        md_table += f"| {model} | {rmse_scores[model]:.2f} | {correlation_scores[model]:.2f} | {time_stats[model]['mean_time']:.2f} | {time_stats[model]['stdev_time']:.2f} |\n"
    md_table += "\n"
    with open(output_dir / f"model_vs_human_stats_{temperature}.md", "w") as f:
        f.write(md_table)


def analyze_model_vs_human(
    df: pd.DataFrame,
    models: list[str],
    temperature: float,
    output_dir: Path,
    light: bool = False) -> dict:
    """Analyze model scores against human scores and generate visualizations.
    
    Args:
        df: DataFrame containing model and human scores
        models: List of model names to analyze
        output_dir: Directory to save output plots and files
        light: Whether to use light theme for plots
        
    Returns:
        Dictionary containing analysis results
    """
    PU.init_plot_style(light=light)
    palette = PU.get_palette()
    color_list = sns.color_palette(palette, n_colors=len(models))
    model_to_color = {model: color_list[i] for i, model in enumerate(models)}

    # Containers
    deviations = []
    rmse_scores = {}
    correlation_scores = {}
    time_stats: dict[str, dict[str, float]] = {}

    for model in models:
        score_col = f"{model}_score"
        time_col = f"{model}_time_sec"

        if score_col not in df or time_col not in df:
            print(f"Warning: Columns for model '{model}' not found.")
            continue

        # Absolute deviation
        abs_deviation = np.abs(df[score_col] - df['human_score'])
        deviations.append(pd.DataFrame({
            'model': model,
            'deviation': abs_deviation
        }))

        # RMSE
        rmse_scores[model] = math.sqrt(mean_squared_error(df['human_score'], df[score_col]))

        # Pearson correlation
        corr, _ = pearsonr(df['human_score'], df[score_col])
        correlation_scores[model] = corr

        # Time stats
        time_values = df[time_col]
        time_stats[model] = {
            'mean_time': time_values.mean(),
            'stdev_time': time_values.std()
        }

    # Combine all deviations into a single DataFrame
    deviation_df = pd.concat(deviations, ignore_index=True)
    print(f"Deviation min/max: {deviation_df['deviation'].min()}, {deviation_df['deviation'].max()}")

    # Generate all plots
    # plot_box_deviation(deviation_df, palette, output_dir)
    # plot_barchart_rmse_and_time(rmse_scores, time_stats, output_dir)
    # plot_correlation_scatterplots(df, correlation_scores, model_to_color, output_dir)
    plot_violin_deviation(deviation_df, temperature, palette, output_dir)
    plot_violin_inference_time(df, models, temperature, palette, output_dir)
    plot_scatterplot_rmse_vs_time(rmse_scores, time_stats, model_to_color, temperature, output_dir)
    export_stats_markdown(correlation_scores, rmse_scores, time_stats, temperature, output_dir)

    # Return analysis results
    return {
        'rmse': rmse_scores,
        'correlation': correlation_scores,
        'time_stats': pd.DataFrame(time_stats),
        'deviation_df': deviation_df
    }


def plot_evaluation(files: list[str], light: bool = False) -> None:
    if len(files) < 2:
        raise ValueError("Need at least 2 files to plot evaluation.")
    
    post_eval_file = Path(files[0])
    human_eval_file = Path(files[1])

    post_eval_df = pd.read_excel(post_eval_file)
    human_eval_df = pd.read_excel(human_eval_file)

    temperature = post_eval_file.stem.split('_')[-1]

    # Merge on 'question'
    eval_df = pd.merge(human_eval_df, post_eval_df, on='answer', how='inner')

    assert len(post_eval_df) == len(eval_df), \
        f"number of rows of post_eval_df and eval_df should be the same: {len(post_eval_df)}, {len(eval_df)}"

    cols = post_eval_df.columns
    llm_models = [col.split('_')[0] for col in cols if col.endswith('_score')]

    analyze_model_vs_human(
        eval_df,
        models=llm_models,
        temperature=temperature,
        output_dir=Path(post_eval_file.parent / "media"),
        light=light
    )



def rearrange(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Rearrange dataframe to have one row per question with experiment info columns."""
    global __DEBUG

    experiment_info_columns  = [
        '_run_name', '_run_state', '_runtime', 'experiment_id', 
        'splitter_type', 'chunk_size', 'chunk_overlap_pct', 'chunk_overlap', 'embedding_model', 
        'similarity_top_k', 
        'llm_model', 'llm_temperature', 'llm_max_tokens', 'is_valid_config', 
        'precision', 'recall', 'graded_accuracy', 
        'using_cached_index', 
        'index_time_sec', 'query_time_sec', 'prompt_time_sec', 'evaluate_time_sec', 'experiment_time_sec', 
    ]
    cols_per_question = [
        'question',
        'expected_answer',
        'answer',
        'precision',
        'recall',
        'graded_accuracy',
        'eval_response'
    ]
    
    # Find all question numbers in the dataframe
    question_nums: list[int] = []
    for col in df.columns:
        match = re.match(r"question_(\d+)", col)
        if match:
            question_nums.append(int(match.group(1)))
    
    max_question_num: int = max(question_nums) if question_nums else 0
    
    # Filter experiment info columns that exist in the dataframe
    exp_info_cols = [col for col in experiment_info_columns if col in df.columns]
    
    # Create a new dataframe with questions as rows
    rows = []
    
    # For each row in the original dataframe
    for _, row in df.iterrows():
        # For each question (including the last one)
        for q_num in range(max_question_num + 1):
            # Create a new row for this question
            question_row = {}
            
            # First add experiment info columns
            for col in exp_info_cols:
                question_row[col] = row.get(col, '')
            
            # Add question number after experiment info
            question_row['question_number'] = q_num
            
            # Add question-specific columns
            prefix = f"question_{q_num:02d}."
            for col in cols_per_question:
                col_name = f"{prefix}{col}"
                if col_name in df.columns:
                    question_row[col] = row.get(col_name, '')
            
            rows.append(question_row)
            if debug: 
                if q_num > 2: break
    
    # Create the new dataframe
    output_df = pd.DataFrame(rows)
    return output_df


def pse_main() -> None:

    args = PU.parse_args()
    if not args.wandb_project and not args.sweep_id and not args.files:
        PU.print_help()
        print("Please provide wandb_project and sweep_id to evaluate sweep or files to plot.")
        print("For plots, provide _post_eval_<temp>.xlsx and _human_eval.xlsx file in this order.")
        return

    if args.files:
        plot_evaluation(args.files, args.light)
        return
    else:
        results_dir = Path(f"results/{args.wandb_project}_sweep_{args.sweep_id}")
        csv_path = results_dir / f"sweep_{args.sweep_id}.csv"
    xlsx_out = results_dir / f"sweep_{args.sweep_id}_post_eval.xlsx"
    human_eval_file_name = results_dir / f"sweep_{args.sweep_id}_human_eval.xlsx"
    temperature = args.temperature
    
    print(f"Reading sweep data from {csv_path}")
    # Check if CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please make sure you've run the sweep and the CSV file exists.")
        return
        
    # Read and filter the dataframe
    df = pd.read_csv(csv_path)
    df = df[df['_run_state'] == 'finished']
    if args.debug:  # limit to a small sample for debugging
        df = df.head(2)

    # Rearrange dataframe to have one row per question
    print("Rearranging dataframe to have one row per question...")
    output_df = rearrange(df, args.debug)
    
    # Remove <think> section from answer (robust to multiline and whitespace)
    def extract_think(text: str) -> str:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else ''

    def remove_think(text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    output_df['think'] = output_df['answer'].apply(extract_think)
    output_df['answer'] = output_df['answer'].apply(remove_think)

    # Create file for human eval if not exists
    human_eval_file_name = results_dir / f"sweep_{args.sweep_id}_human_eval.xlsx"
    if not human_eval_file_name.exists():
        human_eval_cols = ['_run_name', 'experiment_id', 'llm_model', 'question_number', 'question', 'expected_answer', 'think', 'answer']
        human_eval_df = output_df[human_eval_cols].copy()
        human_eval_df['human_score'] = human_eval_df['answer'].apply(lambda x: 0)
        save_excel(human_eval_df, human_eval_file_name, debug=args.debug)
    else:
        print(f"Human eval file {human_eval_file_name} exists.")

    # Calculate BLEU and ROUGE scores
    print("Calculating BLEU and ROUGE scores...")
    output_df['bleu'] = output_df.apply(lambda row: calc_bleu(row['expected_answer'], row['answer']), axis=1)
    output_df['rouge'] = output_df.apply(lambda row: calc_rouge(row['expected_answer'], row['answer']), axis=1)

    # Evaluate responses with multiple LLMs
    if args.llm_model:
        llm_models = [args.llm_model]
        print(f"Evaluating responses with {args.llm_model}...")
    else:
        llm_models = cbfk.models.get_model_registry().list_short_names(cbfk.models.ModelType.LLM)
        if args.debug:
            llm_models = llm_models[:2]
        print(f"Evaluating responses with {len(llm_models):d} models: {','.join(llm_models)}")            

    for llm_model in llm_models:
        # Create temporary columns to store the evaluation results
        # Enable progress bar for pandas apply
        tqdm.pandas(desc=f"Evaluating with {llm_model} temp {temperature}", total=len(output_df))
        
        # Use progress_apply instead of apply
        output_df[[f'{llm_model}_score', f'{llm_model}_response', f'{llm_model}_time_sec']] = \
            output_df.progress_apply(
                lambda row, model=llm_model: evaluate_response(
                    model, temperature, 
                    row['question'], 
                    row['expected_answer'], 
                    row['answer'],
                    row['recall']),
                axis=1, result_type='expand'
            )
        save_excel(output_df, xlsx_out, postfix=f"_{llm_model}_{temperature:0.1f}", debug=args.debug)

    if not args.llm_model:

        # Create combined Excel file
        save_excel(output_df, xlsx_out, postfix=f"_{temperature:0.1f}", debug=args.debug)

        print(f"Post evaluation completed with {len(llm_models):d} models.")
        print(f"\nRun {LC.HI}insights/post_sweep_evaluation.py --files {xlsx_out} {human_eval_file_name}{LC.RESET} to plot.")

if __name__ == "__main__":
    pse_main()
