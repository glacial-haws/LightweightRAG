import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import wandb


def fetch_sweep_runs(project: str, sweep_id: str) -> pd.DataFrame:
    """
    Fetches all runs of a W&B sweep and returns a DataFrame with all config and summary fields, flattened.
    """
    # Defaults from sweep_agent.py
    DEFAULT_ENTITY: str | None = None  # Set to your username if you want to be explicit

    # If sweep_id does not contain a slash, prepend entity/project
    if "/" not in sweep_id:
        # If entity is None, omit it from the path (wandb defaults to your account)
        if DEFAULT_ENTITY is not None:
            sweep_path = f"{DEFAULT_ENTITY}/{project}/{sweep_id}"
        else:
            sweep_path = f"{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    runs = sweep.runs

    records: list[dict[str, object]] = []
    for run in tqdm(runs, desc="Fetching runs"):
        config = run.config
        summary = run.summary
        record: dict[str, object] = {}
        record["_run_id"] = run.id
        record["_run_name"] = run.name
        record["_run_state"] = run.state
        # Safely access metadata fields, handle missing metadata/keys
        meta = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        record["_gitcommit"] = (
            meta.get('git', {}).get('commit') if isinstance(meta.get('git'), dict) else None
        )
        record["_hostname"] = meta.get('host') if 'host' in meta else None
        record.update(config)
        record.update(summary)
        records.append(record)

    df = pd.DataFrame(records)
    return df.dropna(axis=1, how='all')




def save_excel(df: pd.DataFrame, path: Path) -> None:
    # Convert columns with object dtype but numeric-looking data to numeric, so floats/ints are not saved as strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Use to_numeric without errors='ignore'; catch exceptions explicitly (FutureWarning fix)
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, header=True, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        (max_row, max_col) = df.shape
        # Create format objects for numbers
        col_width = 20
        float_format = writer.book.add_format({'num_format': '#,##0.00000'})
        int_format = writer.book.add_format({'num_format': '#,##0'})
        
        # Adjust for 1-based indexing and header row
        worksheet.add_table(0, 0, max_row, max_col-1, {
            'name': 'SweepRuns',
            'columns': [{'header': col} for col in df.columns]
        })
        
        # --- Group columns by prefix (before first '.' or ':') ---
        import re
        from collections import defaultdict
        
        # Group columns by prefix before first '.', ':', or '@'
        prefix_map: dict[str, list[int]] = defaultdict(list)
        for idx, col in enumerate(df.columns):
            m = re.match(r'([^.:@]+)[.:@]', col)
            if m:
                prefix = m.group(1)
                prefix_map[prefix].append(idx)
        # Assign group level 1 to groups with more than 1 column
        group_level = [0] * len(df.columns)
        for group_idxs in prefix_map.values():
            if len(group_idxs) > 1:
                for idx in group_idxs:
                    group_level[idx] = 1

        # Set column formats with group levels
        for idx, col in enumerate(df.columns):
            dtype = df[col].dtype
            level = group_level[idx]
            if dtype.kind == 'i':  # integer type
                worksheet.set_column(idx, idx, col_width, int_format, options={'level': level})  # Pass options as 5th argument
            elif dtype.kind == 'f':  # float type
                worksheet.set_column(idx, idx, col_width, float_format, options={'level': level})
            elif dtype.kind == 'b':  # boolean type
                worksheet.set_column(idx, idx, col_width, int_format, options={'level': level})  # booleans as 0/1
            else:
                worksheet.set_column(idx, idx, col_width, None, options={'level': level})  # Pass options as 5th argument


def sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with '_run_name' and 'experiment_id' as the first columns (if present),
    followed by all other columns sorted in ascending order.
    """
    first_cols: list[str] = [col for col in ['_run_name', 'experiment_id'] if col in df.columns]
    other_cols: list[str] = sorted([col for col in df.columns if col not in first_cols])
    return df[first_cols + other_cols]



def main() -> None:
    parser = argparse.ArgumentParser(description="Dump sweep data to pickle, Excel and CSV")
    parser.add_argument(
        "--sweep-id",
        required=True,
        nargs='+',  # Accept multiple sweep IDs
        help="One or more W&B sweep IDs (e.g., 'sweep_id1 sweep_id2' or 'entity/project/sweep_id')"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help="Weights & Biases project name"
    )
    args = parser.parse_args()

    all_sweep_dfs: list[pd.DataFrame] = []
    for sweep_id_arg in args.sweep_id:
        # Sanitize each sweep_id_arg by replacing slashes with underscores for use in messages, not for API calls
        # The original sweep_id_arg (which might contain '/') is passed to fetch_sweep_runs
        sanitized_sweep_id_for_print = sweep_id_arg.replace('/', '_') 
        print(f"Fetching runs for sweep: {args.wandb_project}/{sanitized_sweep_id_for_print}")
        df_single_sweep = fetch_sweep_runs(args.wandb_project, sweep_id_arg)
        if not df_single_sweep.empty:
            all_sweep_dfs.append(df_single_sweep)
        else:
            print(f"No completed runs with relevant metrics found for sweep: {args.wandb_project}/{sanitized_sweep_id_for_print}")
        

    if not all_sweep_dfs:
        print("No data fetched for any of the provided sweep IDs.")
        return

    # Concatenate DataFrames. 'outer' join fills missing columns with NaN.
    df: pd.DataFrame = pd.concat(all_sweep_dfs, join='outer', ignore_index=True)

    if df.empty:
        print("Combined DataFrame is empty after fetching all sweeps.")
    else:
        # Create a string of all sweep IDs for filenames, sanitizing slashes
        joined_sweep_ids = "_".join(s.replace('/', '_') for s in args.sweep_id)
        
        results_dir_name = f'{args.wandb_project}_{joined_sweep_ids}'
        results_dir = Path('.') / 'results' / results_dir_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = f'{results_dir_name}' # Use the same name for files as for the directory
        pickle_path = results_dir / f'{base_filename}.pkl'
        xlsx_path = results_dir / f'{base_filename}.xlsx'
        ltd_xlsx_path = results_dir / f'{base_filename}_ltd.xlsx'
        noq_xlsx_path = results_dir / f'{base_filename}_noq.xlsx'
        csv_path = results_dir / f'{base_filename}.csv'

        # Convert all non-primitive columns to string to avoid Excel serialization errors
        df_serializable = df.copy()
        for col in df_serializable.columns:
            is_basic_serializable = (
                pd.api.types.is_numeric_dtype(df_serializable[col]) or
                pd.api.types.is_bool_dtype(df_serializable[col]) or
                pd.api.types.is_string_dtype(df_serializable[col])
            )

            if is_basic_serializable and df_serializable[col].dtype != object:
                continue
            
            if df_serializable[col].dtype == object:
                contains_complex_elements = df_serializable[col].apply(
                    lambda x: not isinstance(x, (str, int, float, bool, type(None)))
                ).any()

                if contains_complex_elements:
                    df_serializable[col] = df_serializable[col].apply(
                        lambda x: str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
                    )
            else:
                df_serializable[col] = df_serializable[col].astype(str)
        
        df_serializable = sort_cols(df_serializable)
        df_serializable.to_pickle(pickle_path)
        df_serializable.to_csv(csv_path, index=False)
        print(f"Saved sweep data to:\n{pickle_path}\n{csv_path}\n")
        
        save_excel(df_serializable, xlsx_path)
        print(f"{xlsx_path}")

        keep_cols = []
        end_to_del = ['.question', '.answer', '.eval_response', '.sources', '.expected_answer', '.report']
        exact_cols_to_del = ['_gitcommit', '_run_id', '_step', '_timestamp', '_wandb', 'index_report', 'experiment_run_report']
        for col in df_serializable.columns:
            if re.match(r'^question_\d{2}$', col):
                continue
            if col.endswith(tuple(end_to_del)) or col in exact_cols_to_del:
                continue
            keep_cols.append(col)
        df_limited = df_serializable[keep_cols]
        save_excel(df_limited, ltd_xlsx_path)
        print(f"{ltd_xlsx_path}")

        df_noq = df_limited.drop(columns=[col for col in df_limited.columns if col.startswith('question_')])
        save_excel(df_noq, noq_xlsx_path)
        print(f"{noq_xlsx_path}")


if __name__ == "__main__":
    main()
