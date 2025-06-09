import argparse
import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import wandb
from cbfk.log.log_config import LogConfig as LC

LC.setup_logging()
logger = logging.getLogger(__name__)


# Defaults from sweep_agent.py
DEFAULT_ENTITY: str | None = None  # Set to your username if you want to be explicit



def fetch_project_runs(project: str) -> pd.DataFrame:
    """
    Fetches all runs of a W&B project and returns a DataFrame with all config and summary fields, flattened.
    Similar to fetch_sweep_runs but retrieves all runs in a project regardless of sweep.
    """
    # Prepare the project path
    if DEFAULT_ENTITY is not None:
        project_path = f"{DEFAULT_ENTITY}/{project}"
    else:
        project_path = project

    api = wandb.Api()
    runs = api.runs(project_path)

    records: list[dict[str, object]] = []
    for run in tqdm(runs, desc=f"Fetching runs for {project}"):
        config = run.config
        summary = run.summary
        record: dict[str, object] = {}
        record["_run_id"] = run.id
        record["_run_name"] = run.name
        record["_run_state"] = run.state
        record["_sweep_id"] = run.sweep_id if hasattr(run, 'sweep_id') else None
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

    # Fix column names that changed over time
    df.rename(columns={
        'augment_questions': 'augment_chunks',
        'index.augment_questions': 'index.augment_chunks',
    }, inplace=True)

    return df.dropna(axis=1, how='all')


def sort_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with '_run_name' and 'experiment_id' as the first columns (if present),
    followed by all other columns sorted in ascending order.
    """
    first_cols: list[str] = [col for col in ['_run_name', 'experiment_id'] if col in df.columns]
    other_cols: list[str] = sorted([col for col in df.columns if col not in first_cols])
    return df[first_cols + other_cols]


def make_serializable(df: pd.DataFrame) -> pd.DataFrame:
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
    return df_serializable


def categorize_columns(df: pd.DataFrame) -> dict:
    column_groups = {
        'main': [], 'index': [], 'time': [], 'at': [], 'extras': [],
        'config': [], 'questions': {},
    }
    for col in df.columns:
        if col == 'id':
            continue
        if col.startswith('index.') or col == 'index_report':
            column_groups['index'].append(col)
        elif col.startswith('time.'):
            column_groups['time'].append(col)
        elif m := re.match(r'question_(\d+)\.(.+)', col):
            qid, field = m.group(1), m.group(2)
            column_groups['questions'].setdefault(qid, {})[field] = col
        elif '@' in col:
            column_groups['at'].append(col)
        elif col in ['_step', '_wandb', '_gitcommit', 
                     'experiment_run_report']:
            column_groups['extras'].append(col)
        elif col.startswith(('augmenting_', 'evaluating_', 'query_rewrite_')) \
                    or col in ['corpus_path', 'crossencoder_model', 
                                'llm_max_tokens', 'llm_temperature', 'rag_prompt']:
            column_groups['config'].append(col)
        else:
            column_groups['main'].append(col)

    for col in ['_project', '_run_id']:
        if col not in column_groups['main'] and col in df.columns:
            column_groups['main'].append(col)

    return column_groups


def add_missing_columns(conn: sqlite3.Connection, table: str, columns: list[str]):
    cursor = conn.execute(f'PRAGMA table_info({table})')
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col in columns:
        if col not in existing_cols:
            print(f'Adding column {LC.HI}{col}{LC.RESET} to table {LC.HI}{table}{LC.RESET}')
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" TEXT')


def create_main_table(conn: sqlite3.Connection, main_cols: list[str]):
    col_defs = ', '.join([f'"{col}" TEXT' for col in main_cols])
    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS main (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {col_defs},
            UNIQUE(_project, _run_id)
        )''')
    add_missing_columns(conn, 'main', main_cols)


def get_existing_pairs(conn: sqlite3.Connection) -> set:
    cursor = conn.cursor()
    cursor.execute('SELECT _project, _run_id FROM main')
    return set(cursor.fetchall())


def filter_new_rows(df: pd.DataFrame, existing_pairs: set) -> pd.DataFrame:
    return df[~df.apply(lambda row: (str(row.get('_project')), str(row.get('_run_id'))) in existing_pairs, axis=1)].copy()


def insert_main_data(conn: sqlite3.Connection, df: pd.DataFrame, main_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        df['id'] = []
        return df

    df[main_cols].to_sql('main', conn, if_exists='append', index=False)
    new_pairs = [(str(row['_project']), str(row['_run_id'])) for _, row in df.iterrows()]
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT id, _project, _run_id FROM main
        WHERE (_project, _run_id) IN ({','.join(['(?,?)'] * len(new_pairs))})
    """, [item for pair in new_pairs for item in pair])
    id_map = {(proj, run_id): id_ for id_, proj, run_id in cursor.fetchall()}
    df['id'] = [id_map[(str(row['_project']), str(row['_run_id']))] for _, row in df.iterrows()]
    return df


def create_and_insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame,
                       columns: list[str], prefix: str | None = None):
    if not columns or df.empty:
        return
    schema_cols = [col[len(prefix):] if prefix and col.startswith(prefix) else col for col in columns]
    rename_map = {col: schema_col for col, schema_col in zip(columns, schema_cols, strict=False)}

    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER,
            {', '.join([f'"{col}" TEXT' for col in schema_cols])},
            FOREIGN KEY(id) REFERENCES main(id) ON DELETE CASCADE
        )''')
    add_missing_columns(conn, table, schema_cols)

    df_subset = df[['id', *columns]].copy().rename(columns=rename_map)
    df_subset = df_subset[['id', *schema_cols]]
    df_subset.to_sql(table, conn, if_exists='append', index=False)


def insert_questions(conn: sqlite3.Connection, df: pd.DataFrame, questions: dict):
    if not questions or df.empty:
        return
    all_fields = sorted({f for fields in questions.values() for f in fields})

    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER,
            question_nr TEXT,
            {', '.join([f'"{f}" TEXT' for f in all_fields])},
            FOREIGN KEY(id) REFERENCES main(id) ON DELETE CASCADE
        )''')
    add_missing_columns(conn, 'questions', all_fields)

    rows = []
    for _, row in df.iterrows():
        for qid, fields in questions.items():
            entry = {'id': row['id'], 'question_nr': qid}
            for f in all_fields:
                colname = fields.get(f)
                entry[f] = row[colname] if colname in row and colname else None
            rows.append(entry)
    pd.DataFrame(rows).to_sql('questions', conn, if_exists='append', index=False)


def save_dataframe_to_sqlite(conn: sqlite3.Connection, df: pd.DataFrame, project: str) -> None:
    df = df.reset_index(drop=True).copy()
    col_groups = categorize_columns(df)
    create_main_table(conn, col_groups['main'])

    existing_pairs = get_existing_pairs(conn)
    df_new = filter_new_rows(df, existing_pairs)
    print(f'{LC.HI}{project}{LC.RESET} loaded {LC.HI}{df.shape[0]}{LC.RESET} rows, {LC.HI}{len(df_new)}{LC.RESET} new rows')
    df_new = insert_main_data(conn, df_new, col_groups['main'])

    create_and_insert(conn, 'index_fields', df_new, col_groups['index'], prefix='index.')
    create_and_insert(conn, 'time_fields', df_new, col_groups['time'], prefix='time.')
    create_and_insert(conn, 'at_fields', df_new, col_groups['at'])
    create_and_insert(conn, 'extras', df_new, col_groups['extras'])
    create_and_insert(conn, 'config', df_new, col_groups['config'])
    insert_questions(conn, df_new, col_groups['questions'])

    conn.commit()


def output_sqlite_schema(conn: sqlite3.Connection, output: str | None = None) -> None:
    """
    Outputs the SQLite database schema as a CREATE script.
    If output is None, prints to stdout; otherwise, writes to the given file path.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT type, name, sql FROM sqlite_master WHERE sql NOT NULL AND type IN ('table', 'index', 'view', 'trigger') ORDER BY type, name;")
    schema_entries: list[str] = []
    for type_, name, sql in cursor.fetchall():
        schema_entries.append(f"-- {type_}: {name}\n{sql.strip()};\n")
    schema_script = "\n".join(schema_entries)
    if output is None:
        print(schema_script)
    else:
        with open(output, "w", encoding="utf-8") as f:
            f.write(schema_script)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump sweep data to SQLite")
    parser.add_argument(
        "--wandb-project",
        type=str,
        nargs='+',
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the SQLite DB schema as a CREATE script and exit."
    )
    
    args = parser.parse_args()
 
    conn = sqlite3.connect(Path('results') / '_db' / 'runs.db')

    if getattr(args, "print_schema", False):
        output_sqlite_schema(conn)
        conn.close()
        return

    for project in args.wandb_project:
        try:
            df_single_project = fetch_project_runs(project)
        except Exception as e:
            print(f"Warning: Failed to fetch runs for project '{project}': {e}")
            continue
        if df_single_project.empty:
            print(f"No completed runs with relevant metrics found for project: {project}")
        else:
            # Convert complex types to serializable format before saving to SQLite
            df_serializable = make_serializable(df_single_project)
            # Add _project in a vectorized way to avoid fragmentation
            df_serializable = pd.concat(
                [df_serializable, pd.DataFrame({'_project': [project] * len(df_serializable)})],
                axis=1
            )

            save_dataframe_to_sqlite(conn, df_serializable, project)
    conn.close()

    
if __name__ == "__main__":
    main()
