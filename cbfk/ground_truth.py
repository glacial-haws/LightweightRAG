

from collections import defaultdict
from pathlib import Path

import pandas as pd


class GroundTruth:
    """
    Loads and serves ground truth data for question-answer-source triples.
    Data is loaded from an Excel file with columns:
        - Question
        - Answer
        - File
        - Section
        - Quote
    Each question/answer pair may appear on multiple rows, each with different source info.
    After loading, the data structure is:
        [
            {
                'query': str,
                'answer': str,
                'sources': [
                    {'file_name': str, 'chapter': str, 'quote': str, 'weight': float}
                ]
            },
            ...
        ]
    """
    def __init__(self, excel_path: str | Path) -> None:
        self.ground_truth: list[dict] = read_ground_truth_from_excel(excel_path)

    def get_all(self) -> list[dict]:
        """Return the loaded ground truth list of dicts."""
        return self.ground_truth

def read_ground_truth_from_excel(excel_path: str | Path) -> list[dict]:
    """
    Read ground truth from an Excel file and return as a list of dicts.
    Each dict has 'query', 'answer', and 'sources' (list of file/section/quote dicts).
    """
    df = pd.read_excel(excel_path)
    # Normalize column names (strip, lower)
    df.columns = [c.strip().lower() for c in df.columns]
    # Required columns
    required = {'question', 'answer', 'file', 'section', 'quote'}
    if not required.issubset(df.columns):
        raise ValueError(f"Excel file must contain columns: {required}")

    # Group by (question, answer)
    qa_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for _, row in df.iterrows():
        q = str(row['question']).strip()
        a = str(row['answer']).strip()
        file_name = str(row['file']).strip()
        chapter = str(row['section']).strip()
        quote = str(row['quote']).strip()
        # Optional: weight (default 1.0)
        weight = float(row['weight']) if 'weight' in row and pd.notna(row['weight']) else 1.0
        qa_groups[(q, a)].append({
            'file_name': file_name,
            'chapter': chapter,
            'quote': quote,
            'weight': weight,
        })
    # Build output structure
    ground_truth: list[dict] = []
    for (q, a), sources in qa_groups.items():
        ground_truth.append({
            'query': q,
            'answer': a,
            'sources': sources
        })
    return ground_truth

