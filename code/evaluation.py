import pandas as pd
from typing import Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def split_by_avg_min_max(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into three subsets using average, min, and max predicted scores.

    Each returned DataFrame contains a fixed set of metadata and feature columns,
    with one additional column: 'predicted_probability', populated from
    'average', 'min', or 'max' respectively.

    Args:
        df (pd.DataFrame): Input DataFrame with at least the required columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Three DataFrames: (df_avg, df_min, df_max)
"""
    base_cols = [
        'id',
        'name',
        'doi',
        'paragraph',
        'authors',
        'field/topic/keywords',
        'url (ground truth)',
        'candidate_urls',
        'probability (ground truth)',
        'metadata_name',
        'metadata_authors',
        'metadata_keywords',
        'metadata_description',
        'name_metric',
        'author_metric',
        'paragraph_metric',
        'keywords_metric',
        "language_metric"
    ]

    # average-based
    df_avg = df[base_cols].copy()
    df_avg['predicted_probability'] = df['average']

    # min-based
    df_min = df[base_cols].copy()
    df_min['predicted_probability'] = df['min']

    # max-based
    df_max = df[base_cols].copy()
    df_max['predicted_probability'] = df['max']

    return df_avg, df_min, df_max

def group_by_candidates(df: pd.DataFrame, output_path:str) -> pd.DataFrame:
    """
    Rank and aggregate candidate URLs by predicted probability.

    Groups rows by ('name', 'doi', 'paragraph'), then within each group:
    - Sorts by 'predicted_probability' (descending)
    - Aggregates:
        • id, authors, and ground-truth URL: take first entry
        • candidate_urls: comma-separated ranked URLs
        • predicted probabilities: comma-separated, same order

    Args:
        df (pd.DataFrame): Must contain 'predicted_probability' and candidate info.
        output_path (str): Optional. If provided, saves result to CSV.

    Returns:
        pd.DataFrame: Aggregated candidates with rankings per group.
"""
    # 1) Sort so that within each (name, doi, paragraph) block,
    #    highest predicted_probability comes first.
    df_sorted = df.sort_values(
        by=['name', 'doi', 'paragraph', 'predicted_probability'],
        ascending=[True, True, True, False]
    )

    # 2) Group and aggregate
    grouped = df_sorted.groupby(
        ['name', 'doi', 'paragraph'],
        as_index=False
    ).agg({
        'id': 'first',
        'authors': 'first',
        'field/topic/keywords': 'first',
        'url (ground truth)': 'first',
        'candidate_urls':       lambda urls: ",".join(urls),
        'predicted_probability': lambda probs: ",".join(map(str, probs))
    })

    # 3) Rename and reorder
    grouped = grouped.rename(
        columns={'predicted_probability': 'probability_ranked'}
    )
    ordered_cols = [
        'id',
        'name',
        'doi',
        'paragraph',
        'authors',
        'field/topic/keywords',
        'url (ground truth)',
        'candidate_urls',
        'probability_ranked'
    ]
    # 4) Save to CSV if output_path is provided
    if output_path:
        grouped[ordered_cols].to_csv(output_path, index=False)
        print(f"Grouped DataFrame saved to {output_path}")
    return grouped[ordered_cols]


def split_by_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate binary predictions based on average, min, and max scores.

    Creates three DataFrames with the same base columns and a new
    'prediction' column, where a value of 1 means score > 0.5.

    Args:
        df (pd.DataFrame): Must contain 'average', 'min', and 'max' columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            DataFrames with binary predictions from average, min, and max scores.
"""
    base_cols = [
        'id','name','doi','paragraph','authors',
        'field/topic/keywords','url (ground truth)','candidate_urls'
    ]

    # Average-summary predictions
    avg_df = df[base_cols].copy()
    avg_df['prediction'] = (df['average'] > 0.5).astype(int)

    # Min-summary predictions
    min_df = df[base_cols].copy()
    min_df['prediction'] = (df['min'] > 0.5).astype(int)

    # Max-summary predictions
    max_df = df[base_cols].copy()
    max_df['prediction'] = (df['max'] > 0.5).astype(int)

    return avg_df, min_df, max_df

def mrr_at_1(df: pd.DataFrame) -> float:
    """Compute MRR@1 (i.e. precision@1) over all rows.

    For each row, checks if the top‐ranked URL is in the ground‐truth set.

    Args:
        df (pd.DataFrame): Must contain
            - 'url (ground truth)' (comma-separated correct URLs)
            - 'candidate_urls'   (comma-separated ranked URLs)

    Returns:
        float: Mean of 1.0 where top-1 is correct, else 0.0.
    """
    rr_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        top1 = row["candidate_urls"].split(',')[0]  # highest‐ranked URL
        rr_scores.append(1.0 if top1 in true_set else 0.0)
    return sum(rr_scores) / len(rr_scores)




def full_mrr(
    df: pd.DataFrame
) -> float:
    """Compute full Mean Reciprocal Rank (MRR) over all rows.

    For each row:
      1. Identify the 1‐based rank of the first correct URL.
      2. Compute reciprocal rank = 1 / rank (or 0 if no match).

    Args:
        df (pd.DataFrame): Must contain
            - 'url (ground truth)' (comma-separated correct URLs)
            - 'candidate_urls'   (comma-separated ranked URLs)

    Returns:
        float: Average of reciprocal ranks across all rows.
    """
    rr_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        position = 0
        for idx, candidate in enumerate(row["candidate_urls"].split(','), start=1):
            if candidate in true_set:
                position = idx
                break
        rr_scores.append(1.0 / position if position > 0 else 0.0)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0



def r_precision(df: pd.DataFrame) -> float:
    """Compute average R-Precision over all rows.

    For each row:
      - Let R = number of ground-truth URLs.
      - R-Precision = (# of correct URLs in top-R candidates) / R.

    Args:
        df (pd.DataFrame): Must contain
            - 'url (ground truth)' (comma-separated correct URLs)
            - 'candidate_urls'   (comma-separated ranked URLs)

    Returns:
        float: Mean R-Precision across all rows.
    """
    rp_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        R = len(true_set)
        if R == 0:
            # if you have no ground truth for a mention, you may choose to skip it
            continue
        top_R = row["candidate_urls"].split(',')[:R]
        hits = len(set(top_R) & true_set)
        rp_scores.append(hits / R)
    return sum(rp_scores) / len(rp_scores) if rp_scores else 0.0


def evaluation(df: pd.DataFrame) -> None:
    """
        Compute and print binary classification metrics for a match task.

        Assumes that the ground-truth label (column 'true_label') is already provided,
        with value 1 if the candidate URL is among the correct URLs, and 0 otherwise.

        Args:
            df (pd.DataFrame): Must contain:
                - 'prediction': binary model prediction (0 or 1)
                - 'true_label': binary ground-truth label (0 or 1)

        Prints:
            - Precision, recall, and F1-score (rounded to 2 decimals)
            - Full sklearn classification report
"""

    