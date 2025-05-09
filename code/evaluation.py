import pandas as pd
from typing import Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def split_by_avg_min_max(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a DataFrame `df` with columns:
      id, name, doi, paragraph, authors, field/topic/keywords,
      url (ground truth), candidate_urls, probability (ground truth),
      metadata_name, metadata_authors, metadata_keywords, metadata_description,
      name_metric, author_metric, paragraph_metric, keywords_metric,
      average, min, max

    Returns three DataFrames (df_avg, df_min, df_max), each containing
    the first 17 columns plus a new 'predicted_probability' column taken
    respectively from 'average', 'min', and 'max'.
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
    """Rank and aggregate candidate URLs by predicted probability.

    Groups rows by ('name','doi','paragraph'), orders each group
    by descending 'predicted_probability', then aggregates:
      • id                    : first id
      • authors               : first authors
      • field/topic/keywords  : first topic
      • url (ground truth)    : first ground-truth URL
      • candidate_urls        : comma-joined URLs in rank order
      • probability_ranked    : comma-joined probabilities in same order

    Args:
        df: DataFrame with a 'predicted_probability' column.
        output_path: If non-empty, path to CSV for saving the result.

    Returns:
        A DataFrame with columns
        ['id','name','doi','paragraph','authors',
         'field/topic/keywords','url (ground truth)',
         'candidate_urls','probability_ranked'].
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
    """Split DataFrame into three versions using average, min, and max scores.

    Expects `df` to have at least these columns:
      id, name, doi, paragraph, authors,
      field/topic/keywords, url (ground truth),
      candidate_urls, probability (ground truth),
      metadata_*, *_metric for name, author, paragraph, keywords, language,
      average, min, max

    Returns:
        A tuple of three DataFrames (df_avg, df_min, df_max), each with
        all base columns plus:
          'predicted_probability' ← one of 'average', 'min', or 'max'.
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
    """Compute and print classification metrics for a binary match task.

    For each row in `df`, compares the single `candidate_urls` entry
    against the comma-separated ground truth set in `url (ground truth)`
    to form `true_label`.  Uses the `prediction` column as the predicted label.

    Args:
        df: A DataFrame with columns
            - 'candidate_urls': single-URL string per row
            - 'url (ground truth)': comma-separated list of correct URLs
            - 'prediction': 0 or 1 model output

    Prints:
        Precision, recall, F1-score (to 2 decimal places), and
        a full classification report.
    """
    df['true_label'] = [
    int(c in [u.strip() for u in g.split(',')])
    for c, g in zip(df['candidate_urls'], df['url (ground truth)'])
]   
    y_true = df['true_label']
    y_pred = df['prediction']
    
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Precision: {p:.2f}")
    print(f"Recall:    {r:.2f}")
    print(f"F1-score:  {f1:.2f}\n")
    # if you want the full breakdown:
    print(classification_report(y_true, y_pred, target_names=['non-match','match']))
    print()