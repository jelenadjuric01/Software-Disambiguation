import pandas as pd
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        'keywords_metric'
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
    Groups rows by 'name', 'doi' and 'paragraph', orders each group by
    'predicted_probability' descending, and aggregates:

      • id                    : first id in the group
      • authors               : first authors in the group
      • field/topic/keywords  : first value in the group
      • url (ground truth)    : first URL in the group
      • candidate_urls        : list of URLs in ranked order
      • probability_ranked    : list of predicted probabilities in the same order

    Returns a DataFrame with columns:
    ['id','name','doi','paragraph','authors',
     'field/topic/keywords','url (ground truth)',
     'candidate_urls','probability_ranked']
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
        'candidate_urls': lambda urls: list(urls),
        'predicted_probability': lambda probs: list(probs)
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



def mrr_at_1(df: pd.DataFrame) -> float:
    """
    Compute MRR@1 over all rows in `df`.
    - truth_col: column holding a set (or list) of correct URLs.
    - cand_col: column holding the model’s ranked list of URLs.
    
    Returns mean reciprocal rank clipped at 1 (i.e. 1 if top‐1 is correct, else 0).
    """
    rr_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        top1 = row["candidate_urls"][0]  # highest‐ranked URL
        rr_scores.append(1.0 if top1 in true_set else 0.0)
    return sum(rr_scores) / len(rr_scores)




def full_mrr(
    df: pd.DataFrame
) -> float:
    """
    Compute full Mean Reciprocal Rank (MRR) over all rows in `df`.
    
    For each row:
      
    We find the position (1-indexed) of the first candidate that appears in the ground-truth set.
    Reciprocal rank = 1/position (or 0 if none match).
    
    Returns the average reciprocal rank across all rows.
    """
    rr_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        position = 0
        for idx, candidate in enumerate(row["candidate_urls"], start=1):
            if candidate in true_set:
                position = idx
                break
        rr_scores.append(1.0 / position if position > 0 else 0.0)

    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0



def r_precision(df: pd.DataFrame) -> float:
    """
    Compute mean R-Precision over all rows in df.
    
    For each row:
      - R = len(truth_set)
      - R-Precision = (# of truth URLs in top-R candidates) / R
    
    Returns the average R-Precision across all rows.
    """
    rp_scores = []
    for _, row in df.iterrows():
        true_set = set(row["url (ground truth)"].split(","))
        R = len(true_set)
        if R == 0:
            # if you have no ground truth for a mention, you may choose to skip it
            continue
        top_R = row["candidate_urls"][:R]
        hits = len(set(top_R) & true_set)
        rp_scores.append(hits / R)
    return sum(rp_scores) / len(rp_scores) if rp_scores else 0.0


def evaluation(df: pd.DataFrame) -> None:
    """
    Evaluates the model's predictions in `df` and prints it.
    Has MRR@1, Top-1 accuracy, R-Precision, and classification metrics.
    """
    
    # Compute metrics
    mrr = mrr_at_1(df)
    full_mrr_score = full_mrr(df)

    r_prec = r_precision(df)

    
    # Print results
    print(f"MRR@1: {mrr:.4f}")
    print(f"R-Precision: {r_prec:.4f}")
    print(f"Full MRR: {full_mrr_score:.4f}")
   