import re
from urllib.parse import urlparse, urlunparse
import cloudpickle
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


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

def canonical_url(u: str) -> str:
    """
    Turn a URL into a canonical form so that different CRAN URLs
    (and trivial variations) compare equal.
    """
    if not isinstance(u, str) or not u.strip():
        return ''
    u = u.strip()
    u_low = u.lower()
    # Handle CRAN package URLs in either form
    m = re.search(r'cran\.r-project\.org/web/packages/([^/]+)/', u_low)
    if m:
        pkg = m.group(1)
        return f'https://cran.r-project.org/web/packages/{pkg}'
    m = re.search(r'cran\.r-project\.org/package=([^&/]+)', u_low)
    if m:
        pkg = m.group(1)
        return f'https://cran.r-project.org/web/packages/{pkg}'
    # Fallback normalization: https, lowercase host, drop query/frag, no trailing slash
    p = urlparse(u)
    scheme = 'https'
    netloc = p.netloc.lower()
    path = p.path.rstrip('/')
    return urlunparse((scheme, netloc, path, '', '', ''))

def label_true(row: pd.Series) -> int:
    """
    Given a row with:
      - row['ground_truth']    : one or more correct URLs (string or list)
      - row['candidate_urls']  : one or more candidate URLs (string or list)
    Return 1 if ANY canonical candidate URL appears in the set of
    canonical ground-truth URLs, else 0.
    """
    # --- Build canonical ground-truth set ---
    gt_raw = row.get('ground truth')
    if isinstance(gt_raw, list):
        gt_items = gt_raw
    else:
        gt_items = str(gt_raw).split(',')
    gt_set = {canonical_url(u) for u in gt_items if u and u.strip()}

    # --- Build canonical candidate set ---
    cand_raw = row.get('candidate_urls')
    if isinstance(cand_raw, list):
        cand_items = cand_raw
    else:
        cand_items = str(cand_raw).split(',')
    cand_set = {canonical_url(u) for u in cand_items if u and u.strip()}

    # --- Label = 1 if intersection non-empty ---
    return int(bool(gt_set & cand_set))


def enrich_with_ground_truth(
    df_target: pd.DataFrame,
    df_lookup: pd.DataFrame,
    name_col: str = "name",
    doi_col: str = "doi",
    paragraph_col: str = "paragraph",
    gt_col: str = "ground truth",
    how: str = "left"
) -> pd.DataFrame:
    """
    Enrich df_target by pulling in the ground_truth column from df_lookup,
    matching on (name, doi, paragraph).

    Parameters
    ----------
    df_target : pd.DataFrame
        DataFrame missing the ground_truth column.
    df_lookup : pd.DataFrame
        DataFrame containing the ground_truth column.
    name_col : str
        Column name for 'name' in both frames.
    doi_col : str
        Column name for 'doi' in both frames.
    paragraph_col : str
        Column name for 'paragraph' in both frames.
    gt_col : str
        Column name for the ground truth in df_lookup (and to create in df_target).
    how : str
        Merge method (‘left’, ‘inner’, etc.); default ‘left’ keeps all rows of df_target.

    Returns
    -------
    pd.DataFrame
        A new DataFrame like df_target but with a `ground_truth` column added.
    """

    # sanity checks
    for c in (name_col, doi_col, paragraph_col):
        if c not in df_target.columns:
            raise KeyError(f"Target DataFrame is missing column '{c}'")
        if c not in df_lookup.columns:
            raise KeyError(f"Lookup DataFrame is missing column '{c}'")
    if gt_col not in df_lookup.columns:
        raise KeyError(f"Lookup DataFrame is missing ground truth column '{gt_col}'")

    # drop duplicate key-ground_truth pairs so we don’t multiply rows
    lookup_unique = (
        df_lookup[[name_col, doi_col, paragraph_col, gt_col]]
        .drop_duplicates(subset=[name_col, doi_col, paragraph_col])
    )

    # perform the merge
    merged = df_target.merge(
        lookup_unique,
        on=[name_col, doi_col, paragraph_col],
        how=how
    )

    return merged


if __name__ == "__main__":
    df = pd.read_csv('temp/temp/similarities.csv')
    df_lookup = pd.read_csv('CZI_test.csv')
    df = enrich_with_ground_truth(df, df_lookup)
    df['true_label'] = df.apply(label_true, axis=1)

    evaluation(df)
