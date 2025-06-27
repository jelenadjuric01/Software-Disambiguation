import re
from urllib.parse import urlparse, urlunparse
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



if __name__ == "__main__":
    df = pd.read_csv('temp/temp/similarities.csv',delimiter=';')
    df['true_label'] = df.apply(label_true, axis=1)
    df.to_csv('similarities.csv', index=False)
    evaluation(df)

