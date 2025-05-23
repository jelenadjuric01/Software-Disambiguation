import pandas as pd
from typing import Tuple
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

    