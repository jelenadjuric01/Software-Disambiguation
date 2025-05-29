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
    
if __name__ == "__main__":
    import pandas as pd

# 1. Load the workbook (adjust path if needed)
    df = pd.read_excel('D:\MASTER\TMF\Software-Disambiguation\corpus\corpus_v3_2.xlsx')

    # 2. Randomly sample 20% of the rows
    #    Setting a random_state ensures reproducibility; change or remove for different draws
    sampled = df.sample(frac=0.2, random_state=42)

    # 3. (Optional) Inspect the first few sampled rows
    print(sampled.head())

    # 4. (Optional) Save your sample to a new Excel file
    sampled.to_excel('corpus_v3_sampled20.xlsx', index=False)
