from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import ClassifierMixin
from evaluation import evaluation


class MedianImputerWithIndicator(BaseEstimator, TransformerMixin):
    """
    A scikit-learn–style transformer that:
      - Computes & stores the median of each selected column on fit()
      - On transform():
         * Creates a "<col>_missing" indicator (1 if was NaN, else 0)
         * Fills NaNs in each column with the stored median
      - Always returns the same columns in the same order.
    """
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols  # list of cols to impute; if None, auto-detect numeric on fit
        # Will be filled in fit():
        self.medians_: dict = {}
        self.feature_names_in_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        # Determine which cols to operate on
        if self.cols is None:
            self.cols = X.select_dtypes(include=np.number).columns.tolist()
        # Remember original feature order
        self.feature_names_in_ = list(X.columns)
        # Compute medians
        self.medians_ = {col: X[col].median() for col in self.cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # sanity check
        missing = set(self.cols) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in transform data: {missing}")
        # Apply: for each col, make flag then fill
        for col in self.cols:
            flag_col = f"{col}_missing"
            # 1 where original was NaN
            X[flag_col] = X[col].isna().astype(int)
            # fill with stored median
            X[col] = X[col].fillna(self.medians_[col])
        # Reorder columns:
        # 1) all original cols (with NaNs now filled)
        # 2) all the new "<col>_missing" flags, in the same order
        missing_cols = [f"{col}_missing" for col in self.cols]
        return X[self.feature_names_in_ + missing_cols]

    def get_feature_names_out(self):
        # Optional: for compatibility with newer sklearn
        return np.array(self.feature_names_in_ + [f"{c}_missing" for c in self.cols])
# your imputer class (assumed imported or defined above)
# from your_module import MedianImputerWithIndicator

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train+val and final test sets, stratified on the target.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    return X_trainval, X_test, y_trainval, y_test


def apply_imputer(
    X: pd.DataFrame,
    cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, "MedianImputerWithIndicator"]:
    """
    Fit MedianImputerWithIndicator on X[cols] (or all numeric if None),
    transform X and return (X_imputed, fitted_imputer).
    """
    imputer = MedianImputerWithIndicator(cols=cols)
    X_imp = imputer.fit_transform(X)
    return X_imp, imputer


def plot_and_save_correlation_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    annot: bool = True,
    fmt: str = ".2f",
    filename: str = "correlation_matrix.png",
    figsize: tuple = (10, 8)
) -> None:
    """
    Compute a correlation matrix for df[cols] (or all numeric if cols is None),
    plot it as a heatmap with annotations, and save to filename.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str, optional
        Which columns to include; if None, uses all numeric columns.
    vmin : float
        Minimum value for color scale (default -1).
    vmax : float
        Maximum value for color scale (default +1).
    cmap : str
        Matplotlib colormap name.
    annot : bool
        Whether to write the correlation coefficients on the heatmap.
    fmt : str
        String format for annotations.
    filename : str
        Path (including name) where to save the plot.
    figsize : tuple
        Figure size in inches.
    """
    # Select columns
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    data = df[cols]

    # Compute correlation matrix
    corr = data.corr()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(corr.values, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    # Annotate
    if annot:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(
                    j, i,
                    format(corr.iat[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if abs(corr.iat[i, j]) > 0.5 else "black"
                )

    ax.set_title("Correlation Matrix")
    fig.tight_layout()

    # Save and close
    fig.savefig(filename)
    plt.close(fig)


def apply_scaler(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fit or reuse a StandardScaler on df[cols], return (scaled_df, scaler).
    """
    df_out = df.copy()
    if cols is None:
        cols = df_out.select_dtypes(include=np.number).columns.tolist()

    if scaler is None:
        scaler = StandardScaler()
        df_out[cols] = scaler.fit_transform(df_out[cols])
    else:
        df_out[cols] = scaler.transform(df_out[cols])

    return df_out, scaler

def make_model(name: str, y_train_fold: np.ndarray, params: dict = None) -> ClassifierMixin:
    """
    Factory to build a classifier by name with built-in class-imbalance handling.

    Args:
        name:           one of
                        ['logistic regression','random forest','xgboost',
                         'lightgbm','neural net'] (case-insensitive)
        y_train_fold:   1d array of training labels (for computing class weights)
        params:         optional dict mapping model-name → dict of override kwargs

    Returns:
        An unfit sklearn-compatible estimator ready for .fit().
    """
    if params is None:
        params = {}
    key = name.strip().lower()

    if key == "logistic regression":
        defaults = dict(solver="liblinear", class_weight="balanced", random_state=42)
        return LogisticRegression(**{**defaults, **params.get(key, {})})

    elif key == "random forest":
        defaults = dict(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
        return RandomForestClassifier(**{**defaults, **params.get(key, {})})

    elif key == "xgboost":
        neg, pos = np.bincount(y_train_fold)
        defaults = dict(
            n_estimators=100,
            eval_metric="logloss",
            scale_pos_weight=neg/pos,
            random_state=42,
            n_jobs=-1
        )
        return XGBClassifier(**{**defaults, **params.get(key, {})})

    elif key == "lightgbm":
    # compute class‐weight array and map it to labels
        classes = np.unique(y_train_fold)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train_fold
        )
        cw = dict(zip(classes, weights))

        defaults = dict(
            n_estimators=100,
            class_weight=cw,
            random_state=42,
            n_jobs=-1
        )
        return LGBMClassifier(**{**defaults, **params.get(key, {})})

    elif key in ("neural net","mlp","mlpclassifier"):
        defaults = dict(
            hidden_layer_sizes=(50,30,10),
            activation="relu",    
            solver="adam",        
            max_iter=200,
            random_state=42
        )
        return MLPClassifier(**{**defaults, **params.get(key, {})})

    else:
        valid = ["Logistic Regression","Random Forest","XGBoost","LightGBM","Neural Net"]
        raise ValueError(f"Unknown model name '{name}'. Valid options: {valid}")

if __name__ == "__main__":
    # 1) Load & split
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/model_input.csv")
    X_trainval, X_test, y_trainval, y_test = split_data(df, "true_label", test_size=0.2)

    # 2) Prepare the two “views”
    X_tree_train = X_trainval.copy()
    X_tree_test  = X_test.copy()

    X_imp_train, imputer       = apply_imputer(X_trainval,['author_metric','paragraph_metric','keywords_metric','language_metric'])
    X_dense_train, scaler      = apply_scaler(X_imp_train)

    X_imp_test, _  = imputer.transform(X_test), None
    X_dense_test, _ = apply_scaler(X_imp_test, scaler=scaler)

    # 3) Plot & save the correlation matrix for the dense features
    #plot_and_save_correlation_matrix(
    #    df=X_dense_train,
    #    filename="dense_correlation_matrix.png",
    #    figsize=(12,10),
    #    cmap="coolwarm"
    #)
    print("Saved dense feature correlation matrix to dense_correlation_matrix.png\n")

    # 4) Train each model on the full train+val set and evaluate on test
    models_to_try = [
        'Logistic Regression',"Random Forest","XGBoost",
        "LightGBM","Neural Net"
    ]
    # 4) 5-fold cross-validation on the 80% train+val only:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []

        for train_idx, val_idx in cv.split(X_dense_train, y_trainval):
            # select the proper “view” for this model
            if name in ("XGBoost", "LightGBM"):
                X_tr = X_tree_train.iloc[train_idx]
                X_val = X_tree_train.iloc[val_idx]
            else:
                X_tr = X_dense_train.iloc[train_idx]
                X_val = X_dense_train.iloc[val_idx]

            y_tr = y_trainval.iloc[train_idx]
            y_val = y_trainval.iloc[val_idx]

            # instantiate and fit on this fold of the 80% train+val data
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)

            # predict on the validation fold (still within the 80%)
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val)
            y_pred_oof.extend(y_pred)

        # aggregate out-of-fold results and evaluate
        cv_df = pd.DataFrame({
            "true_label":  y_true_oof,
            "prediction":  y_pred_oof
        })
        print(f"\n=== 5-fold CV evaluation for {name} ===")
        evaluation(cv_df)


    # 5) Final training on the entire 80% train+val, then test on the held-out 20%
    for name in models_to_try:
        # choose view for final training/prediction
        if name in ("XGBoost", "LightGBM"):
            X_tr_full, X_test_view = X_tree_train, X_tree_test
        else:
            X_tr_full, X_test_view = X_dense_train, X_dense_test

        # instantiate and train ONLY on the 80% (never use X_test for fitting)
        model = make_model(name, y_trainval)
        model.fit(X_tr_full, y_trainval)

        # now predict on the held-out 20% test set
        y_pred_test = model.predict(X_test_view)
        test_df = pd.DataFrame({
            "true_label":  y_test.values,
            "prediction":  y_pred_test
        })

        print(f"\n=== Final test evaluation for {name} ===")
        evaluation(test_df)
