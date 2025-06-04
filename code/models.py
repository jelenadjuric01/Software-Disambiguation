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
from sklearn.pipeline import Pipeline
from evaluation import evaluation

class MedianImputerWithIndicator(BaseEstimator, TransformerMixin):
    """
Impute missing values using column medians and add binary missing indicators.

This transformer:
- Replaces NaNs with the median value per column
- Adds a new binary column <col>_missing to indicate missingness
- Supports feature name tracking via get_feature_names_out()

Args:
    cols (List[str], optional): List of numeric columns to impute. If None,
        selects all numeric columns at fit time.

Returns:
    pd.DataFrame: Transformed DataFrame with imputed values and indicator columns.
"""

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols
        self.medians_: dict = {}
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if self.cols is None:
            self.cols = X.select_dtypes(include=np.number).columns.tolist()
        self.feature_names_in_ = list(X.columns)
        self.medians_ = {col: X[col].median() for col in self.cols}
        
        # Store output feature names
        missing_cols = [f"{col}_missing" for col in self.cols]
        self.feature_names_out_ = self.feature_names_in_ + missing_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        missing = set(self.cols) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in transform data: {missing}")
        
        for col in self.cols:
            flag_col = f"{col}_missing"
            X[flag_col] = X[col].isna().astype(int)
            X[col] = X[col].fillna(self.medians_[col])
        
        return X[self.feature_names_in_ + [f"{c}_missing" for c in self.cols]]

    def get_feature_names_out(self, input_features=None):
        """Proper scikit-learn implementation"""
        return np.array(self.feature_names_out_)

def make_model(name: str, y_train_fold: np.ndarray, params: dict = None) -> ClassifierMixin:
    """
Create a classification model by name with built-in handling of class imbalance.

Supports model-specific default hyperparameters and class weighting:
- Logistic Regression and Random Forest use `class_weight='balanced'`
- XGBoost uses `scale_pos_weight` based on label distribution
- LightGBM uses dictionary-based class weights
- MLP does not apply any class imbalance correction directly

Args:
    name (str): Model name (case-insensitive). One of:
        ['logistic regression', 'random forest', 'xgboost', 'lightgbm', 'neural net']
    y_train_fold (np.ndarray): Labels used to compute class imbalance adjustments
    params (dict, optional): Dictionary with model-specific override parameters.

Returns:
    ClassifierMixin: An untrained sklearn-compatible classifier.
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
            n_jobs=-1,
            enable_categorical=True  # Add this for newer XGBoost versions
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

def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
Split a DataFrame into stratified train/validation and test sets.

Args:
    df (pd.DataFrame): Input dataset.
    target_col (str): Name of the target column for prediction.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.

Returns:
    Tuple:
        - X_trainval (pd.DataFrame): Features for training and validation
        - X_test (pd.DataFrame): Features for test set
        - y_trainval (pd.Series): Labels for training and validation
        - y_test (pd.Series): Labels for test set
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

def get_preprocessing_pipeline(cols_to_impute: List[str]) -> Pipeline:
    """
Build a preprocessing pipeline for numerical features.

The pipeline includes:
- Median imputation with missing indicators
- Standard scaling

Args:
    cols_to_impute (List[str]): List of column names to impute and scale.

Returns:
    sklearn.Pipeline: A pipeline object with feature name tracking.
"""

    return Pipeline([
        ('imputer', MedianImputerWithIndicator(cols=cols_to_impute)),
        ('scaler', StandardScaler())
    ])

if __name__ == "__main__":
    # 1) Load & split
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.17/model_input.csv")
    X_trainval, X_test, y_trainval, y_test = split_data(df, "true_label", test_size=0.2)
    
    

    # Columns to impute (only these will be processed)
    cols_to_impute = [ 'paragraph_metric','language_metric','synonym_metric','author_metric']
    
    # 2) Prepare the raw tree-based view (no preprocessing)
    X_tree_train = X_trainval.copy()
    X_tree_test = X_test.copy()

    # 3) 5-fold cross-validation on the 80% train+val only:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_to_try = ['Random Forest','XGBoost','LightGBM']

    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []

        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            # Get raw data for this fold
            X_tr_raw = X_trainval.iloc[train_idx]
            X_val_raw = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx]
            y_val = y_trainval.iloc[val_idx]

            # For tree-based models: ensure proper data types
            if name in ("XGBoost", "LightGBM"):
                X_tr = X_tr_raw.copy()
                X_val = X_val_raw.copy()
                
                # Convert to numpy arrays explicitly
                y_tr = y_tr.values
                y_val = y_val.values
                
            else:
                # Preprocessing for other models
                preprocessor = get_preprocessing_pipeline(cols_to_impute)
                X_tr = preprocessor.fit_transform(X_tr_raw)
                X_val = preprocessor.transform(X_val_raw)
                
                feature_names = preprocessor.get_feature_names_out()
                X_tr = pd.DataFrame(X_tr, columns=feature_names)
                X_val = pd.DataFrame(X_val, columns=feature_names)
                
                y_tr = y_tr.values
                y_val = y_val.values

            # Train model
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            
            # Predict
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val)
            y_pred_oof.extend(y_pred)

        # Evaluate CV performance
        cv_df = pd.DataFrame({
            "true_label": y_true_oof,
            "prediction": y_pred_oof
        })
        print(f"\n=== 5-fold CV evaluation for {name} ===")
        evaluation(cv_df)

    # 4) Final training on the entire 80% train+val, then test on held-out 20%
    for name in models_to_try:
        if name in ("XGBoost", "LightGBM"):
            # Use raw data for tree-based models
            X_tr_full = X_tree_train
            X_test_view = X_tree_test
        else:
            # Preprocess for other models
            preprocessor = get_preprocessing_pipeline(cols_to_impute)
            X_tr_full = preprocessor.fit_transform(X_trainval)
            X_test_view = preprocessor.transform(X_test)
            
            # Convert back to DataFrame (optional)
            feature_names = preprocessor.get_feature_names_out()
            X_tr_full = pd.DataFrame(X_tr_full, columns=feature_names)
            X_test_view = pd.DataFrame(X_test_view, columns=feature_names)

        model = make_model(name, y_tr)
        model.fit(X_tr_full, y_trainval)
        y_pred_test = model.predict(X_test_view)
        
        test_df = pd.DataFrame({
            "true_label": y_test.values,
            "prediction": y_pred_test
        })
        print(f"\n=== Final test evaluation for {name} ===")
        evaluation(test_df)