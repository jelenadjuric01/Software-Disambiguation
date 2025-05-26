import joblib  # ADD THIS AT THE TOP
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
from models import make_model, get_preprocessing_pipeline, split_data
if __name__ == "__main__":
    # 1) Load & split
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.12/model_input_no_keywords.csv")
    X_trainval, X_test, y_trainval, y_test = split_data(df, "true_label", test_size=0.2)

    # Columns to impute (only these will be processed)
    cols_to_impute = ['paragraph_metric', 'language_metric', 'synonym_metric', 'author_metric']

    # 2) Cross-validation on 80% trainval
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    name = 'Random Forest'
    y_true_oof, y_pred_oof = [], []

    for train_idx, val_idx in cv.split(X_trainval, y_trainval):
        X_tr_raw = X_trainval.iloc[train_idx]
        X_val_raw = X_trainval.iloc[val_idx]
        y_tr = y_trainval.iloc[train_idx]
        y_val = y_trainval.iloc[val_idx]

        preprocessor = get_preprocessing_pipeline(cols_to_impute)
        X_tr = preprocessor.fit_transform(X_tr_raw)
        X_val = preprocessor.transform(X_val_raw)

        model = make_model(name, y_tr)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_true_oof.extend(y_val)
        y_pred_oof.extend(y_pred)

    cv_df = pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof})
    print(f"\n=== 5-fold CV evaluation for {name} ===")
    evaluation(cv_df)

    # 3) Final model training on full 80%, and test on 20%
    preprocessor = get_preprocessing_pipeline(cols_to_impute)
    model = make_model(name, y_trainval.values)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_trainval, y_trainval)
    y_pred_test = pipeline.predict(X_test)

    test_df = pd.DataFrame({
        "true_label": y_test.values,
        "prediction": y_pred_test
    })
    print(f"\n=== Final test evaluation for {name} ===")
    evaluation(test_df)

    # 4) Save model
    joblib.dump(pipeline, "model_pipeline.joblib")
    print("✅ Model saved to model_pipeline.joblib")
