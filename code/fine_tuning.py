# fine_tuning.py

import argparse
import os
import csv
import json
from pathlib import Path

from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from models import make_model, split_data, get_preprocessing_pipeline


def get_param_grids():
    """
    Base hyperparameter grids for each model.
    For Random Forest, keys will be prefixed in the tuning loop to match Pipeline syntax.
    """
    return {
        "Random Forest": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [100, 200, 500],
            "num_leaves": [31, 50, 100],
            "learning_rate": [0.01, 0.1, 0.2],
            "min_child_samples": [20, 50, 100],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning: RF with in-fold preprocessing; bare XGB/LGBM"
    )
    parser.add_argument(
        "--models", "-m", nargs="+", required=True,
        choices=["Random Forest", "XGBoost", "LightGBM"],
        help="Which models to tune"
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="CSV path with features + label"
    )
    parser.add_argument(
        "--label-col", "-l", type=str, default="true_label",
        help="Name of the target column"
    )
    parser.add_argument(
        "--cols-to-impute", "-c", nargs="+", default=None,
        help="Numeric cols to median-impute & scale (required for RF)"
    )
    parser.add_argument(
        "--scoring", "-s", type=str, default="f1",
        help="CV scoring metric"
    )
    args = parser.parse_args()

    if "Random Forest" in args.models and not args.cols_to_impute:
        parser.error("--cols-to-impute required for Random Forest tuning")

    # Extract version from folder name
    version = Path(args.data).parent.name.lstrip('v')

    # Load data and stratified split
    df = pd.read_csv(args.data)
    X_trainval_df, X_test_df, y_trainval, y_test = split_data(
        df, args.label_col, test_size=0.2
    )

    # Prepare RF preprocessing
    if "Random Forest" in args.models:
        rf_preproc = get_preprocessing_pipeline(args.cols_to_impute)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_grids = get_param_grids()

    # Setup results CSV
    results_file = 'tuning_results.csv'
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['version', 'model', 'best_params', 'precision', 'recall', 'f1']
            )

    # Loop over models
    for name in args.models:
        print(f"\n>>> Tuning {name} <<<")

        # Select estimator and param grid
        if name == "Random Forest":
            # Pipeline with preprocessing
            estimator = Pipeline([
                ('preproc', rf_preproc),
                ('model', make_model(name, y_trainval.values))
            ])
            # prefix grid keys
            param_grid = {f"model__{k}": v for k, v in base_grids[name].items()}

        else:
            # bare classifier
            estimator = make_model(name, y_trainval.values)
            # use grid keys directly
            param_grid = base_grids[name]

        # Run GridSearchCV
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring=args.scoring,
            n_jobs=-1,
            verbose=2,
            return_train_score=True,
        )
        gs.fit(X_trainval_df, y_trainval.values)

        best_est = gs.best_estimator_
        best_params = gs.best_params_
        print(f"→ Best params for {name}: {best_params}")
        print(f"→ CV {args.scoring}: {gs.best_score_:.4f}")

        # Final test evaluation
        best_est.fit(X_trainval_df, y_trainval.values)
        preds = best_est.predict(X_test_df)
        precision = precision_score(y_test.values, preds)
        recall = recall_score(y_test.values, preds)
        f1 = f1_score(y_test.values, preds)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Append results
        with open(results_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                version, name, json.dumps(best_params),
                precision, recall, f1
            ])

    print("\nAll tuning runs complete. See tuning_results.csv for details.")


if __name__ == '__main__':
    main()
