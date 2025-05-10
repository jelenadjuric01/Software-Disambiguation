import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from evaluation import evaluation
from models import (
    split_data,
    make_model,
    apply_imputer,
    apply_scaler
)

if __name__ == "__main__":
    # 1) Load & split off 20% hold-out for final test
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/model_input.csv")
    X_trainval_raw, X_test_raw, y_trainval, y_test = split_data(
        df, "true_label", test_size=0.2, random_state=42
    )
    # convert to numpy for indexing
    y_trainval = y_trainval.values
    y_test = y_test.values

    # 2) Impute & scale using helper functions
    X_imp_train, imputer = apply_imputer(
        X_trainval_raw,
        cols=['author_metric','paragraph_metric','keywords_metric','language_metric']
    )
    X_imp_test = imputer.transform(X_test_raw)

    X_dense_train_df, scaler = apply_scaler(X_imp_train)
    X_dense_test_df, _ = apply_scaler(X_imp_test, scaler=scaler)

    # Convert to numpy arrays for CV slicing
    X_dense_train = X_dense_train_df.values
    X_dense_test = X_dense_test_df.values

    # 3a) Univariate selection (ANOVA F-test)
    k = 3
    uni = SelectKBest(score_func=f_classif, k=k)
    X_uni_train = uni.fit_transform(X_dense_train, y_trainval)
    X_uni_test = uni.transform(X_dense_test)

    # Report univariate picks
    orig_feats = X_imp_train.columns.tolist()
    scores = uni.scores_
    mask = uni.get_support()
    selected = [f for f, keep in zip(orig_feats, mask) if keep]
    selected_scores = [s for s, keep in zip(scores, mask) if keep]
    print(f"\nUnivariate feature selection (k={k}): picked {len(selected)} features")
    for feat, sc in sorted(zip(selected, selected_scores), key=lambda x: -x[1]):
        print(f"  • {feat:<20s}  F-score = {sc:.2f}")

    # 3b) Multivariate importance with Random Forest on dense features
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_dense_train, y_trainval)
    importances = rf.feature_importances_
    feat_importances = sorted(zip(orig_feats, importances), key=lambda x: -x[1])
    print("\nMultivariate feature importance (Random Forest):")
    for feat, imp_val in feat_importances:
        print(f"  • {feat:<20s} importance = {imp_val:.4f}")

    # 4) Assemble feature sets (all numpy arrays)
    feature_sets = {
        "All Features":       (X_dense_train,  X_dense_test),
        f"Univariate (k={k})": (X_uni_train,    X_uni_test),
        "Multivariate_RF":     (X_dense_train,  X_dense_test)
    }

    models_to_try = [
        "Logistic Regression", "Random Forest", "XGBoost",
        "LightGBM", "Neural Net"
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5) Cross-validated evaluation on 80%
    for feat_name, (X_tr_full, _) in feature_sets.items():
        print(f"\n=== PIPELINE: {feat_name} ===")
        for model_name in models_to_try:
            y_true_oof, y_pred_oof = [], []
            for train_idx, val_idx in cv.split(X_tr_full, y_trainval):
                # slice numpy arrays for rows
                X_tr, X_val = X_tr_full[train_idx], X_tr_full[val_idx]
                y_tr, y_val = y_trainval[train_idx], y_trainval[val_idx]
                model = make_model(model_name, y_tr)
                if model_name == "Neural Net":
                    # oversample minority
                    X_pos = X_tr[y_tr == 1]; y_pos = y_tr[y_tr == 1]
                    X_neg = X_tr[y_tr == 0]; y_neg = y_tr[y_tr == 0]
                    X_pos_up, y_pos_up = resample(
                        X_pos, y_pos,
                        replace=True, n_samples=len(y_neg), random_state=42
                    )
                    X_tr_bal = np.vstack([X_neg, X_pos_up])
                    y_tr_bal = np.hstack([y_neg, y_pos_up])
                    model.fit(X_tr_bal, y_tr_bal)
                else:
                    model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                y_true_oof.extend(y_val);
                y_pred_oof.extend(y_pred)
            print(f"\n--- 5-fold CV: {model_name} on {feat_name} ---")
            oof_df = pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof})
            evaluation(oof_df)

    # 6) Final train on all 80% & evaluate on held-out 20%
    for feat_name, (X_tr_full, X_te_full) in feature_sets.items():
        print(f"\n=== FINAL TEST: {feat_name} ===")
        for model_name in models_to_try:
            model = make_model(model_name, y_trainval)
            if model_name == "Neural Net":
                X_pos = X_tr_full[y_trainval == 1]; y_pos = y_trainval[y_trainval == 1]
                X_neg = X_tr_full[y_trainval == 0]; y_neg = y_trainval[y_trainval == 0]
                X_pos_up, y_pos_up = resample(
                    X_pos, y_pos,
                    replace=True, n_samples=len(y_neg), random_state=42
                )
                X_tr_bal = np.vstack([X_neg, X_pos_up])
                y_tr_bal = np.hstack([y_neg, y_pos_up])
                model.fit(X_tr_bal, y_tr_bal)
            else:
                model.fit(X_tr_full, y_trainval)
            y_test_pred = model.predict(X_te_full)
            test_df = pd.DataFrame({"true_label": y_test, "prediction": y_test_pred})
            print(f"\n--- {model_name} on {feat_name} (held-out 20%) ---")
            evaluation(test_df)
