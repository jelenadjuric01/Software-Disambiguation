import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from models import (
    split_data,
    apply_imputer,
    apply_scaler,
    make_model
)
from evaluation import evaluation





def evaluate_models(models_to_try: list,
                    X_train_dense: pd.DataFrame,
                    X_test_dense: pd.DataFrame,
                    X_train_tree: pd.DataFrame,
                    X_test_tree: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    description: str
                   ) -> None:
    """
    Exactly your 5-fold CV + final-test evaluation on each model.
    Uses tree-view for XGBoost/LightGBM and dense-view otherwise.
    """
    print(f"\n==================== {description} ====================")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1) 5-fold CV on train+val
    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []
        for train_idx, val_idx in cv.split(X_train_dense, y_train):
            if name.lower() in ("xgboost", "lightgbm"):
                X_tr = X_train_tree.iloc[train_idx]
                X_val = X_train_tree.iloc[val_idx]
            else:
                X_tr = X_train_dense.iloc[train_idx]
                X_val = X_train_dense.iloc[val_idx]

            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            y_true_oof.extend(y_val)
            y_pred_oof.extend(y_pred)

        cv_df = pd.DataFrame({
            "true_label": y_true_oof,
            "prediction": y_pred_oof
        })
        print(f"\n--- 5-fold CV results for {name} ---")
        evaluation(cv_df)

    # 2) Final test on held-out 20%
    for name in models_to_try:
        if name.lower() in ("xgboost", "lightgbm"):
            X_tr_full, X_test_view = X_train_tree, X_test_tree
        else:
            X_tr_full, X_test_view = X_train_dense, X_test_dense

        model = make_model(name, y_train)
        model.fit(X_tr_full, y_train)
        y_pred_test = model.predict(X_test_view)

        test_df = pd.DataFrame({
            "true_label": y_test.values,
            "prediction": y_pred_test
        })
        print(f"\n--- Final test results for {name} ---")
        evaluation(test_df)


if __name__ == "__main__":
    # 1) Load & split
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3/model_input.csv")
    X_trainval, X_test, y_trainval, y_test = split_data(df, "true_label", test_size=0.2)

    # 2) Impute & scale to get dense view
    X_imp_train, imputer       = apply_imputer(X_trainval,['author_metric','paragraph_metric','keywords_metric','language_metric'])
    X_dense_train, scaler = apply_scaler(X_imp_train)

    X_imp_test = imputer.transform(X_test)
    X_dense_test, _    = apply_scaler(X_imp_test, scaler=scaler)

    # 3) Tree view is the raw (unscaled) metrics
    X_tree_train = X_trainval.copy()
    X_tree_test  = X_test.copy()

    models_to_try = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Neural Net"
    ]

    # —— A) Baseline with ALL features ——
    evaluate_models(models_to_try,
                    X_dense_train, X_dense_test,
                    X_tree_train,  X_tree_test,
                    y_trainval,    y_test,
                    description="All features (baseline)")

    # —— B) Univariate selection ——
    # pick top-k from the DENSE (imputed+scaled) features:
    k=3
    selector_uni = SelectKBest(score_func=f_classif, k=k)
    selector_uni.fit(X_dense_train, y_trainval)
    uni_scores = pd.Series(selector_uni.scores_, index=X_dense_train.columns)
    selected_feats = list(X_dense_train.columns[selector_uni.get_support()])
    print(f"\n--- Univariate selection (top {k} features) ---")
    print(uni_scores.sort_values(ascending=False).head(k).to_string(), "\n")
    # build DataFrames of just those:
    X_uni_train = pd.DataFrame(
        selector_uni.transform(X_dense_train),
        columns=selected_feats,
        index=X_dense_train.index
    )
    X_uni_test = pd.DataFrame(
        selector_uni.transform(X_dense_test),
        columns=selected_feats,
        index=X_dense_test.index
    )
    # also apply to tree view (for XGBoost/LightGBM):
    X_tree_uni_train = X_tree_train[selected_feats]
    X_tree_uni_test  = X_tree_test[selected_feats]

    evaluate_models(models_to_try,
                    X_uni_train,        X_uni_test,
                    X_tree_uni_train,   X_tree_uni_test,
                    y_trainval,         y_test,
                    description="Univariate-selected features")

    # —— C) Multivariate extraction (PCA) ——
    variance_ratio=0.95
    pca = PCA(n_components=variance_ratio, svd_solver="full", random_state=42)
    X_pca_train = pca.fit_transform(X_dense_train)
    n_comp = pca.n_components_
    comp_names = [f"PC{i+1}" for i in range(n_comp)]
    X_pca_df = pd.DataFrame(X_pca_train, columns=comp_names, index=X_dense_train.index)

    print(f"\n--- PCA extraction ({n_comp} components explaining "
          f"{pca.explained_variance_ratio_.sum():.2%} variance) ---")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {var:.2%}")
    print()
   
    X_pca_test = pd.DataFrame(
        pca.transform(X_dense_test),
        columns=X_pca_train.columns,
        index=X_dense_test.index
    )
    # for tree models we’ll also feed them the PCs:
    evaluate_models(models_to_try,
                    X_pca_train, X_pca_test,
                    X_pca_train, X_pca_test,
                    y_trainval,  y_test,
                    description="PCA-extracted components")
