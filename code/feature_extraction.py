import itertools
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import  StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from evaluation import evaluation
from models import MedianImputerWithIndicator, make_model, split_data, get_preprocessing_pipeline

"""
Script for evaluating classification models with baseline and selected features.

Performs:
- Baseline training with all features
- Univariate feature selection with ANOVA F-score
- Multivariate selection using RandomForest importance
- Cross-validation and test set evaluation for multiple models

Models used: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Net
"""

if __name__ == "__main__":
    # Load & split
    df = pd.read_csv("D:/MASTER/TMF/Software-Disambiguation/corpus/temp/v3.8/model_input.csv")
    X_trainval, X_test, y_trainval, y_test = split_data(df, "true_label", test_size=0.2)
    cols_to_impute = [ 'paragraph_metric','language_metric','synonym_metric','author_metric']
    X_tree_train = X_trainval.copy()
    X_tree_test = X_test.copy()

    models_to_try = ['Random Forest', 'XGBoost', 'LightGBM']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
    # 1) Baseline: CV & test evaluation with all features
    print("\n### Baseline 5-fold CV (all features) ###")
    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []
        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            X_tr_raw = X_trainval.iloc[train_idx]; X_val_raw = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx].values; y_val = y_trainval.iloc[val_idx].values
            if name in ("XGBoost","LightGBM"):
                X_tr, X_val = X_tr_raw.copy(), X_val_raw.copy()
            else:
                preproc = get_preprocessing_pipeline(cols_to_impute)
                X_tr = pd.DataFrame(preproc.fit_transform(X_tr_raw),
                                     columns=preproc.get_feature_names_out())
                X_val = pd.DataFrame(preproc.transform(X_val_raw),
                                      columns=preproc.get_feature_names_out())
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val); y_pred_oof.extend(y_pred)
        print(f"\n--- CV results for {name} ---")
        evaluation(pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof}))

    # Final test on held-out 20%
    print("\n### Final test evaluation (all features) ###")
    for name in models_to_try:
        if name in ("XGBoost","LightGBM"):
            X_tr_full, X_te_full = X_tree_train, X_tree_test
        else:
            preproc = get_preprocessing_pipeline(cols_to_impute)
            X_tr_full = pd.DataFrame(preproc.fit_transform(X_trainval),
                                    columns=preproc.get_feature_names_out())
            X_te_full = pd.DataFrame(preproc.transform(X_test),
                                    columns=preproc.get_feature_names_out())
        model = make_model(name, y_trainval.values)
        model.fit(X_tr_full, y_trainval.values)
        y_pred_test = model.predict(X_te_full)
        print(f"\n--- Test results for {name} ---")
        evaluation(pd.DataFrame({"true_label": y_test.values, "prediction": y_pred_test}))

    # 2) Feature Selection
    # Impute full train+val
    imputer = MedianImputerWithIndicator(cols=cols_to_impute)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_trainval), columns=imputer.get_feature_names_out())
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=imputer.get_feature_names_out())
    all_feats = X_train_imp.columns.tolist()

    # 2a) Univariate selection
    k = 3
    uni = SelectKBest(score_func=f_classif, k=k)
    uni.fit(X_train_imp, y_trainval.values)
    uni_scores = pd.DataFrame({'feature': all_feats, 'score': uni.scores_, 'pvalue': uni.pvalues_})
    uni_scores = uni_scores.sort_values(by='score', ascending=False)
    print("\n### Univariate Feature Scores ###")
    print(uni_scores)
    top_uni = uni_scores['feature'].iloc[:k].tolist()
    print(f"\nSelected top {k} univariate features: {top_uni}")

    # CV with univariate-selected features
    print("\n### Univariate CV (selected features) ###")
    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []
        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            X_tr_raw = X_trainval.iloc[train_idx]; X_val_raw = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx].values; y_val = y_trainval.iloc[val_idx].values
            if name in ("XGBoost","LightGBM"):
                X_tr, X_val = X_tr_raw[top_uni].copy(), X_val_raw[top_uni].copy()
            else:
                imp_fold = MedianImputerWithIndicator(cols=cols_to_impute)
                X_tr_imp = pd.DataFrame(imp_fold.fit_transform(X_tr_raw), columns=imp_fold.get_feature_names_out())
                X_val_imp = pd.DataFrame(imp_fold.transform(X_val_raw), columns=imp_fold.get_feature_names_out())
                X_tr_sel = X_tr_imp[top_uni]; X_val_sel = X_val_imp[top_uni]
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=top_uni)
                X_val = pd.DataFrame(scaler.transform(X_val_sel), columns=top_uni)
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val); y_pred_oof.extend(y_pred)
        print(f"\n--- CV results for {name} (univariate) ---")
        evaluation(pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof}))

    # Test with univariate-selected features
    print("\n### Test Evaluation (univariate-selected) ###")
    for name in models_to_try:
        if name in ("XGBoost","LightGBM"):
            X_tr_sel, X_te_sel = X_trainval[top_uni], X_test[top_uni]
        else:
            X_tr_sel = X_train_imp[top_uni]; X_te_sel = X_test_imp[top_uni]
            scaler = StandardScaler()
            X_tr_sel = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=top_uni)
            X_te_sel = pd.DataFrame(scaler.transform(X_te_sel), columns=top_uni)
        model = make_model(name, y_trainval.values)
        model.fit(X_tr_sel, y_trainval.values)
        y_pred = model.predict(X_te_sel)
        print(f"\n--- Test results for {name} (univariate) ---")
        evaluation(pd.DataFrame({"true_label": y_test.values, "prediction": y_pred}))

    # 2b) Multivariate (model-based) selection
    base = RandomForestClassifier(n_estimators=100, random_state=42)
    base.fit(X_train_imp, y_trainval.values)
    msel = SelectFromModel(base, threshold='median')
    msel.fit(X_train_imp, y_trainval.values)
    imp_df = pd.DataFrame({'feature': all_feats, 'importance': base.feature_importances_})
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    sel_mtv = imp_df.loc[msel.get_support(), 'feature'].tolist()
    print("\n### Multivariate Feature Importances ###")
    print(imp_df)
    print(f"\nSelected multivariate features: {sel_mtv}")

    # CV with multivariate-selected features
    print("\n### Multivariate CV (selected features) ###")
    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []
        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            X_tr_raw = X_trainval.iloc[train_idx]; X_val_raw = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx].values; y_val = y_trainval.iloc[val_idx].values
            if name in ("XGBoost","LightGBM"):
                X_tr, X_val = X_tr_raw[sel_mtv].copy(), X_val_raw[sel_mtv].copy()
            else:
                imp_fold = MedianImputerWithIndicator(cols=cols_to_impute)
                X_tr_imp = pd.DataFrame(imp_fold.fit_transform(X_tr_raw), columns=imp_fold.get_feature_names_out())
                X_val_imp = pd.DataFrame(imp_fold.transform(X_val_raw), columns=imp_fold.get_feature_names_out())
                X_tr_sel = X_tr_imp[sel_mtv]; X_val_sel = X_val_imp[sel_mtv]
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=sel_mtv)
                X_val = pd.DataFrame(scaler.transform(X_val_sel), columns=sel_mtv)
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val); y_pred_oof.extend(y_pred)
        print(f"\n--- CV results for {name} (multivariate) ---")
        evaluation(pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof}))

    # Test with multivariate-selected features
    print("\n### Test Evaluation (multivariate-selected) ###")
    for name in models_to_try:
        if name in ("XGBoost","LightGBM"):
            X_tr_sel, X_te_sel = X_trainval[sel_mtv], X_test[sel_mtv]
        else:
            X_tr_sel = X_train_imp[sel_mtv]; X_te_sel = X_test_imp[sel_mtv]
            scaler = StandardScaler()
            X_tr_sel = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=sel_mtv)
            X_te_sel = pd.DataFrame(scaler.transform(X_te_sel), columns=sel_mtv)
        model = make_model(name, y_trainval.values)
        model.fit(X_tr_sel, y_trainval.values)
        y_pred = model.predict(X_te_sel)
        print(f"\n--- Test results for {name} (multivariate) ---")
        evaluation(pd.DataFrame({"true_label": y_test.values, "prediction": y_pred}))
 
    print("\n### CV (selected features) ###")
    selected_columns = ['name_metric', 'paragraph_metric','language_metric','synonym_metric','author_metric']
    print(f"Selected columns: {selected_columns}")
    selected_columns_imp=selected_columns+[col+"_missing" for col in selected_columns if col !="name_metric"]
    print(f"Selected columns (imputed): {selected_columns_imp}")
    for name in models_to_try:
        y_true_oof, y_pred_oof = [], []
        for train_idx, val_idx in cv.split(X_trainval, y_trainval):
            X_tr_raw = X_trainval.iloc[train_idx]; X_val_raw = X_trainval.iloc[val_idx]
            y_tr = y_trainval.iloc[train_idx].values; y_val = y_trainval.iloc[val_idx].values
            if name in ("XGBoost","LightGBM"):
                X_tr, X_val = X_tr_raw[selected_columns].copy(), X_val_raw[selected_columns].copy()
            else:
                imp_fold = MedianImputerWithIndicator(cols=cols_to_impute)
                X_tr_imp = pd.DataFrame(imp_fold.fit_transform(X_tr_raw), columns=imp_fold.get_feature_names_out())
                X_val_imp = pd.DataFrame(imp_fold.transform(X_val_raw), columns=imp_fold.get_feature_names_out())
                X_tr_sel = X_tr_imp[selected_columns_imp]; X_val_sel = X_val_imp[selected_columns_imp]
                scaler = StandardScaler()
                X_tr = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=selected_columns_imp)
                X_val = pd.DataFrame(scaler.transform(X_val_sel), columns=selected_columns_imp)
            model = make_model(name, y_tr)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_true_oof.extend(y_val); y_pred_oof.extend(y_pred)
        print(f"\n--- CV results for {name} ---")
        evaluation(pd.DataFrame({"true_label": y_true_oof, "prediction": y_pred_oof}))

    print("\n### Test Evaluation (selected) ###")
    for name in models_to_try:
        if name in ("XGBoost","LightGBM"):
            X_tr_sel, X_te_sel = X_trainval[selected_columns], X_test[selected_columns]
        else:
            X_tr_sel = X_train_imp[selected_columns_imp]; X_te_sel = X_test_imp[selected_columns_imp]
            scaler = StandardScaler()
            X_tr_sel = pd.DataFrame(scaler.fit_transform(X_tr_sel), columns=selected_columns_imp)
            X_te_sel = pd.DataFrame(scaler.transform(X_te_sel), columns=selected_columns_imp)
        model = make_model(name, y_trainval.values)
        model.fit(X_tr_sel, y_trainval.values) 
        y_pred = model.predict(X_te_sel)
        print(f"\n--- Test results for {name} ---")
        evaluation(pd.DataFrame({"true_label": y_test.values, "prediction": y_pred}))
    # 3) Feature Combination Evaluation
    '''metrics = ['name_metric', 'paragraph_metric', 'language_metric',
           'synonym_metric', 'keywords_metric', 'author_metric']

    # Store results in a list of dicts
    results = []

    # iterate over combination sizes: 1 to N
    for k in range(1, len(metrics) + 1):
        for combo in itertools.combinations(metrics, k):
            combo = list(combo)
            combo_str = ",".join(combo)

            # prepare imputed feature names for non-tree models
            imp_features = combo + [col + '_missing' for col in combo if col != 'name_metric']

            # ----- Cross-validation -----
            for name in models_to_try:
                y_true_oof, y_pred_oof = [], []

                for train_idx, val_idx in cv.split(X_trainval, y_trainval):
                    X_tr_raw = X_trainval.iloc[train_idx]
                    X_val_raw = X_trainval.iloc[val_idx]
                    y_tr = y_trainval.iloc[train_idx].values
                    y_val = y_trainval.iloc[val_idx].values

                    if name in ("XGBoost", "LightGBM"):
                        X_tr = X_tr_raw[combo].copy()
                        X_val = X_val_raw[combo].copy()
                    else:
                        imp = MedianImputerWithIndicator(cols=cols_to_impute)
                        X_tr_imp = pd.DataFrame(
                            imp.fit_transform(X_tr_raw),
                            columns=imp.get_feature_names_out()
                        )
                        X_val_imp = pd.DataFrame(
                            imp.transform(X_val_raw),
                            columns=imp.get_feature_names_out()
                        )

                        X_tr_sel = X_tr_imp[imp_features]
                        X_val_sel = X_val_imp[imp_features]

                        scaler = StandardScaler()
                        X_tr = pd.DataFrame(
                            scaler.fit_transform(X_tr_sel),
                            columns=imp_features
                        )
                        X_val = pd.DataFrame(
                            scaler.transform(X_val_sel),
                            columns=imp_features
                        )

                    model = make_model(name, y_tr)
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)

                    y_true_oof.extend(y_val)
                    y_pred_oof.extend(y_pred)

                # compute CV metrics
                results.append({
                    'phase': 'CV',
                    'model': name,
                    'metrics_used': combo_str,
                    'precision': precision_score(y_true_oof, y_pred_oof, zero_division=0),
                    'recall': recall_score(y_true_oof, y_pred_oof, zero_division=0),
                    'f1': f1_score(y_true_oof, y_pred_oof, zero_division=0)
                })

            # ----- Test evaluation -----
            for name in models_to_try:
                if name in ("XGBoost", "LightGBM"):
                    X_tr_sel = X_trainval[combo]
                    X_te_sel = X_test[combo]
                else:
                    X_tr_sel = X_train_imp[imp_features]
                    X_te_sel = X_test_imp[imp_features]
                    scaler = StandardScaler()
                    X_tr_sel = pd.DataFrame(
                        scaler.fit_transform(X_tr_sel),
                        columns=imp_features
                    )
                    X_te_sel = pd.DataFrame(
                        scaler.transform(X_te_sel),
                        columns=imp_features
                    )

                model = make_model(name, y_trainval.values)
                model.fit(X_tr_sel, y_trainval.values)
                y_pred_test = model.predict(X_te_sel)

                # compute test metrics
                results.append({
                    'phase': 'Test',
                    'model': name,
                    'metrics_used': combo_str,
                    'precision': precision_score(y_test.values, y_pred_test, zero_division=0),
                    'recall': recall_score(y_test.values, y_pred_test, zero_division=0),
                    'f1': f1_score(y_test.values, y_pred_test, zero_division=0)
                })

    # save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('metric_combinations_results.csv', index=False)
    print("Saved results to 'metric_combinations_results.csv'")'''