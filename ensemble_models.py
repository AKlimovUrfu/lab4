from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from grader_contracts.ensembles import EnsembleInput, EnsembleResult

CATEGORICAL = {
    'Account_Balance', 'Payment_Status_of_Previous_Credit', 'Purpose',
    'Value_Savings_Stocks', 'Length_of_current_employment', 'Sex_Marital_Status',
    'Guarantors', 'Most_valuable_available_asset', 'Concurrent_Credits',
    'Type_of_apartment', 'Occupation', 'Telephone', 'Foreign_Worker',
}


def candidate_specs(seed=42):
    return [
        ('ExtraTrees_leaf2', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=0.85, random_state=seed, n_jobs=1)),
        ('ExtraTrees_leaf4', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4, max_features=0.85, random_state=seed, n_jobs=1)),
        ('ExtraTrees_sqrt', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt', random_state=seed, n_jobs=1)),
        ('RandomForest', RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features='sqrt', random_state=seed, n_jobs=1)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed)),
        ('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)),
        ('MLP', MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=600, early_stopping=True, random_state=seed)),
    ]


def make_pipeline(X, seed=42, estimator=None, **params):
    X = pd.DataFrame(X)
    categorical = [c for c in X.columns if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    prep = ColumnTransformer([
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric),
    ], sparse_threshold=0)
    if estimator is None:
        defaults = dict(n_estimators=400, min_samples_leaf=2, max_features=0.85,
                        random_state=seed, n_jobs=1)
        defaults.update(params)
        estimator = ExtraTreesClassifier(**defaults)
    return Pipeline([('preprocess', prep), ('model', estimator)])


def evaluate_candidates(X_train, y_train, seed=42):
    X_train = pd.DataFrame(X_train)
    y_train = np.asarray(y_train).ravel().astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    for name, estimator in candidate_specs(seed):
        model = make_pipeline(X_train, seed, estimator=estimator)
        proba = cross_val_predict(model, X_train, y_train, cv=cv,
                                  method='predict_proba', n_jobs=1)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append(dict(model=name, cv_roc_auc=float(roc_auc_score(y_train, proba)),
                            cv_accuracy=float(accuracy_score(y_train, pred)),
                            cv_precision=float(precision_score(y_train, pred, zero_division=0)),
                            cv_recall=float(recall_score(y_train, pred, zero_division=0)),
                            cv_f1=float(f1_score(y_train, pred, zero_division=0))))
    best = max(results, key=lambda r: r['cv_roc_auc'])['model']
    return results, best


def fit_selected(X_train, y_train, seed, best_name):
    estimator = dict(candidate_specs(seed))[best_name]
    model = make_pipeline(X_train, seed, estimator=estimator)
    model.fit(X_train, y_train)
    return model


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    X_train = pd.DataFrame(data.train_features)
    X_test = pd.DataFrame(data.test_features)
    y_train = np.asarray(data.train_target).ravel().astype(int)
    y_test = np.asarray(data.test_target).ravel().astype(int)
    _, best_name = evaluate_candidates(X_train, y_train, data.random_seed)
    model = fit_selected(X_train, y_train, data.random_seed, best_name)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    predictions = (probabilities >= 0.5).astype(int)
    return EnsembleResult(
        model_name=best_name, probabilities=probabilities, predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
