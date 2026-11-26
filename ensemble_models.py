from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if {"Credit_Amount", "Duration_of_Credit_monthly"}.issubset(data.columns):
        duration = data["Duration_of_Credit_monthly"].replace(0, 1)
        data["Monthly_Payment_Est"] = data["Credit_Amount"] / duration
        data["Log_Credit_Amount"] = np.log1p(np.maximum(data["Credit_Amount"], 0))
    if {"Credit_Amount", "Age_years"}.issubset(data.columns):
        age = data["Age_years"].replace(0, 1)
        data["Credit_to_Age"] = data["Credit_Amount"] / age
    return data


def _to_dataframe(features: object) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()
    arr = np.asarray(features)
    cols = [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _create_estimator(model_key: str, seed: int):
    if model_key == "random_forest":
        return (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=seed,
                n_jobs=1,
            ),
        )
    if model_key == "extra_trees":
        return (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=1,
            ),
        )
    if model_key == "gradient_boosting":
        return (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=seed,
            ),
        )
    if model_key == "hist_gradient_boosting":
        return (
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                max_depth=4,
                l2_regularization=1.0,
                random_state=seed,
            ),
        )
    if model_key == "mlp":
        return (
            "MLPClassifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.05,
                learning_rate_init=0.001,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            ),
        )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=seed,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("et", et)],
        voting="soft",
        weights=[2, 2, 1],
    )
    return ("SoftVotingEnsemble(RF+GB+ET)", voting)


def run_ensemble_experiment(
    data: EnsembleInput,
    model_key: str = "extra_trees",
    use_feature_engineering: bool = True,
) -> EnsembleResult:
    X_train = _to_dataframe(data.train_features)
    X_test = _to_dataframe(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    if use_feature_engineering:
        X_train = _add_features(X_train)
        X_test = _add_features(X_test)

    preprocessor = _build_preprocessor(X_train)
    model_name, estimator = _create_estimator(model_key, data.random_seed)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name=model_name,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )


def _run_local_study() -> None:
    csv_path = Path(__file__).with_name("german.csv")
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, sep=";")
    target_col = "Creditability"
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dataset = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        random_seed=42,
    )

    configs = [
        ("random_forest", False),
        ("random_forest", True),
        ("extra_trees", True),
        ("gradient_boosting", True),
        ("hist_gradient_boosting", True),
        ("mlp", True),
        ("voting", True),
    ]

    results = []
    for key, fe in configs:
        res = run_ensemble_experiment(dataset, model_key=key, use_feature_engineering=fe)
        results.append(
            {
                "model_name": res.model_name,
                "feature_engineering": fe,
                "roc_auc": round(res.roc_auc, 4),
                "accuracy": round(res.accuracy, 4),
                "precision": round(res.precision, 4),
                "recall": round(res.recall, 4),
                "f1": round(res.f1, 4),
            }
        )

    best = max(results, key=lambda item: item["roc_auc"])
    payload = {
        "random_seed": 42,
        "test_size": 0.2,
        "selected_model": best,
        "experiments": results,
    }

    out_path = Path(__file__).with_name("experiment_results.json")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _run_local_study()