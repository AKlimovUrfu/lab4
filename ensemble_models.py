"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def to_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.array(X, dtype=float)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает модели, выбирает лучшую на валидации и оценивает на тесте."""
    seed = data.random_seed

    X_train_full = to_numpy(data.train_features)
    y_train_full = np.asarray(data.train_target)
    X_test = to_numpy(data.test_features)
    y_test = np.asarray(data.test_target)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
    )

    scaler_val = StandardScaler()
    X_fit_scaled = scaler_val.fit_transform(X_fit)
    X_val_scaled = scaler_val.transform(X_val)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=seed, early_stopping=True)
    }

    best_name = None
    best_model_template = None
    best_roc_auc = -1.0

    for name, model in models.items():
        model.fit(X_fit_scaled, y_fit)
        val_probs = model.predict_proba(X_val_scaled)[:, 1]
        val_roc_auc = roc_auc_score(y_val, val_probs)
        print(f"[Валидация] {name} ROC-AUC: {val_roc_auc:.4f}")

        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            best_name = name
            best_model_template = model

    print(f"-> Выбрана лучшая модель: {best_name}")

    scaler_final = StandardScaler()
    X_train_scaled_final = scaler_final.fit_transform(X_train_full)
    X_test_scaled_final = scaler_final.transform(X_test)

    final_model = clone(best_model_template)
    final_model.fit(X_train_scaled_final, y_train_full)

    probabilities = final_model.predict_proba(X_test_scaled_final)[:, 1]
    predictions = final_model.predict(X_test_scaled_final)

    return EnsembleResult(
        model_name=best_name,
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0))
    )


# ==========================================
# ЧЕСТНОЕ СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ ДЛЯ ОТЧЁТА
# ==========================================
if __name__ == "__main__":
    print(" Запуск полного честного сравнения всех моделей...")
    
    df = pd.read_csv("german.csv", sep=";")
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models_to_compare = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "MLPClassifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True)
    }
    
    all_results = {}
    
    print("\n--- РЕАЛЬНЫЕ МЕТРИКИ НА ТЕСТЕ ДЛЯ ВСЕХ МОДЕЛЕЙ ---")
    for name, model in models_to_compare.items():
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
        
        r_auc = float(roc_auc_score(y_test, probs))
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        
        all_results[name] = {
            "roc_auc": round(r_auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
        print(f"{name:16} | ROC-AUC: {r_auc:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

    # Сохраняем реальные метрики всех моделей в JSON
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    
    print("\n Реальные метрики всех моделей сохранены в experiment_results.json!")
    
    # Также проверяем работу контрактной функции
    class MockInput:
        def __init__(self):
            self.train_features = X_train
            self.train_target = y_train.values
            self.test_features = X_test
            self.test_target = y_test.values
            self.random_seed = 42

    final_result = run_ensemble_experiment(MockInput())
    print(f"\n Контрактная функция вернула: {final_result.model_name}, ROC-AUC: {final_result.roc_auc:.4f}")