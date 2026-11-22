"""Запуск: python run_experiments.py; рядом должен находиться german.csv."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from ensemble_models import evaluate_candidates, fit_selected

SEED = 42
path = Path(__file__).with_name('german.csv')
if not path.exists():
    raise SystemExit('Поместите german.csv рядом со скриптом.')
df = pd.read_csv(path, sep=';')
X = df.drop(columns='Creditability')
y = df['Creditability'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
results, best_name = evaluate_candidates(X_train, y_train, SEED)
model = fit_selected(X_train, y_train, SEED, best_name)
proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
pred = (proba >= 0.5).astype(int)
output = dict(
    seed=SEED, split='stratified 80/20', train_size=len(X_train), test_size=len(X_test),
    experiments=results, best_cv_model=best_name, submitted_model=best_name,
    test_metrics=dict(
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    ),
)
Path(__file__).with_name('experiment_results.json').write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(output, ensure_ascii=False, indent=2))
