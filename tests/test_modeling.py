import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling import build_pipeline, train_and_evaluate


def test_pipeline_runs():
    X = pd.Series(["fire in forest", "just a normal day"])
    y = pd.Series([1, 0])
    pipe = build_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)

def test_train_and_evaluate_metrics():
    X = pd.Series([
        "disaster fire", "normal news",
        "earthquake hits", "sunny day",
        "big storm", "clear sky"
    ])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    _, metrics, _, _ = train_and_evaluate(X, y)
    assert all(m in metrics for m in ['accuracy', 'precision', 'recall', 'f1'])


def test_pipeline_handles_empty_text():
    X = pd.Series(["", "short"])
    y = pd.Series([0, 1])
    pipe = build_pipeline()
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 2
