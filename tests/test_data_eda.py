import pytest
import pandas as pd

@pytest.fixture
def df():
    return pd.read_csv("data/tweets.csv")

def test_colonnes(df):
    assert 'text' in df.columns
    assert 'target' in df.columns
    assert df['text'].dtype == object
    assert pd.api.types.is_numeric_dtype(df['target'])

def test_manquantes_doublons(df):
    assert df['text'].isnull().sum() == 0
    assert df['target'].isnull().sum() == 0
    assert df.duplicated().sum() == 0

def test_non_vide(df):
    assert all(df['text'].astype(str).str.strip().str.len() > 0)

def test_classes(df):
    assert set(df['target'].unique()) <= {0, 1}

def test_longueur(df):
    longueurs = df['text'].astype(str).str.len()
    assert longueurs.min() >= 1
    assert longueurs.mean() > 10
    assert longueurs.max() < 300

    