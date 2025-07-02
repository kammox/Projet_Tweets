# tests/test_preprocessing.py

from src.preprocessing import tokenize, clean_tokens, clean_text

def test_tokenize():
    tokens = tokenize("Hello world! 123.")
    assert "Hello" in tokens
    assert "world" in tokens
    assert "!" in tokens
    assert "123" in tokens

def test_clean_tokens_basic():
    tokens = ["Hi", "123", ".", "the", "cat", "on", "mat"]
    cleaned = clean_tokens(tokens)
    # Vérifie que les chiffres, ponctuation, mots courts et stopwords sont supprimés
    assert "123" not in cleaned
    assert "." not in cleaned
    assert "the" not in cleaned
    assert "on" not in cleaned
    assert "hi" not in [t.lower() for t in cleaned]
    assert all(len(t) >= 3 for t in cleaned)

def test_clean_text_empty():
    assert clean_text("") == ""

def test_clean_text_stopwords_and_short_words():
    text = "This is a test of stopwords"
    cleaned = clean_text(text)
    cleaned_tokens = cleaned.split()
    assert "this" not in cleaned_tokens
    assert "is" not in cleaned_tokens
    assert "of" not in cleaned_tokens
    assert all(len(t) >= 3 for t in cleaned_tokens)

def test_vocab_reduction():
    raw = "The cats are on the mat and the cats are happy."
    tokens_before = tokenize(raw)
    cleaned = clean_text(raw)
    tokens_after = cleaned.split()
    assert len(tokens_after) < len(tokens_before)

def test_stemming_lemmatization_impact():
    tokens = ["running", "flies", "better"]
    stemmed = clean_tokens(tokens, use_stemming=True, use_lemmatization=False)
    lemmatized = clean_tokens(tokens, use_stemming=False, use_lemmatization=True)

    assert "run" in stemmed or "fli" in stemmed
    assert "running" not in stemmed
    assert "flies" not in stemmed

    assert "fly" in lemmatized
    assert "flies" not in lemmatized

def test_pipeline_full_stemming():
    text = "Wow! Running with the cats 123 on mat."
    cleaned = clean_text(text, use_stemming=True, use_lemmatization=False)
    tokens = cleaned.split()
    assert "wow" in tokens
    assert "running" not in tokens
    assert "run" in tokens or "cat" in tokens

def test_pipeline_full_lemmatization():
    text = "Wow! Running with the cats 123 on mat."
    cleaned = clean_text(text, use_stemming=False, use_lemmatization=True)
    tokens = cleaned.split()
    assert "wow" in tokens
    assert "running" not in tokens
    assert "run" in tokens or "cat" in tokens