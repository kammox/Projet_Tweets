import pandas as pd
from src.preprocessing import clean_text, corpus_statistics
from src.modeling import train_and_evaluate

def main():
    # Charger données
    df = pd.read_csv("data/tweets.csv")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)

    stats = corpus_statistics(df["clean_text"].tolist())
    print("Corpus stats:", stats)

    # Entraîner pipeline
    _, metrics, _, _ = train_and_evaluate(df["clean_text"], df["target"])
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
