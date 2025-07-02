## Tweet Classifier

Ce projet est un pipeline de classification de tweets, avec analyse EDA, prétraitement des textes (tokenization NLTK), modélisation et tests automatisés sous pytest.
Tout est containerisé avec Docker pour une exécution reproductible. 


## 📂 Structure du projet

- **TWEET_PROJECT3/**
  - **data/**
    - `tweets.csv`
  - **notebooks/**
    - `01_EDA.ipynb`
    - `02_Preprocessing_Modeling.ipynb`
  - **src/**
    - `preprocessing.py`
    - `modeling.py`
  - **tests/**
    - `test_data_eda.py`
    - `test_preprocessing.py`
    - `test_modeling.py`
  - `Dockerfile`
  - `main.py`
  - `requirements.txt`
  - `README.md`


## Prérequis
Docker installé.

(Optionnel) Python 3.12 pour exécuter localement


## Exécution locale (hors Docker):

1. Installe les dépendances :
pip install -r requirements.txt

2. Télécharge punkt :
import nltk
nltk.download('punkt')

3. Lancer le main :
python main.py 

5. Lancer les tests :
pytest


## 🐳 Lancer le projet avec Docker

1. Construire l’image Docker:

docker build -t tweet-classifier .

2. Exécuter les tests :

docker run --rm tweet-classifier

## À quoi sert chaque module ?

| Fichier                | Description                       |
| ---------------------- | --------------------------------- |
| src/data_eda.py        | Scripts d’exploration des données |
| src/preprocessing.py   | Nettoyage & tokenization (NLTK)   |
| src/modeling.py        | Modèles de classification         |
| tests/                 | Tous les tests unitaires pytest   |


## Attention : Problème fréquent avec NLTK
La première fois que tu exécutes le container, NLTK peut manquer le modèle punkt nécessaire pour tokenizer le texte.
Ce projet télécharge automatiquement punkt dans le Dockerfile :
RUN python -m nltk.downloader punkt

Vérifie bien que tu as reconstruit l’image après avoir modifié le Dockerfile :
docker build -t tweet-classifier .


## À propos

Auteur : Kamelia TRIKI

Contact : kamelia.triki@ynov.com
