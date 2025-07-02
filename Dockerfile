FROM python:3.12-slim

# Définit le répertoire de travail
WORKDIR /app

# Définir le chemin pour nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

# Copie les fichiers du projet
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["pytest", "--maxfail=1", "--disable-warnings"]
