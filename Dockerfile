# 1️⃣ Utiliser une image Python optimisée pour le ML
FROM python:3.10

# 2️⃣ Définir le répertoire de travail dans le conteneur
WORKDIR /app

# 3️⃣ Copier les fichiers du projet dans l’image Docker
COPY . /app

# 4️⃣ Installer les dépendances nécessaires (FastAPI, TensorFlow, etc.)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5️⃣ Exposer le port 8000 pour l’API
EXPOSE 8000

# Ajouter un volume pour sauvegarder les modèles téléchargés
VOLUME ["/app/models"]

# 6️⃣ Démarrer FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
