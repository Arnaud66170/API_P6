from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import cv2
from PIL import Image
import io
import uvicorn
import tensorflow_hub as hub
from transformers import BertTokenizer
import nest_asyncio
import os

# Application du patch pour éviter les conflits asyncio
nest_asyncio.apply()

# Initialisation de l'API
app = FastAPI(
    title="API de Classification de Produits",
    description="API permettant la classification des produits en fonction du texte et des images.",
    version="1.0"
)

# Définition des chemins des modèles et fichiers
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "use_classification_model.h5")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "InceptionV3.h5")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Chargement des modèles
print("🔄 Chargement des modèles...")
text_model = tf.keras.models.load_model(TEXT_MODEL_PATH)
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, "rb"))
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print("✅ Modèles chargés avec succès !")

# Classe de validation pour le texte
class TextRequest(BaseModel):
    description: str

# Fonction pour encoder du texte avec USE
def encode_with_use(text):
    return use_model([text]).numpy().flatten()

# Fonction pour le prétraitement d'image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Endpoint pour la prédiction du texte
@app.post("/predict_text")
def predict_text(request: TextRequest):
    try:
        embedding = encode_with_use(request.description)
        embedding = np.expand_dims(embedding, axis=0)
        prediction = text_model.predict(embedding)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        return {"category": predicted_label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Endpoint pour la prédiction de l'image
@app.post("/predict_image")
def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = file.file.read()
        img_array = preprocess_image(image_bytes)
        prediction = image_model.predict(img_array)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        return {"category": predicted_label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Endpoint pour extraire un embedding de description
@app.post("/extract_embedding")
def extract_embedding(request: TextRequest):
    try:
        embedding = encode_with_use(request.description)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

# Endpoint de santé pour vérifier si l'API fonctionne
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Endpoint racine pour accueil
@app.get("/")
def root():
    return {"message": "Bienvenue dans l'API de classification"}

# Exécution de l'API (lancement avec `uvicorn projet_api.main:app --reload`)
if __name__ == "__main__":
    uvicorn.run("projet_api.main:app", host="127.0.0.1", port=8000, reload=True)
