#!/bin/bash

# Définition du dossier cible
MODEL_DIR="models"
MODEL_NAME="InceptionV3.h5"
FILE_ID="1_17-7b2xlYOKyEWYIg1lDfZqN2lvi5CA"  # ID du fichier Google Drive

# Vérifier si le dossier existe, sinon le créer
mkdir -p $MODEL_DIR

# Télécharger le fichier depuis Google Drive
echo "Téléchargement de $MODEL_NAME depuis Google Drive..."
# curl -L -o $MODEL_DIR/$MODEL_NAME "https://drive.google.com/uc?export=download&id=$FILE_ID"
gdown --id 1_17-7b2xlYOKyEWYIg1lDfZqN2lvi5CA -O $MODEL_DIR/InceptionV3.h5

echo "Le fichier a été téléchargé dans $MODEL_DIR/$MODEL_NAME"
