# Speech Emotion Recognition (SER) - CodeAlpha

Ce projet est une application de reconnaissance des émotions à partir de signaux audio (la voix) utilisant des techniques de Deep Learning et de traitement du signal sonore. L'objectif est d'analyser les caractéristiques acoustiques d'un enregistrement vocal pour en déterminer l'état émotionnel (joie, tristesse, colère, peur, etc.).

## 🚀 Fonctionnalités
- **Extraction de caractéristiques acoustiques** : Analyse et traitement des signaux audio (MFCC, Chroma, Mel Spectrogram) via `librosa`.
- **Modèle de Deep Learning** : Architecture optimisée sous Keras/TensorFlow pour la classification des émotions.
- **Pipeline d'évaluation** : Notebooks pas-à-pas pour l'extraction, l'entraînement et les tests en direct.
- **Interface Utilisateur** : Une application interactive (`app.py`) permettant de tester facilement des fichiers audio.

## 📁 Structure du Projet

Conformément aux bonnes pratiques d'ingénierie logicielle, ce dépôt ne contient que le code source léger. Les fichiers de données volumineux et les modèles entraînés sont exclus du suivi Git pour des raisons de performance et de sécurité.

```text
├── 3_model_training.ipynb   # Pipeline d'entraînement du modèle d'IA
├── 4_live_test.ipynb        # Scripts de tests et prédictions en direct
├── app.py                   # Application principale (Interface utilisateur)
├── extraction.ipynb         # Prétraitement et extraction des caractéristiques audio
├── .gitignore               # Configuration des fichiers exclus (modèles, datasets)
└── README.md                # Documentation du projet (ce fichier)
