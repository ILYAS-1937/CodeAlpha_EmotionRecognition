import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import noisereduce as nr

# 1. Configuration de la page web
st.set_page_config(page_title="IA - Émotions Vocales", page_icon="🎙️", layout="centered")

st.title("🎙️ Détecteur d'Émotions Vocales")
st.write("Application web propulsée par un Réseau de Neurones Convolutif (CNN). Projet CodeAlpha.")
st.markdown("---")

# 2. Chargement du "cerveau" (Mise en cache pour la rapidité du site)
@st.cache_resource
def load_ai():
    model = tf.keras.models.load_model('emotion_recognition_model.keras')
    scaler = joblib.load('scaler.save')
    return model, scaler

try:
    model, scaler = load_ai()
    modele_charge = True
except Exception as e:
    st.error(f"⚠️ Erreur de chargement du modèle. Vérifiez que les fichiers .keras et .save sont dans le même dossier. ({e})")
    modele_charge = False

# Dictionnaire des émotions
emotions_dict = {
    0: "Neutre 😐", 1: "Calme 😌", 2: "Heureux 😄", 3: "Triste 😢",
    4: "En colère 😠", 5: "Effrayé 😨", 6: "Dégoûté 🤢", 7: "Surpris 😲"
}

# 3. Interface Utilisateur
if modele_charge:
    st.markdown("### 🔴 Testez le modèle en direct")
    
    # Le widget magique de Streamlit pour capter le micro
    audio_value = st.audio_input("Cliquez sur le micro et parlez pendant 3 à 5 secondes :")

    if audio_value is not None:
        st.success("Enregistrement capturé avec succès ! Analyse en cours...")
        
        try:
            # Traitement de l'audio directement depuis le navigateur
            y, sr = librosa.load(audio_value, sr=22050)
            
            # Nettoyage du bruit
            y_clean = nr.reduce_noise(y=y, sr=sr)
            
            # Extraction des caractéristiques
            mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            # Préparation pour le réseau de neurones
            mfccs_scaled = scaler.transform([mfccs_mean])
            donnee_prete = np.expand_dims(mfccs_scaled, axis=2)
            
            # Prédiction
            prediction = model.predict(donnee_prete)
            index_emotion = np.argmax(prediction)
            confiance = float(np.max(prediction) * 100)
            
            # Affichage spectaculaire du résultat
            resultat = emotions_dict.get(index_emotion, "Inconnue")
            
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center;'>🎯 Émotion : {resultat}</h2>", unsafe_allow_html=True)
            st.progress(int(confiance))
            st.markdown(f"<p style='text-align: center;'>Indice de confiance : <b>{confiance:.2f}%</b></p>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Oups ! Une erreur s'est produite lors de l'analyse : {e}")