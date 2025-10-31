# import streamlit as st
# from difflib import SequenceMatcher

# # Configuration de la page
# st.set_page_config(page_title="Analyse IA & Plagiat", layout="centered", page_icon="🧠")

# #En-tête stylisé
# st.markdown("""
#     <style>
#     # .main {
#     #     background-color: #f9f9f9;
#     # }
#     .sidebar{
#         background-color: #3498db;
#     }
#     .sidebar .sidebar-content {
#         background: #3498db;
#         color: white;
#     }
#     .sidebar .sidebar-content a {
#         color: white;
#     }
#     .stButton>button {
#         background-color: #3498db;
#         color: white;
#         border-radius: 8px;
#         padding: 10px 20px;
#     }
#     textarea {
#         border-radius: 10px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Menu latéral
# st.sidebar.title("🧭 Menu")
# choix = st.sidebar.radio("Choisissez une section :", ["Plagiat", "Génération par IA"])

# # FONCTION : Calcul de similarité (plagiat simple avec difflib)
# def calc_similarity(text1, text2):
#     ratio = SequenceMatcher(None, text1, text2).ratio()
#     return round(ratio * 100, 2)

# # Section Plagiat
# if choix == "Plagiat":
#     st.title("🕵️ Détection de Plagiat")
#     st.write("Comparez deux textes pour voir leur similarité.")

#     texte_1 = st.text_area("Texte 1", height=200, placeholder="Entrez le premier texte ici...")
#     texte_2 = st.text_area("Texte 2", height=200, placeholder="Entrez le second texte ici...")

#     if st.button("🔍 Générer le score de similarité"):
#         if texte_1.strip() and texte_2.strip():
#             score = calc_similarity(texte_1, texte_2)
#             st.success(f"📊 Score de similarité : **{score}%**")
#             if score > 80:
#                 st.warning("⚠️ Ces textes semblent très similaires. Risque de plagiat élevé.")
#             elif score > 50:
#                 st.info("ℹ️ Ces textes ont une similarité modérée.")
#             else:
#                 st.info("✅ Ces textes sont probablement originaux.")
#         else:
#             st.error("Veuillez remplir les deux champs de texte.")

# # Section Génération IA
# elif choix == "Génération par IA":
#     st.title("🤖 Analyse de Texte Généré par IA")
#     st.write("Entrez un texte généré pour l'analyser.")

#     texte_genere = st.text_area("Texte généré", height=300, placeholder="Collez ici le texte généré...")
#     st.button("🔍 Générer le score de generation")

#     if texte_genere:
#         st.info("🔎 Analyse simple :")
#         longueur = len(texte_genere.split())
#         st.write(f"📄 Nombre de mots : **{longueur}**")

#         if longueur < 30:
#             st.warning("Le texte semble court, il pourrait être peu informatif.")
#         else:
#             st.success("Le texte semble assez détaillé.")
import streamlit as st
#from difflib import SequenceMatcher
import joblib  # Pour charger les modèles
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords


# Téléchargement des stopwords français (à faire une seule fois)
nltk.download('stopwords')

# Création de l'ensemble des stopwords français
stop_words = set(stopwords.words('english'))

# Chargement des modèles (à adapter selon vos fichiers)
@st.cache_resource
def load_models():
    try:
        model_plagiat = joblib.load('plagiat_best_nodel_RF.pkl')
        model_ia = joblib.load('ai_text_detector_model.pkl')
        vectorizer_plagiat = joblib.load('vectorizer_plagiat.pkl')
        vectorizer_ia = joblib.load('vectorizer.pkl')
        return model_plagiat, model_ia, vectorizer_plagiat, vectorizer_ia
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None, None
# 🔧 Fonctions de prétraitement
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_pair(text1, text2):
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    return clean1 + " [SEP] " + clean2

def preprocess_pair_ia(text):
    clean1 = clean_text(text)
    return clean1

model_plagiat, model_ia, vectorizer_plagiat, vectorizer_ia = load_models()

# En-tête stylisé (votre CSS existant)
st.markdown("""
    <style>
    .sidebar{
        background-color: #3498db;
    }
    .sidebar .sidebar-content {
        background: #3498db;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: white;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    textarea {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration de la page
st.set_page_config(page_title="Analyse IA & Plagiat", layout="centered", page_icon="🧠")

# Menu latéral
st.sidebar.title("🧭 Menu")


choix = st.sidebar.radio("Choisissez une section :", ["Génération par IA","Détection de plagiat"])

# Section Plagiat
if choix == "Détection de plagiat":
    st.title("🕵️ Détection de Plagiat Avancée")
    
   # method = st.radio("Méthode d'analyse:", ["Règle simple (difflib)", "Modèle ML"])
    
    texte_1 = st.text_area("Texte original", height=200, placeholder="Entrez le texte source...")
    texte_2 = st.text_area("Texte à comparer", height=200, placeholder="Entrez le texte suspect...")

    if st.button("🔍 Analyser"):
        if not texte_1.strip() or not texte_2.strip():
            st.error("Veuillez remplir les deux champs de texte.")
            
        # if method == "Règle simple (difflib)":
        #     # Méthode existante
        #     score = round(SequenceMatcher(None, texte_1, texte_2).ratio() * 100, 2)
        #     st.success(f"📊 Score de similarité : {score}%")
            
        else:
            # Utilisation du modèle ML
            if model_plagiat is None:
                st.error("Modèle non chargé")
                
            # Préparation des données (adaptez selon votre pipeline)
            text_diff = preprocess_pair(texte_1,texte_2)
            X = vectorizer_plagiat.transform([text_diff])
            
            # Prédiction
            prediction = model_plagiat.predict(X)[0]
            proba = model_plagiat.predict_proba(X)[0]
            
            # Affichage
            st.subheader("🔎 Résultats du modèle ML")
            st.write(f"Prédiction: {'🟢 Non-plagiat' if prediction == 0 else '🔴 Plagiat'}")
            st.write(f"Confiance: {max(proba)*100:.1f}%")
            # Affichage
            st.subheader("Résultats de l'analyse")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prédiction", 
                    value="IA 🤖" if prediction == 1 else "Humain 👨💻",
                    delta=f"{max(proba)*100:.1f}% de confiance")
                
            with col2:
                st.write("Indices détectés:")
                if prediction == 1:
                    st.error("✅ Patterns typiques des LLM")
                    st.error("✅ Structure trop parfaite")
                else:
                    st.success("✅ Style humain détecté")
                    st.success("✅ Imperfections naturelles")
                
            # Visualisation
            st.progress(proba[1] if prediction == 1 else proba[0])
            st.caption(f"Probabilité: {max(proba)*100:.1f}%")
            # Explication
            # if prediction == 1:
            #     st.warning("Le modèle a détecté des signes de plagiat.")
            #     st.write("Caractéristiques suspectes:")
            #     st.write("- Similarité structurelle élevée")
            #     st.write("- Répétition de phrases uniques")
            # else:
            #     st.success("Le texte semble original.")

# Section Génération IA
elif choix == "Génération par IA":
    st.title("🤖 Détection de Contenu Généré par IA")
    
    texte_genere = st.text_area("Texte à analyser", height=300, 
                              placeholder="Collez ici le texte suspect...")
    
    if st.button("🔍 Vérifier l'origine"):
        if not texte_genere.strip():
            st.error("Veuillez entrer un texte à analyser.")
            
        if model_ia is None:
            st.error("Modèle IA non chargé")
            
        # Vectorisation
        text_traite = preprocess_pair_ia(texte_genere)
        X = vectorizer_ia.transform([text_traite])
        
        # Prédiction
        prediction = model_ia.predict(X)[0]
        proba = model_ia.predict_proba(X)[0]
        
        # Affichage
        if texte_genere.strip() :
            st.subheader("Résultats de l'analyse")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prédiction", 
                        value="IA 🤖" if prediction == 1 else "Humain 👨💻",
                        delta=f"{max(proba)*100:.1f}% de confiance")
            
            with col2:
                st.write("Indices détectés:")
                if prediction == 1:
                    st.error("✅ Patterns typiques des LLM")
                    st.error("✅ Structure trop parfaite")
                else:
                    st.success("✅ Style humain détecté")
                    st.success("✅ Imperfections naturelles")
            
            # Visualisation
            st.progress(proba[1] if prediction == 1 else proba[0])
            st.caption(f"Probabilité: {max(proba)*100:.1f}%")

