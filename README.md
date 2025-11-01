# 🧠 Vérificateur d'Intégrité : Détection IA & Plagiat (ML)

## 🎯 Double Objectif
Application complète basée sur le **Machine Learning (ML)** et le **Traitement Automatique du Langage Naturel (NLP)**.

1.  **Détection de Contenu IA :** Modèle ML entraîné pour prédire si le texte a été généré par un LLM (e.g., ChatGPT).
2.  **Détection de Plagiat :** Utilisation de l'analyse de similarité NLP pour quantifier le risque de plagiat.

---

## 🧠 Méthodologie et Technologies Clés
| Domaine | Outils et Technologies | Rôle dans le Projet |
| :--- | :--- | :--- |
| **Machine Learning (ML)** | Scikit-learn (`LogisticRegression`, `RandomForestClassifier`) | Modèles de classification pour la détection IA. |
| **NLP & Vectorisation** | `NLTK`, `TfidfVectorizer` | Pré-traitement et extraction de caractéristiques textuelles. |
| **Déploiement** | **Streamlit** | Interface utilisateur interactive. |



## ⚙️ Exécution Locale
1.  Installez les dépendances : `pip install -r requirements.txt`
2.  Lancez l'application : `streamlit run app.py`
