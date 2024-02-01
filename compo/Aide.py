import streamlit as st
from compo.pageIntro import page_intro_modeles_supervises

def page_aide():

    st.title("Page d'Aide - Tutoriel Machine Learning")


    # Appliquer des styles avec du CSS
    st.markdown("""
        <style>
            h1 {
                color: #1a5e9b;
                text-align: center;
                transition: color 0.3s ease-in-out;
            }
            h1:hover {
                color: #667fa7;
            }
            h2 {
                color: #4E84B9;
                transition: color 0.3s ease-in-out;
            }
            h2:hover {
                color: #074b7b;
            }
            p {
                font-size: 16px;
                
            }
        </style>
    """, unsafe_allow_html=True)

    st.header("Principe de l'Application")
    st.markdown("""
        Bienvenue dans la page d'aide de notre application de machine learning. 
        Suivez les instructions ci-dessous pour utiliser l'application efficacement.
    """)

    st.write("""
        
        Notre application a pour objectif de simplifier le processus de construction de modèles de machine learning (Supervisés), 
        en vous offrant une expérience sans coder. Voici comment vous pouvez utiliser notre application :
    """)

    st.markdown("""
        Avec notre approche sans coder, nous visons à rendre l'expérience de création de modèles de machine learning 
        accessible même à ceux qui n'ont pas de connaissances avancées en programmation.
    """)

    # Bouton stylisé avec CSS
    if st.button("Pour plus d'informations sur les modèles utilisés, cliquez ici"):
        page_intro_modeles_supervises()
        if st.button("Fermer"):

            st.experimental_rerun()


    st.header(" Menu de Navigation")

    st.markdown("""
        Utilisez la barre de navigation en haut de la page pour accéder aux différentes sections de l'application.
        N'hésitez pas à explorer et expérimenter avec les fonctionnalités disponibles.
    """)

    st.header("1. Accueil")

    st.markdown("""
        Sur la page d'accueil, vous pouvez effectuer plusieurs actions pour préparer vos données avant l'analyse.
        - :file_folder: **Création du dataset**: Créez un ensemble de données pour votre analyse.
        - :pencil: **Modification des données**: Modifiez vos données existantes si nécessaire.
        - :arrow_up: **Importation des données**: Importez un ensemble de données pré-existant.
    """)

    st.markdown("""
        Après l'importation des données, l'application traitera votre ensemble et vous dirigera vers le modèle approprié 
        (Classification ou Régression) en fonction de la nature de vos données.
    """)

    st.header("2. Modèles")

    st.markdown("""
        La page des Modèles se compose de trois sections :
        - :chart_with_upwards_trend: **Visualisation de l'ensemble de données**: Visualisez différents graphiques pour mieux comprendre vos données.
        - :bar_chart: **Résultats du modèle**: Une fois un modèle sélectionné et les paramètres ajustés, examinez la performance du modèle 
          ainsi que diverses métriques dans cette section.
          De même, vous pouvez exporter les résultats de votre modèle à tout moment !
        - :test_tube: **Test**: Effectuez des tests supplémentaires pour évaluer la robustesse de votre modèle.
    """)


