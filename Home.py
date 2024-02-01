

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np 
import base64
from compo.Models import classification
from compo.Regression import regression_page
from compo.dataEntry import main
from compo.importData import importData_page
from compo.modify_data import modify_page
from compo.Aide import page_aide
from compo.clustering import clusteringPage


st.set_page_config(page_title='PROJET DE FIN DE MODULE ' ,
    layout='wide')
with open('styles.css') as f:

    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

########1 ere etppe : notre intefrcace en utilisant la biblio streamlit

st.markdown("""
    <style>

    @keyframes colorVibrate {

        0% { color: #1a3359; }
        25% { color: #2b4a6b; }
        50% { color: #4e7296; }
        75% { color: #667fa7; }
        100% { color: #536a53; }
}
        .vibrant-title {
            text-align: center;
            animation: colorVibrate 2s infinite;
        }
    </style>
""", unsafe_allow_html=True)
# Titre
st.markdown("<h1 class='vibrant-title'>Bienvenue dans votre application de Machine Learning ! </h1>", unsafe_allow_html=True)

# Option menu# Main page de app
selected=option_menu(
menu_title=None , #c le required
options=["Acceuil","Modèles","Aide"],
icons=["house-door-fill","box-fill","info-square-fill"],
default_index=0,  #kibda mn kluwl
orientation="horizontal",
styles={
    "container":{"padding":"0"},
    "icon":{},
    "nav-link":{},
    "nav-link-selected":{"background-color":"#0d3b66"},

},
)
df = None
if selected=="Acceuil":
    tab1,tab2,tab3=st.tabs(["Creation du dataset","Modification des données ","Importation des données"])
    with tab1:
        main()
    with tab2:
        modify_page()
    with tab3:
        def calculate_variance(data):
            target_var = data.iloc[:, -1]
            variance = target_var.var()
            print(variance)
            return variance
        st.header('Importation des données ')
        uploaded_file = st.file_uploader("Importer ici votre fichier (CSV)", type=["csv"])
        @st.cache_data(persist=True,experimental_allow_widgets=True)
        def data(uploaded_file): 

            df = pd.DataFrame()
            if uploaded_file is not None:

                
                data = pd.read_csv(uploaded_file)
                df=data.copy()
                        
            return df
        df=data(uploaded_file) 
        if not df.empty:
            target_var = df.iloc[:, -1]
            print(type(target_var[1]))
            if type(target_var[1]) == np.float64:
                variance = calculate_variance(df)

                if variance < 2:
                    st.subheader(f"Selon votre variable cible {df.columns[-1]}, vous devriez utiliser des modèles de calassification.")        
                else:
                    st.subheader(f"Selon votre variable cible {df.columns[-1]}, vous devriez utiliser des modèles de régression.")
            else:
                st.subheader(f"Selon votre variable cible {df.columns[-1]}, vous devriez utiliser des modèles de calassification.")          
            
            
if selected=="Modèles":
    option = st.selectbox(' ', ('Classification', 'Régression', 'Clustering'))
    if option == 'Classification':
        classification()
    elif option == 'Régression':
        regression_page()
    elif option == 'Clustering':
        clusteringPage()
            
if selected=="Aide":
    page_aide()  




st.markdown("""
        ***
        #### Created with :heart: by EL GHAZI LOUBNA & ZAOUI HANANE
""")



