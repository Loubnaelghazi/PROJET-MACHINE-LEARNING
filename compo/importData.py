import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np 

from compo.Regression import regression_page


def importData_page():
    
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
        
        variance = calculate_variance(df)

        if variance < 2:
            st.header("Classification Models")
            
                
        else:
            st.subheader(f"Selon votre variable cible {df.columns[-1]}, vous devriez utiliser des modèles de régression.")
            clicked = st.button("🔽 Aller aux modèles de régression")
            
            
