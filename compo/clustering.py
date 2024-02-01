import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score ,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from sklearn.cluster import KMeans
import base64
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
import json
from sklearn.preprocessing import LabelEncoder
import zipfile
import io
import plotly.io as pio





def clusteringPage():

    st.title("Modèle de clustering ") 
    st.sidebar.header('Importation des données ')
    uploaded_file = st.sidebar.file_uploader("Importer ici votre fichier (CSV)", type=["csv"])
    tab1,tab2=st.tabs(["Visualisation de l'ensemble de données  ","Résultats du modèle  "])


    @st.cache_data(persist=True,experimental_allow_widgets=True)
    def data(uploaded_file): 

        df = pd.DataFrame()
        if uploaded_file is not None:

            
            data = pd.read_csv(uploaded_file)
            df=data.copy()
                      
        return df
    df=data(uploaded_file) 

    



    def get_classifier(clf_name, params):
    
        

        return clf
   

    
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a') as zip_file:
            for i, graph in enumerate(selected_graphs,start= 1):
                img_bytes = pio.to_image(graph, format="png")
                zip_file.writestr(f"graph_{i}.png", img_bytes)




        zip_buffer.seek(0)
        return base64.b64encode(zip_buffer.read()).decode()



    def visualize_data(df):
        
      
            
        st.write("Visualisation de la dataset sous forme de graphes")
        selected_var = st.selectbox("Sélectionnez la caractéristique à visualiser", df.columns[:-1])

        st.plotly_chart(px.histogram(df, x=selected_var, title=f'Histogramme de {selected_var} '))

        st.plotly_chart(px.scatter(df, x=selected_var, y=df.columns[1], color=df.columns[-1], title='Scatter Plot'))  
        st.plotly_chart(px.bar(df, x=df.columns[-1], title='Distribution des classes cibles'))
        st.plotly_chart(px.pie(df, names=df.columns[-1], title='Répartition des classes cibles'))
        
        st.plotly_chart(px.scatter_matrix(df, dimensions=df.columns[:-1], color=df.columns[-1], title='Nuage de points pour les caractéristiques'))
        st.plotly_chart(px.bar(x=df.columns, y=df.isna().sum(), title='Nombre de valeurs manquantes par caractéristique'))

        nan_counts = df.isna().sum()

        st.write("Nombre de valeurs nulles par colonne :")
        st.write(nan_counts)

        total_nan_count = df.isna().sum().sum()
        st.write(f"Nombre total de valeurs nulles dans l'ensemble du DataFrame avant traitement : {total_nan_count}") 
        df = df.dropna()
        total_nan_count_1 = df.isna().sum().sum()
        st.write(f"Nombre total de valeurs nulles dans l'ensemble du DataFrame après traitement : {total_nan_count_1}")
        remove_duplicates = st.checkbox("Supprimer les valeurs dupliquées", value=False)

        if remove_duplicates:
            if df.duplicated().any():
                st.warning("Des valeurs dupliquées ont été détectées.")
        
                total_duplicates = df.duplicated().sum()
                st.write(f"Nombre total de valeurs dupliquées : {total_duplicates}")
                duplicate_count_by_class = df[df.duplicated()].groupby(df.columns[-1]).size()
                st.write("Nombre de valeurs dupliquées par classe :")
                st.write(duplicate_count_by_class)
                df = df.drop_duplicates()
                st.success("Les valeurs dupliquées ont été supprimées.")
            else:
                st.success("Aucune valeur dupliquée n'a été détectée dans l'ensemble de données.")
        else:
            st.info("Vous avez choisi de ne pas traiter les valeurs dupliquées.")
        remove_columns = st.checkbox("Supprimez les colonnes qui ne sont pas importantes dans le dataset", value=False)
        if remove_columns:
            column_names = df.columns[:-1].tolist()
            selected_columns = st.multiselect("Sélectionnez les columns à supprimer", column_names)
            df = df.drop(columns=selected_columns)
            if st.button("Afficher dataset après la modification"):
                st.write(df)
    
 
        st.write("Traitement des données aberrantes")
        selected_feature = st.selectbox("Sélectionnez la caractéristique pour la boîte à moustaches à visualiser", df.columns )
        selected_graphs = []

        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            boxplot_chart = px.box(df, x=df.columns[-1], y=selected_feature, title=f'Boîte à moustaches pour {selected_feature} par classe')
            st.plotly_chart(boxplot_chart)
            


            Q1 = df[selected_feature].quantile(0.25)
            Q3 = df[selected_feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = df[(df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)].groupby(df.columns[-1]).size()

            st.write(f"Nombre de valeurs aberrantes pour {selected_feature} par classe :")
            st.write(outliers_count)

            

        else:
            st.plotly_chart(px.histogram(df, x=selected_feature, color=df.columns[-1], title=f'Distribution de {selected_feature} par classe'))

            threshold_rare_category = 10  
            rare_category_count = df[df.groupby(df.columns[-1])[selected_feature].transform('count') < threshold_rare_category].groupby(df.columns[-1]).size()

            st.write(f"Nombre de catégories rares pour {selected_feature} par classe :")
            st.write(rare_category_count)
  




        
    
    with tab1:
        st.write("""
        # Visualisation de l'ensemble de données 
    
         """)
        if not df.empty:

            st.write("Votre ensemble de données ")  
            st.write(df)
            X = df.iloc[:, :-1]  # toutes les colonnes pour X
            Y = df.iloc[:, -1]  # la dernière pour Y

            st.write("La taille de votre Dataset", X.shape)
            st.write("Le nombre de classes :", len(np.unique(Y.astype(str))))
            st.markdown('Votre variable target *Y* :')
            st.info(Y.name)
        
            st.subheader("Visualisation de la dataset")
            visualize_data(df)
            df=df.dropna()
            

        else:
            st.warning("Veuillez charger ou générer des données pour continuer.")
        # Test de prétraitement
       
    classifier_name = st.sidebar.text("KMeans ")
    num_clusters = st.sidebar.slider("Entrez le nombre des classes", 1, 15)
    with tab2:

        st.write("""
        # Résultats et performance
        """)

        if not df.empty:
            df=df.dropna()
            results_list = []
            X = df.iloc[:, :-1]  
            Y = df.iloc[:, -1]   
            target = df.columns[-1]
            
            
            kmeans = KMeans(n_clusters=num_clusters)

            scaler = StandardScaler()
            #df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
            
            numeric_columns = X.select_dtypes(include=['float64']).columns
            categorical_cols = X.select_dtypes(include=['object']).columns

            # Imputer les valeurs manquantes pour les colonnes numériques

            if not categorical_cols.empty:
                df = pd.get_dummies(df, columns=categorical_cols)
           
                
            if not numeric_columns.empty:
                imputer_numeric = SimpleImputer(strategy='mean')
                df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            else:
                st.warning("Aucune colonne numérique trouvée. ")
            
            
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(X)  
            cluster_assignments = kmeans.predict(X)
            centres = kmeans.cluster_centers_

            colors = ["y", "g", "c"]
            cluster_colors = [colors[index] for index in cluster_assignments]
            fig, ax = plt.subplots()
            ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=cluster_colors, s=3 )
            ax.scatter(centres[:, 0], centres[:, 1], c='r', marker='*', s=100)
            axes = plt.gca()
            axes.set_axis_off()
            st.pyplot(fig)
            

        else:

            st.warning("Veuillez charger ou générer des données pour continuer.")


    