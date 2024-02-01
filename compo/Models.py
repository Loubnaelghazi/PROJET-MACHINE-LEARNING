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
import base64
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
import json
import zipfile
import io
import plotly.io as pio





def classification():

    st.title("Modèles de classification ") 
    st.sidebar.header('Importation des données ')
    uploaded_file = st.sidebar.file_uploader("Importer ici votre fichier (CSV)", type=["csv"])
    tab1,tab2,tab3=st.tabs(["Visualisation de l'ensemble de données  ","Résultats du modèle  ","Test"])


    @st.cache_data(persist=True,experimental_allow_widgets=True)
    def data(uploaded_file): 

        df = pd.DataFrame()
        if uploaded_file is not None:

            
            data = pd.read_csv(uploaded_file)
            df=data.copy()
                      
        return df
    df=data(uploaded_file) 

    def add_parameter_ui(clf_name):


        params = dict()
        if clf_name == "KNN(le plus prochain voisin)":

            k = st.sidebar.slider("K", 1, 15)
            params["k"] = k
    
        elif clf_name == "SVM (Support Vector machine Classifier)":
            C = st.sidebar.slider("C", 0.01, 10.0)
            kernel = st.sidebar.selectbox("Noyau", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = C
            params["kernel"] = kernel
    
        elif clf_name == "Arbres de décisions":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            params["max_depth"] = max_depth
        elif clf_name == "Regression logistique":
            c = st.sidebar.slider("Paramètre de régularisation (C)", 0.01, 10.0)
            solver = st.sidebar.selectbox("Algorithme d'optimisation", ["liblinear", "lbfgs", "sag", "saga"])
            params["C"] = c
            params["solver"] = solver
        elif clf_name == "Random Forest":
            st.sidebar.subheader("Hyperparamètres du modèle ")
            n_estimators2 = st.sidebar.slider("Nombre d'estimateurs", 1, 100)
            max_depth2 = st.sidebar.slider("Profondeur maximale", 2, 15)
            min_samples_split = st.sidebar.slider("Nombre minimum d'échantillons pour la division", 2, 10)
            min_samples_leaf = st.sidebar.slider("Nombre minimum d'échantillons dans une feuille", 1, 10)
            params["n_estimators2"] = n_estimators2
            params["max_depth2"] = max_depth2
            params["min_samples_split"] = min_samples_split
            params["min_samples_leaf"] = min_samples_leaf

        elif clf_name == "Naive Bayes":
            pass

        return params



    def get_classifier(clf_name, params):
    
        if clf_name == "KNN(le plus prochain voisin)":
            clf = KNeighborsClassifier(n_neighbors=params["k"])

        elif clf_name == "SVM (Support Vector machine Classifier)":
            clf = SVC(C=params["C"], kernel=params["kernel"])
        elif clf_name == "Arbres de décisions":
            clf = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=100)
        elif clf_name == "Regression logistique":

            clf = LogisticRegression(C=params["C"], solver=params["solver"])
        elif clf_name == "Random Forest":

            clf = RandomForestClassifier(max_depth=params["max_depth2"], n_estimators=params["n_estimators2"],
                                     min_samples_leaf=params["min_samples_leaf"],
                                     min_samples_split=params["min_samples_split"], random_state=100)
        elif clf_name == "Naive Bayes":
         
            clf = GaussianNB()

    
        else:
            raise ValueError(f"Classifieur non pris en charge : {clf_name}")

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

        # Scatter plot interactif
        st.plotly_chart(px.scatter(df, x=df.columns[0], y=df.columns[1], color=df.columns[-1], title='Scatter Plot'))  
        st.plotly_chart(px.bar(df, x=df.columns[-1], title='Distribution des classes cibles'))
        st.plotly_chart(px.pie(df, names=df.columns[-1], title='Répartition des classes cibles'))
        selected_feature = st.selectbox("Sélectionnez la caractéristique à visualiser", df.columns[:-1],key="selectbox2")

        st.plotly_chart(px.histogram(df, x=selected_feature ,color=df.columns[-1], title=f'Distribution de {selected_feature} par classe'))
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
        selected_feature = st.selectbox("Sélectionnez la caractéristique pour la boîte à moustaches à visualiser", df.columns)
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
  



    def save_model(clf, filename="model.pkl"):
        joblib.dump(clf, filename)
        st.success(f"Le modèle a été sauvegardé sous le nom {filename}.")
    def export_results(results_list):
      
        csv_data = pd.DataFrame(results_list).to_csv(index=False).encode()
        csv_file = BytesIO(csv_data)
        st.download_button(
            label="Exporter les résultats CSV",
            data=csv_file,
            file_name="results_export.csv",
            key="export_results",
         )


        

        
    
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
       
    classifier_name = st.sidebar.selectbox("Selectionner votre classifieur", {"Regression logistique", "Naive Bayes", "SVM (Support Vector machine Classifier)", "Arbres de décisions", "KNN(le plus prochain voisin)","Random Forest" })
    with tab2:

        st.write("""
        # Résultats et performance
        """)

        if not df.empty:
            df=df.dropna()
            results_list = []
            X = df.iloc[:, :-1]  
            Y = df.iloc[:, -1]   

            params = add_parameter_ui(classifier_name)
            clf = get_classifier(classifier_name, params)

            scaler = StandardScaler()
            #df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

            st.write("Le classifieur choisi :", classifier_name)
            
            numeric_columns = X.select_dtypes(include=['float64']).columns
            categorical_cols = X.select_dtypes(include=['object']).columns

            # Imputer les valeurs manquantes pour les colonnes numériques

            if not categorical_cols.empty:
                df = pd.get_dummies(df, columns=categorical_cols)
            else:
                st.warning("Aucune colonne catégorique trouvée.")
            if not numeric_columns.empty:
                imputer_numeric = SimpleImputer(strategy='mean')
                df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
                df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            else:
                st.warning("Aucune colonne numérique trouvée. ")
            
            # Data split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=64)
            indices_nan = Y_test[pd.isna(Y_test)].index
            Y_test_without_nan = Y_test.drop(index=indices_nan)

            # Data preprocessing
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            X_test_scaled = scaler.transform(X_test)

            clf.fit(X_train_scaled, Y_train)  
            y_pred = clf.predict(X_test_scaled)  

            # Evaluation and visualization
            acc = accuracy_score(Y_test, y_pred)
            precision = precision_score(Y_test, y_pred,average='weighted')
            recall = recall_score(Y_test, y_pred,average='weighted')
            unique_classes = np.unique(np.concatenate((Y_test, y_pred)))
            class_labels = [f"Classe {label}" for label in unique_classes]

            st.write("La matrice de confusion ")
            conf_matrix = confusion_matrix(Y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
            plt.xlabel("Prédiction")
            plt.ylabel("Réelle")
            st.pyplot(fig)

           

            # Rapport de classification
            st.write("Rapport de Classification :")
            st.write(f"Accuracy : {acc}")
            st.write(f"Precision : {precision}")
            st.write(f"Recall : {recall}")
            st.write(f"Exactitude du modèle : {acc:.2%}")

            results_list = []
            result_dict = {
                'Classifier': classifier_name,
                'Accuracy': acc,
                'Precision': precision,
                'Recall': recall,
                'Matrice de confusion': conf_matrix.tolist()   
             }
            save_model(clf)
            results_list.append(result_dict)

            st.write("Résultats exportables :")
            results_df = pd.DataFrame(results_list)
            export_results(results_df)

        else:

            st.warning("Veuillez charger ou générer des données pour continuer.")


    with tab3:
        st.write("""
        # Tester le Modèle
        """)

        if not df.empty:
            df_test = pd.DataFrame()
            column_names = X_train.columns.tolist()
            selected_columns = st.multiselect("Sélectionnez les fonctionnalités à utiliser", column_names)

            if not selected_columns:
                st.warning("Veuillez sélectionner au moins une fonctionnalité.")
            else:
                df_test = pd.DataFrame(columns=selected_columns)

                def add_row():
                    row = []
                    for column in selected_columns:
                        value = st.text_input(f"Entrez une valeur pour {column}")
                        row.append(value)
                    df_test.loc[len(df_test)] = row
                    df_test_updated = df_test[X_train.columns.intersection(selected_columns)]
                    return pd.DataFrame([row], columns=df_test_updated.columns)

                def perform_prediction(df_test):
                    df_test = df_test[X_train.columns.tolist()]
                    #df_test = df_test.dropna() 
                    df_test_scaled = scaler.transform(df_test)

                    prediction = clf.predict(df_test_scaled)
                    return prediction

                add_row()

                if st.button("Tester"):
                    prediction = perform_prediction(df_test)
                    st.write("La prédiction du modèle est :", prediction)
        else:
            st.warning("Veuillez charger ou générer des données pour continuer.")