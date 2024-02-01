from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import  SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score ,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from io import BytesIO
import json
import joblib

def regression_page():
    st.title("Modèles de Régression")
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
    columns_test = df.columns.tolist()
    columns_test = columns_test[:-1]  

    def add_parameter_ui(clf_name):
        params = dict()
    
        if clf_name == "Gradient Boosting Régression":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            params["max_depth"] = max_depth
            n_estimators = st.sidebar.slider("n_estimators", 50, 300,10)
            params["n_estimators"] = n_estimators
            learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3,0.02)
            params["learning_rate"] = learning_rate
        elif clf_name == "KNN Régression":
            k = st.sidebar.slider("K", 1, 15)
            params["k"] = k
    
        elif clf_name == "SVM Régression":
            C = st.sidebar.slider("C", 0.01, 10.0)
            kernel = st.sidebar.selectbox("Noyau", ["linear", "rbf", "poly", "sigmoid"])
            params["C"] = C
            params["kernel"] = kernel
    
        elif clf_name == "Regression logistique":
            c = st.sidebar.slider("Paramètre de régularisation (C)", 0.01, 10.0)
            solver = st.sidebar.selectbox("Algorithme d'optimisation", ["liblinear", "lbfgs", "sag", "saga"])
            params["C"] = c
            params["solver"] = solver
        
        elif clf_name == "Régression linéaire":
            pass
        
        elif clf_name == "Random Forest Régression":
            st.sidebar.subheader("Hyperparamètres du modèle ")
            n_estimators2 = st.sidebar.slider("Nombre d'estimateurs", 1, 100)
            max_depth2 = st.sidebar.slider("Profondeur maximale", 2, 15)
            min_samples_split = st.sidebar.slider("Nombre minimum d'échantillons pour la division", 2, 10)
            min_samples_leaf = st.sidebar.slider("Nombre minimum d'échantillons dans une feuille", 1, 10)
            params["n_estimators2"] = n_estimators2
            params["max_depth2"] = max_depth2
            params["min_samples_split"] = min_samples_split
            params["min_samples_leaf"] = min_samples_leaf
        elif clf_name == "Arbres de décisions Régression":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            params["max_depth"] = max_depth

        elif clf_name == "Naive Bayes":
            pass

        return params 
       
    def predict_with_model(clf, input_values):
        input_df = pd.DataFrame([input_values], columns=X.columns)
    
        imputer_numeric = SimpleImputer(strategy='mean')
        numeric_columns = input_df.select_dtypes(include=['float64']).columns
        if not numeric_columns.empty:
            input_df[numeric_columns] = imputer_numeric.fit_transform(input_df[numeric_columns])

        categorical_cols = input_df.select_dtypes(include=['object']).columns
        input_df = pd.get_dummies(input_df, columns=categorical_cols)

        # Normalisation des données
        input_df[input_df.columns[:-1]] = scaler.transform(input_df[input_df.columns[:-1]])

        prediction = clf.predict(input_df)[0]
    
        return prediction
    
    def get_classifier(clf_name, params):
    
   
        if clf_name == "KNN Régression":
            clf = KNeighborsRegressor(n_neighbors=params["k"])
    
        elif clf_name == "Regression logistique":
            clf = LogisticRegression(C=params["C"], solver=params["solver"])
        
        elif clf_name == "Naive Bayes":
            clf = GaussianNB()

        elif clf_name == "Régression linéaire":
            clf = LinearRegression()

        elif clf_name == "Arbres de décisions Régression":
            clf = DecisionTreeRegressor(max_depth=params["max_depth"], random_state=100)

        elif clf_name == "Random Forest Régression":
            clf = RandomForestRegressor(max_depth=params["max_depth2"], n_estimators=params["n_estimators2"],
                                        min_samples_leaf=params["min_samples_leaf"],
                                        min_samples_split=params["min_samples_split"], random_state=100)
        elif clf_name == "SVM Régression":
            clf = SVR(C=params["C"], kernel=params["kernel"])
        elif clf_name == "Gradient Boosting Régression":
            clf = GradientBoostingRegressor(n_estimators=params["n_estimators"],learning_rate=params["learning_rate"],max_depth=params["max_depth"])
        else:
            raise ValueError(f"Classifieur non pris en charge : {clf_name}")

        return clf


    def visualize_data(df):
        

        st.plotly_chart(px.scatter(df, x=df.columns[0], y=df.columns[-1], color=df.columns[-1], title='Scatter Plot')) 
        selected_fea = st.selectbox("Sélectionnez la caractéristique pour visualiser", df.columns) 
        bar_chart = px.bar(df, x=df.columns[-1], y=selected_fea, color=df.columns[1], title='Bar Chart')
        st.plotly_chart(bar_chart) 
        scatter_matrix = px.scatter_matrix(df, dimensions=[df.columns[0], df.columns[1], df.columns[1]], color=df.columns[0], title='Scatter Matrix')
        st.plotly_chart(scatter_matrix)


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
                st.write("Nombre de valeurs dupliquées :")
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
            if st.button("Afficher dataset aprés la modification"):
                st.write(df)
        st.write("Traitement des données aberrantes")
        # Ajoutez un sélecteur de caractéristique pour la boîte à moustaches dans votre interface utilisateur
        selected_feature = st.selectbox("Sélectionnez la caractéristique pour la boîte à moustaches à visualiser", df.columns)

        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            boxplot_chart = px.box(df, x=df.columns[-1], y=selected_feature, title=f'Boîte à moustaches pour {selected_feature} ')
            st.plotly_chart(boxplot_chart)

            Q1 = df[selected_feature].quantile(0.25)
            Q3 = df[selected_feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_count = df[(df[selected_feature] < lower_bound) | (df[selected_feature] > upper_bound)].groupby(df.columns[-1]).size()

            st.write(f"Nombre de valeurs aberrantes pour {selected_feature} :")
            st.write(outliers_count)

        else:
            st.plotly_chart(px.histogram(df, x=selected_feature, color=df.columns[-1], title=f'Distribution de {selected_feature} '))

    
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
            Y = df.iloc[:, -1] # la dernière pour Y

            st.write("La taille de votre Dataset", X.shape)
            st.markdown('Votre variable target **Y** :')
            st.info(Y.name)
        
            st.subheader("Visualisation de la dataset")
            visualize_data(df)
            df=df.dropna()
            

        else:
            st.warning("Veuillez charger ou générer des données pour continuer.")
        # Test de prétraitement
       
    classifier_name = st.sidebar.selectbox("Selectionner votre régresseur", { "Arbres de décisions Régression" , "Random Forest Régression" , "SVM Régression","KNN Régression","Régression linéaire","Gradient Boosting Régression"})

    with tab2:
        st.write("""
        # Résultats et performance  
         """)
        if not df.empty:

            X = df.iloc[:, 1:]  # toutes les colonnes pour X
            Y = df.iloc[:, 1]  # la dernière pour Y
            params = add_parameter_ui(classifier_name)
            clf = get_classifier(classifier_name, params) 
            st.write("Le régresseur choisi :",classifier_name)
            # Imputation des valeurs manquantes pour les caractéristiques numériques
            imputer_numeric = SimpleImputer(strategy='mean')
            numeric_columns = X.select_dtypes(include=['float64']).columns

            if not numeric_columns.empty:
                X[numeric_columns] = imputer_numeric.fit_transform(X[numeric_columns])
            else:
                st.warning("Aucune colonne numérique trouvée. Veuillez vérifier vos données.")
            categorical_cols = df.select_dtypes(include=['object']).columns
            df = pd.get_dummies(df, columns=categorical_cols)

            # Normalisation des données
            scaler = StandardScaler()
            df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

    
            # 
            X = df[df.columns[:-1]]
            Y_column_names = df.columns.difference(X.columns).tolist()

            if Y_column_names:
                target_name = Y_column_names[0]
                Y = df[target_name]
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=64) 
                clf.fit(X_train, Y_train)  # Entrainement du modèle
                y_pred = clf.predict(X_test)  # Prédictions sur l'ensemble de test
                
                mse = mean_squared_error(Y_test, y_pred)
                cv_scores = cross_val_score(clf, X, Y, cv=5)
                mae = mean_absolute_error(Y_test, y_pred)
                r2 = r2_score(Y_test, y_pred)
                st.write(f"Mean Squared Error :{mse}")
                st.write(f"Mean Absolute Error :{mae}")
                st.write(f"R-squared (R²) :{r2}")
                
                results_list = []
                result_dict = {
                    'Régresseur': classifier_name,
                    'Mean Squared Error ': mse,
                    'Mean Absolute Error': mae,
                    'R-squared (R²)': r2,   
                }
                save_model(clf)
                results_list.append(result_dict)

                st.write("Résultats exportables :")
                results_df = pd.DataFrame(results_list)
                export_results(results_df)
                              
            else:

                st.warning("Aucune colonne de variable cible trouvée. Veuillez vérifier vos données.")
        else:
            st.warning("Veuillez charger ou générer des données pour continuer.")

    with tab3:
            st.header("Tester le Modèle")
            print(columns_test)
            if not df.empty:
                
                df_test = pd.DataFrame(columns=columns_test)
                print("df",df_test)
                def add_row():
                    row = []
                    for column in columns_test:
                        value = st.text_input(f"Entrez une valeur pour {column}")
                        row.append(value)
                    df_test.loc[len(df)] = row
                def perform_prediction(df_test): 
                    prediction = clf.predict(df_test)
                    return prediction
                
                add_row()
                if st.button("Tester"):
                    prediction = perform_prediction(df_test)
                    st.write("La prédiction du modèle est : " , prediction)
            else:
                st.warning("Veuillez charger ou générer des données pour continuer.")
            


















        




















