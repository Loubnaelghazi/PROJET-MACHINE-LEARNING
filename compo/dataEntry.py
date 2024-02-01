import streamlit as st
import pandas as pd

def save_data_to_csv(data, feature_names,fileName):
    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(f'{fileName}.csv', index=False)
    st.success(F'Les données sont enregistrées dans le fichier {fileName}.csv')

def main():
        st.header("Création de l'ensemble de données")

        num_features = st.number_input("Nombre de Features", min_value=1, value=1, step=1)

        feature_names = []
        data = []

        for i in range(num_features):
            feature_name = st.text_input(f"Entrer le nom de la Feature {i+1}", key=f"feature_name_{i}")
            if feature_name.strip() != "" and feature_name not in feature_names:
                feature_names.append(feature_name)

        
        additional_feature_names_placeholder = st.empty()

        
        for i in range(num_features, len(feature_names)):
            feature_name = additional_feature_names_placeholder.text_input(f"Entrer le nom de la Feature {i+1}", key=f"additional_feature_name_{i}")
            if feature_name.strip() != "" and feature_name not in feature_names:
                feature_names.append(feature_name)

        row_count = 0
        while True:
            row = []
            for i, feature_name in enumerate(feature_names):
                value = st.text_input(f"Entrez la valeur pour {feature_name}", key=f"value_{row_count}_{i}")
                row.append(value)
            data.append(row)

            add_another_row = st.checkbox(f"Ajouter une autre ligne ", key=f"add_another_row_{row_count}")
            if not add_another_row:
                break

            row_count += 1
        fileName = st.text_input(f"Entrez le nom du fichier :")
        if st.button("Enregister Data"):
            if feature_names:
                st.subheader("Données saisies : ")
                save_data_to_csv(data, feature_names,fileName)
            else:
                st.warning("Aucun nom de caractéristique saisi. Les données n'ont pas été enregistrées.")

        if data and feature_names:
            df = pd.DataFrame(data, columns=feature_names)
            st.dataframe(df)
    

if __name__ == "__main__":
    main()
