import streamlit as st
import pandas as pd

def save_data_to_csv(df, filename):
    df.to_csv(filename, index=False)
    st.success(f"Enregistré sous{filename}")

def modify_page():
    st.header("Éditeur de fichiers CSV")
    st.markdown("<style>div.row-widget.stRadio>div{flex-direction:row;margin-bottom:-10px;}</style>", unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Télécharger le fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Votre ensemble de données:")
        edit_df = df.copy()

        num_rows = len(edit_df)
        num_cols = len(edit_df.columns)

        with st.form(key='edit_form'):
            cols = st.columns(num_cols)

            for j, col in enumerate(cols):
                col.write(edit_df.columns[j])

            for i in range(num_rows):
                cols = st.columns(num_cols)
                for j, col in enumerate(cols):
                    cell_value = edit_df.iat[i, j]
                    edited_value = col.text_input( label=' ',value=cell_value, key=(i, j))
                    edit_df.iat[i, j] = edited_value
                    

            if st.form_submit_button("Enregistrer les modifications"):
                save_data_to_csv(edit_df, uploaded_file.name)

        st.subheader("Données modifiées:")
        st.dataframe(edit_df)

if __name__ == "__main__":
    modify_page()