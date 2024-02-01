
import streamlit as st

def page_intro_modeles_supervises():
    st.title("Introduction aux Modèles Supervisés")

    st.write("""
        Les modèles supervisés sont des algorithmes d'apprentissage automatique qui apprennent à partir de données étiquetées. 
        Ils sont utilisés pour prédire ou classer de nouvelles données en se basant sur des exemples passés.
    """)

    st.header("Différence entre Régression et Classification")

    st.write("""
        - **Régression**: Utilisée pour prédire une valeur numérique. Par exemple, prédire le prix d'une maison en fonction de ses caractéristiques.
        - **Classification**: Utilisée pour assigner une catégorie à une observation. Par exemple, prédire si un e-mail est spam ou non.
    """)

    st.header("Modèles Supervisés Courants")

    st.subheader("1. Naïve Bayes")
    st.write("""
        Le modèle Naïve Bayes est basé sur le théorème de Bayes et suppose l'indépendance conditionnelle entre les caractéristiques. 
        Il est souvent utilisé pour la classification de textes, comme la détection de spam.
    """)

    st.subheader("2. KNN (k-plus proches voisins)")
    st.write("""
        KNN attribue une observation à la classe majoritaire parmi ses k voisins les plus proches dans l'espace des caractéristiques. 
        Il est utilisé pour la classification et la régression.
    """)

    st.subheader("3. Arbres de Décision")
    st.write("""
        Les arbres de décision sont des modèles graphiques de prise de décision qui utilisent des règles pour partitionner les données. 
        Ils sont utilisés pour la classification et la régression.
    """)

    st.subheader("4. Gradient Boosting")
    st.write("""
        Gradient Boosting construit une série de modèles faibles, les combine pour former un modèle fort. 
        Il est utilisé pour la classification et la régression.
    """)

    st.subheader("5. Régression Logistique")
    st.write("""
        La régression logistique est utilisée pour prédire la probabilité d'appartenance à une classe. 
        Elle est largement utilisée en classification binaire.
    """)

    st.subheader("6. Régression Linéaire")
    st.write("""
        La régression linéaire modélise la relation linéaire entre une variable dépendante et une ou plusieurs variables indépendantes. 
        Elle est utilisée pour la prédiction numérique.
    """)

    st.subheader("7. Random Forest")
    st.write("""
        Random Forest est un ensemble d'arbres de décision qui agrège les prédictions pour améliorer la performance. 
        Il est utilisé pour la classification et la régression.
    """)
    

    # Section sur les Modèles Non Supervisés
    st.title("Introduction aux Modèles Non Supervisés")

    st.write("""
    Les modèles non supervisés sont des algorithmes d'apprentissage automatique qui travaillent avec des données non étiquetées.
    Ils sont utilisés pour découvrir des structures cachées ou des patterns dans les données.
    """)

    st.header("K-Means Clustering")

    st.write("""
    K-Means est un algorithme de clustering qui partitionne les données en k clusters en minimisant la variance intra-cluster.
    C'est une technique populaire pour regrouper des données similaires ensemble.
    """)

    st.header("Paramètres de K-Means")


    st.header("Différence entre Supervisé et Non Supervisé")
    st.write("""
    Les modèles supervisés et non supervisés diffèrent principalement dans la nature des données qu'ils utilisent et dans leurs objectifs respectifs :

    - **Nature des données**:
        - **Supervisés**: Utilisent des données étiquetées, où chaque exemple est associé à une étiquette ou une réponse connue. L'algorithme apprend à partir de cette correspondance entre les entrées et les sorties.
        - **Non supervisés**: Travaillent avec des données non étiquetées, ne disposant pas d'informations de sortie prédéfinies. L'algorithme explore la structure sous-jacente des données sans indications explicites.

    - **Objectifs**:
        - **Supervisés**: Ont pour objectif de faire des prédictions ou de classer de nouvelles données en se basant sur les modèles appris à partir des exemples étiquetés.
        - **Non supervisés**: Ont pour objectif de découvrir des structures, des patterns ou des relations intrinsèques dans les données, souvent par le biais de regroupements ou de réductions de dimensionnalité.

    - **Utilisation typique**:
        - **Supervisés**: Convient lorsque les étiquettes des données sont disponibles et que l'on souhaite automatiser la prise de décision ou la prédiction.
        - **Non supervisés**: Utiles lorsque les données n'ont pas d'étiquettes explicites, mais que l'on veut explorer des tendances, des similitudes ou des groupes inhérents.


    En résumé, les modèles supervisés sont adaptés à la prédiction sur des données étiquetées, tandis que les modèles non supervisés sont utilisés pour explorer et découvrir des structures dans des données non étiquetées.
    """)