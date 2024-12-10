from io import BytesIO
import pandas as pd # type: ignore
from modules.translator import load_and_translate
import streamlit as st # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
import requests  # type: ignore
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.applications import InceptionV3 # type: ignore
import tensorflow_hub as hub # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore


# Charger les données
@st.cache_data
def load_data():
    file_path = "hf://datasets/DBQ/Chanel.Product.prices.Germany/data/train-00000-of-00001-d681c47b79d4401f.parquet"
    return pd.read_parquet(file_path)

# Extraire les embeddings des images avec InceptionV3
@st.cache_resource
def load_image_model():
    # Modèle préentraîné (InceptionV3)
    model = InceptionV3(include_top=False, pooling='avg', weights='imagenet')
    return model


def preprocess_image(image_url):
    """
    Prépare l'image pour l'entrée dans le modèle InceptionV3.
    """
    from PIL import Image # type: ignore
    response = requests.get(image_url)
    response.raise_for_status() 
    # Ouvrir l'image (PIL)
    img = Image.open(BytesIO(response.content))
    
    # Si l'image est RGBA, convertissez-la en RGB
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    # Redimensionner l'image à (299, 299) (attendu par InceptionV3)
    img = img.resize((299, 299))
    
    # Convertir en tableau numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Ajouter une dimension pour le batch
    img_array = tf.expand_dims(img_array, axis=0)
    
    # Normaliser les valeurs des pixels (entre -1 et 1)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    st.write("Dimensions de l'image prétraitée :", img_array.shape)

    return img_array


def extract_image_features(image, model):
    img_array = preprocess_image(image)
    features = model.predict(img_array)
    return features

def find_similar_images(uploaded_image_features, dataset_features, top_n=10):
    """
    Trouve les N images les plus similaires visuellement dans le dataset.
    """
    # Calcul de la similarité cosinus
    similarities = cosine_similarity(uploaded_image_features, dataset_features)
    
    # Obtenir les indices des N plus grandes similarités
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    
    return similar_indices

def vectorize_text(text):
    from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([text])
    return text_vector.toarray().flatten()  # Aplatir le vecteur de caractéristiques


# Calcul de la similarité cosinus
def calculate_cosine_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])  # Calcul de la similarité cosinus
    return similarity[0][0]

# Fonction de pondération de la similarité
def weighted_similarity(cosine_similarity_image, cosine_similarity_text, weight_image=0.5, weight_text=0.5):
    combined_similarity = (weight_image * cosine_similarity_image) + (weight_text * cosine_similarity_text)
    return combined_similarity

def download_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def find_similar_images(uploaded_image_features, dataset_features, top_n=10):
    similarities = cosine_similarity(uploaded_image_features, dataset_features)
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]
    return similar_indices

def main():
    st.title("Système de recommandation d'articles par image")

    # Afficher le bouton pour télécharger l'image
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Si l'image est téléchargée, l'afficher
        st.image(uploaded_file, caption="Image téléchargée", use_container_width=True)

        # Charger le modèle
        image_model = load_image_model()

        # Extraire les caractéristiques de l'image
        uploaded_image_features = extract_image_features(uploaded_file, image_model)

        # Charger les données
        data = load_data()

      

        # Assurer que la colonne contenant les URLs des images est bien présente
        if 'imageurl' not in data.columns:
            st.error("La colonne 'imageurl' n'existe pas dans le dataset.")
            return

        # Extraire les caractéristiques des images du dataset
        def process_image_from_url(url):
            image = download_image_from_url(url)
            return extract_image_features(image, image_model)

        data['image_features'] = data['imageurl'].apply(process_image_from_url)

        # Calculer la similarité cosinus avec les caractéristiques extraites
        similarities = data['image_features'].apply(
            lambda features: cosine_similarity([uploaded_image_features], [features])[0][0]
        )

        # Trier les produits les plus similaires
        top_10 = data.iloc[similarities.nlargest(10).index]
        st.write("### Articles recommandés :")
        for index, row in top_10.iterrows():
            st.image(row['imageurl'], caption=row['title'])  # Afficher l'image de l'article
            st.write(f"title : {row['title']}")
# Extraire les caractéristiques des images du dataset
        def process_image_from_url(url):
            image = download_image_from_url(url)
            return extract_image_features(image, image_model)

        data['image_features'] = data['imageurl'].apply(process_image_from_url)

        # Calculer la similarité cosinus avec les caractéristiques extraites
        similarities = data['image_features'].apply(
            lambda features: cosine_similarity([uploaded_image_features], [features])[0][0]
        )

        # Trier les produits les plus similaires
        top_10 = data.iloc[similarities.nlargest(10).index]
        st.write("### Articles recommandés :")
        for index, row in top_10.iterrows():
            st.image(row['imageurl'], caption=row['title'])  # Afficher l'image de l'article
            st.write(f"title : {row['title']}")


if __name__ == "__main__":
    main()

# Vectoriser les titles textuelles
@st.cache_resource
def vectorize_text(data):
    vectorizer = TfidfVectorizer()
    text_embeddings = vectorizer.fit_transform(data)
    return vectorizer, text_embeddings

# Charger les données
data = load_data()

# Charger le modèle d'images
image_model = load_image_model()

# Préparer les embeddings textuels
vectorizer, text_embeddings = vectorize_text(data['title'])

# Titre de l'application
st.title("Système de Recommandation - Fonctionnalités")

# Menu de sélection
option = st.selectbox(
    "Choisissez une option :",
    ["Recherche par image", "Recherche par texte", "Recherche combinée (image + texte)"]
)

if option == "Recherche par image":
    st.write("### Option 1 : Recherche par image")
    uploaded_image = st.file_uploader("Téléchargez une image :", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)

        
        # Extraire les caractéristiques de l'image téléchargée
        uploaded_image_features = extract_image_features(image, image_model)
        
        # Calculer la similarité avec les images du dataset
        data['image_features'] = data['imageurl'].apply(
            lambda path: extract_image_features(Image.open(path), image_model)
        )
        similarities = data['image_features'].apply(
            lambda features: cosine_similarity(
                [uploaded_image_features], [features]
            )[0][0]
        )
        
        # Trier les produits les plus similaires
        top_10 = data.iloc[similarities.nlargest(10).index]
        st.write("### Articles recommandés :")
        for index, row in top_10.iterrows():
            st.image(row['imageurl'], caption=row['title'])
            st.write(f"title : {row['title']}")

elif option == "Recherche par texte":
    st.write("### Option 2 : Recherche par texte")
    title = st.text_input("Entrez une title textuelle :")
    if title:
        st.write(f"title fournie : {title}")
        
        # Vectoriser la title utilisateur
        user_vector = vectorizer.transform([title])
        
        # Calculer la similarité avec les titles du dataset
        similarities = cosine_similarity(user_vector, text_embeddings).flatten()
        
        # Trier les produits les plus similaires
        top_10 = data.iloc[similarities.argsort()[-10:][::-1]]
        st.write("### Articles recommandés :")
        for index, row in top_10.iterrows():
            st.image(row['imageurl'], caption=row['title'])
            st.write(f"title : {row['title']}")

elif option == "Recherche combinée (image + texte)":
    st.write("### Option 3 : Recherche combinée")
    uploaded_image = st.file_uploader("Téléchargez une image :", type=["jpg", "png", "jpeg"])
    title = st.text_input("Entrez une title textuelle :")
    
    if uploaded_image and title:
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)

        st.write(f"title fournie : {title}")
        
        # Extraire les caractéristiques de l'image
        uploaded_image_features = extract_image_features(image, image_model)
        
        # Vectoriser la title utilisateur
        user_vector = vectorizer.transform([title])
        
        # Combiner les similarités
        data['combined_score'] = data.apply(
            lambda row: 0.5 * cosine_similarity(
                [uploaded_image_features], [row['image_features']]
            )[0][0] + 0.5 * cosine_similarity(
                user_vector, text_embeddings[row.name]
            )[0][0],
            axis=1
        )
        
        # Trier les produits les plus similaires
        top_10 = data.iloc[data['combined_score'].nlargest(10).index]
        st.write("### Articles recommandés :")
        for index, row in top_10.iterrows():
            st.image(row['imageurl'], caption=row['title'])
            st.write(f"title : {row['title']}")
