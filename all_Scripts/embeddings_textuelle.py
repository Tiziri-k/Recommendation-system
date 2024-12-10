import os
import pandas as pd  
from transformers import AutoTokenizer, AutoModel  
import torch  
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt  
import numpy as np  
import re
import seaborn as sns 
from sklearn.cluster import KMeans 


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = text.lower()

    return text


def generate_text_embeddings(input_file: str, output_file: str):

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Le fichier d'entrée '{input_file}' est introuvable. Vérifiez le chemin.")

    
    print(f"Chargement du dataset depuis {input_file}...")
    data = pd.read_csv(input_file)

    if 'traduction_anglais' not in data.columns:
        raise ValueError("La colonne 'traduction_anglais' est absente du dataset.")

    data['traduction_anglais'] = data['traduction_anglais'].apply(clean_text)

    print(data['traduction_anglais'].head())

    print("Chargement du modèle NLP...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def get_embedding(text):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings
        except Exception as e:
            print(f"Erreur lors de la génération de l'embedding pour '{text}': {e}")
            return None

    
    print("Génération des embeddings...")
    data['embeddings'] = data['traduction_anglais'].apply(lambda x: get_embedding(x))

    # Sauvegarder le DataFrame avec les embeddings
    print(f"Sauvegarde des résultats dans {output_file}...")
    data.to_parquet(output_file, index=False)
    print("Génération des embeddings terminée avec succès !")


def visualize_embeddings(parquet_file: str):
    """
    Visualise les embeddings textuels avec t-SNE.
    """
    print(f"Chargement des embeddings depuis {parquet_file}...")
    data = pd.read_parquet(parquet_file)

    if 'embeddings' not in data.columns:
        raise ValueError("La colonne 'embeddings' est absente du fichier chargé.")

    embeddings = np.array(data['embeddings'].tolist())

    kmeans = KMeans(n_clusters=49, random_state=42)  
    data['cluster'] = kmeans.fit_predict(embeddings)
    
    print("Réduction de dimension avec t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],  
                hue=data['cluster'], palette='tab10', s=100, edgecolor='k', 
                alpha=0.7, legend='full')  
    
    print("Visualisation des embeddings...")
    plt.title("Visualisation des Embeddings avec t-SNE")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


def compare_embeddings(embedding1, embedding2):
 
    cosine_sim = cosine_similarity([embedding1], [embedding2]) # type: ignore
    return cosine_sim[0][0]


def analyze_embeddings(data):

    similar_example_1 = data['embeddings'][0]
    similar_example_2 = data['embeddings'][4]  
    different_example_1 = data['embeddings'][0]
    different_example_2 = data['embeddings'][50]  

    similarity_similar = compare_embeddings(similar_example_1, similar_example_2)
    similarity_different = compare_embeddings(different_example_1, different_example_2)

    print(f"Similarité entre produits similaires : {similarity_similar}")
    print(f"Similarité entre produits différents : {similarity_different}")

    cluster_0 = data[data['cluster'] == 0]['traduction_anglais'].tolist()
    cluster_1 = data[data['cluster'] == 1]['traduction_anglais'].tolist()

    print(f"Exemples de produits similaires dans le cluster 0 : {cluster_0[:5]}")
    print(f"Exemples de produits similaires dans le cluster 1 : {cluster_1[:5]}")


if __name__ == "__main__":
    input_file = "dataset_traduit.csv" 
    output_file = "dataset_embeddings.parquet"  

    generate_text_embeddings(input_file, output_file)

    visualize_embeddings(output_file)
