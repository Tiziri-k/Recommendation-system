from langdetect import detect # type: ignore
from transformers import pipeline # type: ignore
import pandas as pd # type: ignore

def load_and_translate():
    data = pd.read_parquet("hf://datasets/DBQ/Chanel.Product.prices.Germany/data/train-00000-of-00001-d681c47b79d4401f.parquet")

    if 'title' not in data.columns:
        raise ValueError("La colonne 'title' n'existe pas dans le dataset")

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")  #
    def translate_title(title):
        try:
            lang = detect(title)
            if lang == 'de':  # Si la langue détectée est l'allemand
                translation = translator(title)[0]['translation_text']
                return translation
            else:
                return title  
        except Exception as e:
            print(f"Erreur lors de la traduction du titre '{title}': {e}")
            return None

    data['traduction_anglais'] = data['title'].apply(translate_title)

    data.to_csv("dataset_traduit.csv", index=False, encoding='utf-8')
    print("Traduction terminée. Dataset sauvegardé")

if __name__ == "__main__":
    load_and_translate()
