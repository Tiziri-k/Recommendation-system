# Recommendation-system

## Description

Ce projet implémente une plateforme de recommandation d'articles en ligne en utilisant des techniques d'apprentissage automatique et de traitement d'images. La plateforme offre trois fonctionnalités principales : 

1. **Recherche par image** : L'utilisateur charge une image, et la plateforme propose les 10 articles les plus similaires visuellement.
2. **Recherche par texte** : L'utilisateur saisit une description textuelle, et la plateforme propose les 10 articles les plus similaires en termes de description.
3. **Recherche combinée** : L'utilisateur fournit à la fois une image et une description textuelle. La plateforme combine les similarités visuelles et textuelles pour proposer les 10 articles les plus pertinents.

## Fonctionnalités

### 1. **Recherche par Image**

L'utilisateur charge une image d'un article, et la plateforme recherche les articles les plus visuellement similaires. Cette fonctionnalité repose sur des embeddings d'images générés à l'aide de modèles de deep learning.

- **Entrée** : Une image (format PNG, JPG, etc.)
- **Sortie** : Affichage des 10 articles les plus similaires à l'image, accompagnés de leurs images.

### 2. **Recherche par Texte**

L'utilisateur entre une description textuelle d'un article. La plateforme propose ensuite les 10 articles les plus similaires en termes de description textuelle.

- **Entrée** : Un texte (description d'un article)
- **Sortie** : Affichage des 10 articles les plus similaires à la description, avec leurs titres, descriptions et images.

### 3. **Recherche Combinée**

L'utilisateur fournit à la fois une image et un texte. La plateforme combine les similarités visuelles et textuelles pour proposer les 10 articles les plus pertinents.

- **Entrée** : Une image et une description textuelle.
- **Sortie** : Affichage des 10 articles les plus pertinents en tenant compte à la fois des similarités visuelles et textuelles.

## Approche Technique

### 1. **Calcul de Similarité**

La similarité entre les articles est calculée en utilisant la **cosine similarity**. Cette métrique évalue la similarité entre les embeddings des articles. Les embeddings sont des représentations vectorielles des images et des textes qui capturent les caractéristiques importantes des articles.

Formule de la cosine similarity :

\[
\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
\]

Où :
- \( A \) et \( B \) sont les vecteurs d'embeddings des deux articles.
- \( \cdot \) est le produit scalaire entre les deux vecteurs.
- \( \|A\| \) et \( \|B\| \) sont les normes des vecteurs \( A \) et \( B \).

### 2. **Pondération des Similarités (Option Combinée)**

Pour l'option combinée, la similarité entre l'image et le texte est calculée séparément. Ensuite, ces deux valeurs sont combinées à l'aide d'une pondération pour obtenir un score final.

\[
\text{similarity\_combined} = w_{\text{image}} \times \text{similarity\_image} + w_{\text{text}} \times \text{similarity\_text}
\]

Où :
- \( w_{\text{image}} \) et \( w_{\text{text}} \) sont des poids ajustables.
- \( \text{similarity\_image} \) et \( \text{similarity\_text} \) sont les scores de similarité respectifs pour l'image et le texte.

## Validation

La plateforme a été testée sur des pratiques pour évaluer la pertinence des recommandations. Les résultats montrent que la plateforme est capable de fournir des recommandations précises en fonction de l'image ou de la description textuelle fournie par l'utilisateur. La fonctionnalité combinée améliore la précision en tenant compte des deux types de données.

## Installation

### Prérequis

- Python 3.7+
- Les librairies suivantes doivent être installées :
  - `streamlit` : Pour créer l'interface interactive.
  - `numpy` : Pour les calculs matriciels.
  - `pandas` : Pour manipuler les données.
  - `scikit-learn` : Pour les modèles de machine learning et la réduction dimensionnelle.
  - `torch` : Pour les modèles de deep learning (si utilisés pour générer les embeddings).

Vous pouvez installer les dépendances avec :

```bash
pip install -r requirements.txt
