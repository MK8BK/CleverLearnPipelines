import os
import openai
import numpy as np
from dotenv import load_dotenv

# Charge le fichier .env depuis src/prompters/.env
load_dotenv(dotenv_path="src/prompters/.env")

# Récupère la clé API depuis la variable d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Clé API introuvable. Vérifiez votre fichier .env.")

# Utilise les fonctions utilitaires pour embeddings
from openai.embeddings_utils import get_embedding, cosine_similarity

def filter_similar_concepts(concepts, threshold=0.85, engine="text-embedding-ada-002"):
    """
    Calcule l'embedding de chaque concept et conserve un seul représentant par groupe
    de concepts trop similaires (selon le seuil défini).
    """
    embeddings = {concept: get_embedding(concept, engine=engine) for concept in concepts}
    representatives = []
    for concept in concepts:
        if any(cosine_similarity(embeddings[concept], embeddings[rep]) >= threshold
               for rep in representatives):
            continue
        representatives.append(concept)
    return representatives

if __name__ == "__main__":
    # Liste d'exemple de concepts ou phrases
    concepts = [
        "intelligence artificielle",
        "IA",
        "apprentissage automatique",
        "machine learning",
        "réseaux de neurones"
    ]

    reps = filter_similar_concepts(concepts, threshold=0.85)
    print("Concepts représentatifs :", reps)
