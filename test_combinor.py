import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sentence_transformers import SentenceTransformer

# Charger un modèle multilingue qui fonctionne bien en français
# Nécessite d'installer: pip install sentence-transformers
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def calculate_phrase_similarity(phrases):
    """
    Calcule une matrice de similarité sémantique entre phrases
    en utilisant un modèle de transformer pré-entraîné.
    """
    # Calculer les embeddings pour toutes les phrases
    embeddings = model.encode(phrases)
    
    # Calculer la similarité cosinus entre toutes les paires d'embeddings
    n = len(phrases)
    similarity_matrix = np.zeros((n, n))
    
    # Remplir la matrice de similarité
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Une phrase est identique à elle-même
            else:
                # Calcul de la similarité cosinus entre les embeddings
                similarity_matrix[i, j] = model.similarity(embeddings[i], embeddings[j])
    
    return similarity_matrix

def extract_representative_concept(phrase):
    words = phrase.lower().split()
    stop_words = {'le', 'la', 'les', 'des', 'du', 'de', 'un', 'une', 'à', 'au', 'aux', 'par', 'pour', 'en', 'avec', 'est', 'sont'}
    important_words = [w for w in words if w not in stop_words]
    concept_words = important_words[:min(3, len(important_words))]
    concept = ' '.join(concept_words)
    if concept:
        concept = concept[0].upper() + concept[1:]
    return concept

def cluster_phrases(phrases, similarity_threshold=0.3):
    """
    Regroupe les phrases par similarité et retourne un dictionnaire simple.
    
    Args:
        phrases: Liste des phrases à regrouper
        similarity_threshold: Seuil minimal de similarité
        
    Returns:
        Un dictionnaire {concept: {"représentant": phrase_complète, "phrases": [liste_des_phrases]}}
    """
    similarity_matrix = calculate_phrase_similarity(phrases)
    distance_matrix = 1 - similarity_matrix
    
    n = len(phrases)
    condensed_distance = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed_distance.append(distance_matrix[i, j])
    
    linked = linkage(condensed_distance, method='average')
    
    distance_threshold = 1 - similarity_threshold
    labels = fcluster(linked, distance_threshold, criterion='distance') - 1
    
    clusters_by_label = {}
    for i, label in enumerate(labels):
        if label not in clusters_by_label:
            clusters_by_label[label] = []
        clusters_by_label[label].append(phrases[i])
    
    concept_clusters = {}
    
    for label, phrase_list in clusters_by_label.items():
        if len(phrase_list) == 1:
            representative_phrase = phrase_list[0]
        else:
            avg_similarities = []
            indices = [phrases.index(p) for p in phrase_list]
            
            for idx in indices:
                similarities = [similarity_matrix[idx, other_idx] for other_idx in indices if idx != other_idx]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                avg_similarities.append((avg_similarity, idx))
            
            representative_phrase = phrases[max(avg_similarities, key=lambda x: x[0])[1]]
        
        concept = extract_representative_concept(representative_phrase)
        
        base_concept = concept
        suffix = 1
        while concept in concept_clusters:
            concept = f"{base_concept} {suffix}"
            suffix += 1
        
        concept_clusters[concept] = {
            "représentant": representative_phrase,
            "phrases": phrase_list
        }
    
    return concept_clusters

# Exemple d'utilisation
if __name__ == "__main__":
    phrases = [
        "Le développement durable implique l'équilibre des ressources naturelles.",
        "La gestion des ressources naturelles est essentielle pour la durabilité.",
        "Les systèmes complexes montrent des propriétés émergentes.",
        "L'émergence est une caractéristique des systèmes adaptatifs complexes.",
        "La transformation des énergies est régie par des lois thermodynamiques.",
        "Les flux d'énergie sont soumis aux principes de la thermodynamique.",
        "L'adaptation aux changements environnementaux favorise l'évolution.",
        "L'évolution des espèces est influencée par leur capacité d'adaptation.",
        "Les réseaux d'information facilitent le partage des connaissances.",
        "Le partage d'information est amplifié par les structures en réseau.",
        "il adore manger des pommes",
        "les pommes sont ses fruits préférés",
    ]
    
    # Clustering avec un seuil de similarité - valeur ajustée pour fonctionner
    # avec les similarités sémantiques (généralement plus faibles)
    result = cluster_phrases(phrases, 0.4)
    
    # Afficher les représentants de chaque cluster
    print("Liste des phrases représentatives par cluster:")
    for concept, info in result.items():
        print(f"• {info['représentant']}")
    
    # Créer une liste des représentants pour utilisation ultérieure
    representants = [info['représentant'] for info in result.values()]
    
    # Pour voir chaque cluster avec son représentant et ses phrases
    print("\nDétail des clusters:")
    for i, (concept, info) in enumerate(result.items()):
        print(f"\nCluster {i+1}:")
        print(f"Représentant: {info['représentant']}")
        if len(info['phrases']) > 1:
            print("Phrases similaires:")
            for phrase in info['phrases']:
                if phrase != info['représentant']:
                    print(f"- {phrase}")
                    
    