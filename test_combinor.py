import difflib

def group_by_similarity(concepts, threshold=0.3):
    """
    Regroupe les concepts selon une similarité calculée avec difflib.
    Pour chaque concept, on recherche un représentant déjà existant ayant une similarité
    (ratio) >= threshold. Si trouvé, le concept est ajouté au groupe correspondant ;
    sinon, un nouveau groupe est créé avec le concept comme représentant.
    """
    groups = {}
    for concept in concepts:
        assigned = False
        for rep in groups:
            if difflib.SequenceMatcher(None, concept, rep).ratio() >= threshold:
                groups[rep].append(concept)
                assigned = True
                break
        if not assigned:
            groups[concept] = [concept]
    return groups

# Liste de concepts
concepts = [
    "énergie", "système", "flux", "information", "interaction",
    "transformation", "développement", "complexité", "structure", "réseau",
    "équilibre", "dynamique", "rétroaction", "résilience", "émergence",
    "adaptation", "évolution", "innovation", "durabilité", "interconnexion"
]

# Regroupement des concepts
groupes = group_by_similarity(concepts, threshold=0.3)

# Dictionnaire regroupé (clé: concept principal, valeur: liste des concepts associés)
print("Dictionnaire des concepts groupés :")
print(groupes)

# Liste des groupes de concepts
groupes_liste = list(groupes.values())
print("\nListe des groupes :")
print(groupes_liste)
