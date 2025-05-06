import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import os
import time
import random

# Ajouter les chemins pour l'importation des modules du projet
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from src.index import WikiTestDataIndex
from src.corpus.corpus import Corpus
from src.pipelines.implemented.text_chunker import TextChunker
from src.pipelines.implemented.semantic_text_chunker import SemanticTextChunker
from src.pipelines.implemented.concept_extractor import ConceptExtractor
from src.pipelines.implemented.concept_combiner import ConceptCombiner

# Importer le ConceptClusterCombiner si implémenté
try:
    from src.pipelines.implemented.concept_cluster_combiner import ConceptClusterCombiner
    concept_cluster_combiner_imported = True
except ImportError:
    concept_cluster_combiner_imported = False

# Pour la visualisation du clustering
from sentence_transformers import SentenceTransformer, util
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance

# Chemin vers les données de test
TEST_DATA_PATH = Path("test_data")

# Classe de base pour les visualisateurs de pipeline
class PipelineVisualizer:
    """Classe de base pour tous les visualisateurs de pipeline"""
    
    def __init__(self, index):
        self.index = index
    
    def display(self):
        """Méthode à implémenter dans les classes enfants"""
        raise NotImplementedError


# Fonction pour mesurer le temps d'exécution
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper


class TextChunkingVisualizer(PipelineVisualizer):
    """Visualiseur pour l'étape de chunking de texte"""
    
    def display(self):
        st.header("Extraction de texte")
        
        st.subheader("Fichiers disponibles")
        wiki_files = list(TEST_DATA_PATH.joinpath("wiki").glob("*.md"))
        selected_file = st.selectbox(
            "Sélectionner un fichier",
            options=wiki_files,
            format_func=lambda x: x.name
        )
        
        if selected_file:
            with open(str(selected_file), "r", encoding="utf8") as f:
                corpus_text = f.read()
                corpus = Corpus(corpus_text)
            
            with st.expander("Afficher le texte brut"):
                st.text_area("Texte brut", corpus_text, height=200)
            
            with st.expander("Afficher le texte nettoyé"):
                st.text_area("Texte nettoyé", corpus.clean_text, height=200)
            
            chunking_method = st.radio(
                "Méthode de chunking",
                ["Chunking simple", "Chunking sémantique"]
            )
            
            if st.button("Lancer l'extraction"):
                with st.spinner("Chunking en cours..."):
                    if chunking_method == "Chunking simple":
                        chunker = TextChunker()
                        chunks, execution_time = measure_execution_time(chunker.process)(corpus.clean_text)
                    else:
                        chunker = SemanticTextChunker()
                        chunks, execution_time = measure_execution_time(chunker.process)(corpus.clean_text)
                
                self.display_chunks_results(chunks, execution_time)
    
    def display_chunks_results(self, chunks, execution_time):
        st.success(f"Extraction terminée en {execution_time:.2f} secondes")
        st.write(f"Nombre de chunks: {len(chunks)}")
        
        chunks_df = pd.DataFrame({
            "Chunk ID": range(1, len(chunks) + 1),
            "Texte": chunks,
            "Longueur": [len(chunk) for chunk in chunks]
        })
        
        st.dataframe(chunks_df)
        
        fig, ax = plt.subplots()
        ax.hist([len(chunk) for chunk in chunks], bins=20)
        ax.set_xlabel("Longueur du chunk (caractères)")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution de la longueur des chunks")
        st.pyplot(fig)


class ConceptExtractionVisualizer(PipelineVisualizer):
    """Visualiseur pour l'étape d'extraction de concepts"""
    
    def display(self):
        st.header("Extraction de concepts")
        
        st.warning("Cette étape utilise l'API OpenAI et nécessite une clé API valide dans le fichier `.env`")
        
        chunking_output_files = list(TEST_DATA_PATH.joinpath("pipelines").joinpath("text_chunker_pipeline").glob("*.json"))
        semantic_chunking_output_files = list(TEST_DATA_PATH.joinpath("pipelines").joinpath("semantic_text_chunker_pipeline").glob("*.json"))
        
        all_chunking_files = chunking_output_files + semantic_chunking_output_files
        
        if not all_chunking_files:
            st.error("Aucun fichier de chunking trouvé. Veuillez d'abord lancer l'extraction de texte.")
        else:
            selected_chunk_file = st.selectbox(
                "Sélectionner un fichier de chunks",
                options=all_chunking_files,
                format_func=lambda x: f"{x.parent.name}/{x.name}"
            )
            
            if selected_chunk_file:
                with open(str(selected_chunk_file), "r", encoding="utf8") as f:
                    chunks = json.loads(f.read())
                
                st.write(f"Nombre de chunks: {len(chunks)}")
                
                with st.expander("Visualiser les chunks"):
                    for i, chunk in enumerate(chunks):
                        st.text_area(f"Chunk {i+1}", chunk, height=100)
                
                if st.button("Lancer l'extraction de concepts"):
                    try:
                        with st.spinner("Extraction des concepts en cours..."):
                            ce = ConceptExtractor()
                            concepts_output, execution_time = measure_execution_time(ce.process)(chunks)
                        
                        self.display_concepts_results(chunks, concepts_output, execution_time)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'extraction des concepts: {e}")
    
    def display_concepts_results(self, chunks, concepts_output, execution_time):
        st.success(f"Extraction terminée en {execution_time:.2f} secondes")
        
        # Afficher les concepts
        all_concepts = [concept for sublist in concepts_output for concept in sublist]
        st.write(f"Nombre total de concepts: {len(all_concepts)}")
        
        # Créer un dataframe pour visualiser les concepts par chunk
        concepts_data = []
        for i, (chunk, concepts) in enumerate(zip(chunks, concepts_output)):
            for concept in concepts:
                concepts_data.append({
                    "Chunk ID": i+1,
                    "Concept": concept,
                    "Longueur du chunk": len(chunk)
                })
        
        concepts_df = pd.DataFrame(concepts_data)
        st.dataframe(concepts_df)
        
        # Visualiser la distribution des concepts par chunk
        fig, ax = plt.subplots()
        concept_counts = [len(concept_list) for concept_list in concepts_output]
        ax.hist(concept_counts, bins=range(min(concept_counts), max(concept_counts) + 2))
        ax.set_xlabel("Nombre de concepts par chunk")
        ax.set_ylabel("Fréquence")
        ax.set_title("Distribution du nombre de concepts par chunk")
        st.pyplot(fig)


# Fonction pour visualiser le dendrogramme
def plot_dendrogram(linkage_matrix, labels, threshold=0.4):
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram = sch.dendrogram(
        linkage_matrix,
        labels=labels,
        orientation='top',
        leaf_font_size=10,
        distance_sort='descending',
        ax=ax
    )
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Seuil = {threshold}')
    plt.title('Dendrogramme du clustering hiérarchique')
    plt.xlabel('Concepts')
    plt.ylabel('Distance')
    plt.legend()
    plt.tight_layout()
    return fig


class ClusterProcessor:
    """Classe pour effectuer le clustering des concepts"""
    
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
    @measure_execution_time
    def perform_clustering(self, concepts):
        # Aplatir la liste de listes
        flat_concepts = [concept for sublist in concepts for concept in sublist]
        
        if not flat_concepts:
            return {"message": "Aucun concept à traiter"}, {}
        
        # Charger le modèle pour les embeddings
        model = SentenceTransformer(self.model_name)
        
        # Calculer les embeddings
        embeddings = model.encode(flat_concepts)
        
        # Calculer la matrice de similarité
        similarities = util.cos_sim(embeddings, embeddings)
        
        # Convertir en matrice de distance
        distance_matrix = 1 - similarities
        
        # Assurer que la diagonale est à zéro
        distance_matrix_np = distance_matrix.numpy()
        np.fill_diagonal(distance_matrix_np, 0.0)
        
        # Extraire la partie triangulaire supérieure
        condensed_distance = scipy.spatial.distance.squareform(distance_matrix_np)
        
        # Effectuer le clustering hiérarchique
        Z = sch.linkage(condensed_distance, method='average')
        
        # Obtenir les clusters en fonction du seuil
        cluster_labels = sch.fcluster(Z, t=self.threshold, criterion='distance')
        
        # Organiser les résultats
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(flat_concepts[i])
        
        # Extraire un représentant par cluster (le premier élément)
        representatives = [cluster[0] for cluster in clusters.values()]
        
        result = {
            "original_concepts": flat_concepts,
            "clusters": clusters,
            "representatives": representatives,
            "linkage_matrix": Z
        }
        
        return result, flat_concepts


class ConceptClusteringVisualizer(PipelineVisualizer):
    """Visualiseur pour l'étape de clustering de concepts"""
    
    def display(self):
        st.header("Clustering de concepts")
        
        # Vérifier si les fichiers de concepts existent
        concept_files = list(TEST_DATA_PATH.joinpath("pipelines").joinpath("concept_extractor_pipeline").glob("*.json"))
        
        if not concept_files:
            st.error("Aucun fichier de concepts trouvé. Veuillez d'abord lancer l'extraction de concepts.")
        else:
            selected_concept_file = st.selectbox(
                "Sélectionner un fichier de concepts",
                options=concept_files,
                format_func=lambda x: x.name
            )
            
            if selected_concept_file:
                with open(str(selected_concept_file), "r", encoding="utf8") as f:
                    concepts = json.loads(f.read())
                
                flat_concepts = [concept for sublist in concepts for concept in sublist]
                st.write(f"Nombre total de concepts: {len(flat_concepts)}")
                
                with st.expander("Visualiser les concepts bruts"):
                    for i, concept_list in enumerate(concepts):
                        st.write(f"Chunk {i+1}: {len(concept_list)} concepts")
                        st.write(", ".join(concept_list))
                        st.write("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    clustering_method = st.radio(
                        "Méthode de clustering",
                        ["Simple (concat)", "Clustering hiérarchique"]
                    )
                
                with col2:
                    if clustering_method == "Clustering hiérarchique":
                        threshold = st.slider(
                            "Seuil de similarité",
                            min_value=0.1,
                            max_value=0.9,
                            value=0.4,
                            step=0.05,
                            help="Plus le seuil est bas, plus les clusters seront nombreux"
                        )
                
                if st.button("Lancer le clustering"):
                    with st.spinner("Clustering en cours..."):
                        if clustering_method == "Simple (concat)":
                            self.run_simple_clustering(concepts, flat_concepts)
                        else:  # Clustering hiérarchique
                            if concept_cluster_combiner_imported:
                                self.run_hierarchical_clustering_with_implementation(concepts, flat_concepts, threshold)
                            else:
                                self.run_hierarchical_clustering_with_backup(concepts, threshold)
    
    def run_simple_clustering(self, concepts, flat_concepts):
        cc = ConceptCombiner()
        combined_concepts, execution_time = measure_execution_time(cc.process)(concepts)
        
        st.success(f"Clustering terminé en {execution_time:.2f} secondes")
        st.write(f"Nombre de concepts après combinaison: {len(combined_concepts)}")
        
        st.subheader("Concepts combinés")
        concepts_df = pd.DataFrame({"Concept": combined_concepts})
        st.dataframe(concepts_df)
    
    def run_hierarchical_clustering_with_implementation(self, concepts, flat_concepts, threshold):
        ccc = ConceptClusterCombiner(threshold=threshold)
        clustered_concepts, execution_time = measure_execution_time(ccc.process)(concepts)
        
        st.success(f"Clustering terminé en {execution_time:.2f} secondes")
        st.write(f"Nombre de concepts après clustering: {len(clustered_concepts)}")
        
        st.subheader("Concepts représentatifs")
        concepts_df = pd.DataFrame({"Concept": clustered_concepts})
        st.dataframe(concepts_df)
        
        st.write(f"Réduction: {len(flat_concepts)} → {len(clustered_concepts)} concepts ({(1 - len(clustered_concepts)/len(flat_concepts))*100:.1f}%)")
        
        # Visualiser à nouveau pour le dendrogramme
        processor = ClusterProcessor(threshold=threshold)
        results, flat_concepts_list = processor.perform_clustering(concepts)
        
        # Afficher le dendrogramme
        if "linkage_matrix" in results and len(flat_concepts_list) <= 100:
            st.subheader("Dendrogramme")
            fig = plot_dendrogram(
                results["linkage_matrix"], 
                [f"{i+1}: {c[:30]}..." if len(c) > 30 else f"{i+1}: {c}" for i, c in enumerate(flat_concepts_list)],
                threshold
            )
            st.pyplot(fig)
        elif len(flat_concepts_list) > 100:
            st.warning("Trop de concepts pour afficher le dendrogramme (>100)")
    
    def run_hierarchical_clustering_with_backup(self, concepts, threshold):
        processor = ClusterProcessor(threshold=threshold)
        results, execution_time = processor.perform_clustering(concepts)
        
        self.display_clustering_results(results, execution_time)
    
    def display_clustering_results(self, results, execution_time):
        st.success(f"Clustering terminé en {execution_time:.2f} secondes")
        st.write(f"Nombre de concepts originaux: {len(results['original_concepts'])}")
        st.write(f"Nombre de clusters: {len(results['clusters'])}")
        st.write(f"Nombre de représentants: {len(results['representatives'])}")
        
        # Afficher les clusters
        st.subheader("Clusters de concepts")
        for cluster_id, cluster in results["clusters"].items():
            with st.expander(f"Cluster {cluster_id} ({len(cluster)} éléments)"):
                st.write(", ".join(cluster))
        
        # Afficher les représentants
        st.subheader("Concepts représentatifs")
        reps_df = pd.DataFrame({"Concept": results["representatives"]})
        st.dataframe(reps_df)
        
        # Afficher le dendrogramme si pas trop de concepts
        if len(results["original_concepts"]) <= 100:
            st.subheader("Dendrogramme")
            fig = plot_dendrogram(
                results["linkage_matrix"], 
                [f"{i+1}: {c[:30]}..." if len(c) > 30 else f"{i+1}: {c}" for i, c in enumerate(results["original_concepts"])],
                processor.threshold
            )
            st.pyplot(fig)
        else:
            st.warning("Trop de concepts pour afficher le dendrogramme (>100)")


class QuizVisualizer(PipelineVisualizer):
    """Visualiseur pour les QCM générés"""
    
    def display(self):
        st.header("Visualisation des QCM générés")
        
        # Rechercher les fichiers de QCM disponibles
        quiz_files = list(TEST_DATA_PATH.joinpath("quizzes").glob("*.json"))
        
        if not quiz_files:
            st.error("Aucun fichier de QCM trouvé dans le dossier test_data/quizzes/")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_quiz_file = st.selectbox(
                    "Sélectionner un fichier de QCM",
                    options=quiz_files,
                    format_func=lambda x: f"Quiz {x.stem}"
                )
            
            with col2:
                num_questions = st.slider(
                    "Nombre de questions à afficher",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Sélectionnez le nombre de questions à afficher"
                )
                
                show_mode = st.radio(
                    "Mode d'affichage",
                    ["Aléatoire", "Séquentiel", "Toutes les questions"],
                    index=0
                )
            
            if selected_quiz_file:
                # Charger le QCM
                with open(str(selected_quiz_file), "r", encoding="utf8") as f:
                    quiz_data = json.load(f)
                
                # Obtenir les MCQs
                all_mcqs = quiz_data.get("mcqs", [])
                total_questions = len(all_mcqs)
                
                st.write(f"QCM chargé avec succès: {total_questions} questions disponibles")
                
                # Sélectionner les questions à afficher
                selected_mcqs = self.select_mcqs(all_mcqs, num_questions, total_questions, show_mode)
                
                # Options d'affichage
                display_options = st.radio(
                    "Options d'affichage",
                    ["Vue de QCM", "Vue de tableau"]
                )
                
                if display_options == "Vue de QCM":
                    self.display_qcm_view(selected_mcqs)
                else:  # Vue de tableau
                    self.display_table_view(selected_mcqs)
                
                # Option d'exportation
                if st.button("Exporter en format JSON"):
                    export_json = json.dumps({"mcqs": selected_mcqs}, indent=2)
                    st.download_button(
                        label="Télécharger le JSON",
                        data=export_json,
                        file_name=f"quiz_export_{selected_quiz_file.stem}.json",
                        mime="application/json"
                    )
    
    def select_mcqs(self, all_mcqs, num_questions, total_questions, show_mode):
        if show_mode == "Aléatoire":
            if num_questions > total_questions:
                num_questions = total_questions
            return random.sample(all_mcqs, num_questions)
        elif show_mode == "Séquentiel":
            return all_mcqs[:num_questions]
        else:  # Toutes les questions
            return all_mcqs
    
    def display_qcm_view(self, selected_mcqs):
        # Affichage des QCM en lecture seule (sans interaction)
        st.subheader("Questions à choix multiples")
        
        for i, mcq in enumerate(selected_mcqs):
            question = mcq["question"]
            answer = mcq["answer"]
            distractors = mcq["distractors"]
            
            # Afficher la question avec un style
            st.markdown(f"""
            <div style="background-color:hsl(50, 33%, 25%);padding:10px;border-radius:5px;margin-bottom:10px">
                <h4>Question {i+1}:</h4>
                <p>{question}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher la réponse correcte
            st.markdown(f"""
            <div style="background-color:rgba(255, 255, 128, 0.5);padding:10px;border-radius:5px;margin-bottom:5px">
                <p><strong>Réponse correcte:</strong> {answer}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les distracteurs
            st.markdown("<p><strong>Distracteurs:</strong></p>", unsafe_allow_html=True)
            for j, distractor in enumerate(distractors):
                st.markdown(f"""
                <div style="background-color:brown;padding:5px;border-radius:5px;margin-bottom:5px">
                    <p>{distractor}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
    
    def display_table_view(self, selected_mcqs):
        quiz_data_for_df = []
        for i, mcq in enumerate(selected_mcqs):
            quiz_data_for_df.append({
                "N°": i+1,
                "Question": mcq["question"],
                "Réponse correcte": mcq["answer"],
                "Distracteurs": ", ".join(mcq["distractors"])
            })
        
        quiz_df = pd.DataFrame(quiz_data_for_df)
        st.dataframe(quiz_df, use_container_width=True)


class HomeVisualizer(PipelineVisualizer):
    """Visualiseur pour la page d'accueil"""
    
    def display(self):
        st.markdown("""
        ## CleverLearn Pipelines
        
        CleverLearn Pipelines est un système de traitement NLP et RAG pour:
        - Extraire des compétences de textes pédagogiques
        - Générer automatiquement des questions et QCM
        
        ### Étapes du pipeline
        1. **Chunking de texte** - Découpage du texte en paragraphes sémantiques
        2. **Extraction de concepts** - Identifier les concepts importants dans chaque paragraphe
        3. **Clustering de concepts** - Regrouper les concepts similaires pour éliminer les redondances
        4. **Génération de questions** - Créer des questions et réponses basées sur les concepts
        5. **Génération de QCM** - Ajouter des distracteurs pour créer des QCM complets
        
        Utilisez les options dans la barre latérale pour explorer les différentes étapes.
        """)
        
        # Afficher la structure du workspace
        with st.expander("Structure du workspace"):
            st.code("""
            CleverLearnPipelines/
            ├── README.md
            ├── requirements.txt
            ├── app.py (ce fichier)
            ├── src/
            │   ├── index.py
            │   ├── corpus/
            │   ├── llms/
            │   ├── pipelines/
            │   └── scraping/
            └── test_data/
                ├── wiki/
                └── pipelines/
                └── quizzes/
            """)


def main():
    st.set_page_config(
        page_title="CleverLearn Pipelines",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("CleverLearn Pipelines")
    st.markdown("""
    Cette application permet de lancer et visualiser les différentes étapes 
    du pipeline de traitement pour l'extraction et le clustering de concepts.
    """)
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choisir une page",
        ["Accueil", "Extraction de texte", "Extraction de concepts", "Clustering de concepts", "Visualisation des QCM"]
    )
    
    # Initialiser l'index
    index = WikiTestDataIndex(TEST_DATA_PATH)
    
    # Créer le visualiseur approprié en fonction de la page sélectionnée
    if page == "Accueil":
        visualizer = HomeVisualizer(index)
    elif page == "Extraction de texte":
        visualizer = TextChunkingVisualizer(index)
    elif page == "Extraction de concepts":
        visualizer = ConceptExtractionVisualizer(index)
    elif page == "Clustering de concepts":
        visualizer = ConceptClusteringVisualizer(index)
    elif page == "Visualisation des QCM":
        visualizer = QuizVisualizer(index)
    
    # Afficher le contenu de la page
    visualizer.display()


if __name__ == "__main__":
    main()