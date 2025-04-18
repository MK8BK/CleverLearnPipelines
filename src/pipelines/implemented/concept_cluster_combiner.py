from typing import List
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download as cached_download
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
from typing import List, Dict


# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline 

"""
Pipeline 3: combiner tous les C/I en eliminant les redondances
            --> mesure de distance entre les concepts
"""

class ConceptCombiner(Pipeline):
    title = "concept_combiner_pipeline"
    """
        sbert.net/docs/quickstart.html
    """
    """
        Initialize the ConceptClusterCombiner pipeline.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            threshold (float): Similarity threshold for clustering (0-1)
        """
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",threshold=0.4 ,*args, **kwargs):
        super().__init__(ConceptCombiner.title)
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
    
    def _load_model(self):
        """Load the sentence transformer model if not already loaded"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def get_clusters(self, linkage_matrix: np.ndarray, sentences: List[str], threshold: float) -> Dict[int, List[str]]:
        """
        Group sentences into clusters based on the linkage matrix and threshold.
        
        Args:
            linkage_matrix: Hierarchical clustering linkage matrix found using linkage
            sentences: List of input sentences/concepts
            threshold: Distance threshold for clustering
            
        Returns:
            Dictionary mapping cluster IDs to lists of sentences
        """
        cluster_labels = sch.fcluster(linkage_matrix, t=threshold, criterion='distance')
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[i])
        
        return clusters
    
    def get_representatives(self,clusters: dict[int, List(str)]) -> List[str]:
        """
        Extract one representative from each cluster.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of sentences
        
        Returns:
            List of representative sentences (one from each cluster)
        """
        representatives = []
        for cluster_id in clusters:
            representatives.append(clusters[cluster_id][0])
        
        return representatives

    def _process(self, input_data: List[List[str]],model_name,treshold) -> List[str]:
        """
        Process the input data by:
        1. Flattening the list of lists
        2. Computing embeddings for all concepts
        3. Calculating pairwise similarities
        4. Performing hierarchical clustering
        5. Extracting one representative from each cluster
        
        Args:
            input_data: List of lists of concepts/ideas
            
        Returns:
            List of representative concepts (after deduplication)
        """
        flat_concepts =  [l for lst in input_data for l in lst]
        
        # Skip processing if there are no concepts
        if not flat_concepts:
            return []
        
        model = self._load_model()
        embeddings = model.encode(flat_concepts)
        similarities = util.cos_sim(embeddings, embeddings)
        
        #conversion distance en similarité 
        distance_matrix = 1 - similarities 

        #convert to numpy and ensure diag is 0
        distance_matrix_np = distance_matrix.numpy() 
        np.fill_diagonal(distance_matrix_np, 0.0)
        
        # We select only the upper triangular part of the matrix
        # because the distance matrix is symmetric
        condensed_distance = scipy.spatial.distance.squareform(distance_matrix)
        
        Z = sch.linkage(condensed_distance, method='average') #clustering part
        
        
        
        
        

    def _validate(self, input_data, output_data):
        # TODO: implement later
        return True

def main():
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from concept_extractor import ConceptExtractor

    index = WikiTestDataIndex(TEST_DATA_PATH)
    # print(index.data_path)

    cc = ConceptCombiner()
    index.ensure_pipeline_dir(cc.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        ConceptExtractor.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())



    @measure_time
    def p(input_: List[List[str]]):
        return cc.process(input_)
    pipeline_output = p(paragraphs)
    print(cc._validate(paragraphs, pipeline_output))
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(cc.title, pipeline_output, "out1.json")

if __name__=="__main__":
    main()