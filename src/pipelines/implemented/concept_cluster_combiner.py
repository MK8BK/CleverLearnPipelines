from typing import List
from sentence_transformers import SentenceTransformer, util
import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
from typing import List, Dict


# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------
from pipelines.base_pipeline import Pipeline 


# TODO:
#   - jouer avec threshold
#   - retenir get_representatives pour ranker les clusters/concepts


"""
Pipeline 3: combiner tous les C/I en eliminant les redondances
            --> mesure de distance entre les concepts
"""

class ConceptClusterCombiner(Pipeline):
    title = "concept_cluster_combiner_pipeline"
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
        super().__init__(ConceptClusterCombiner.title)
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
    
    def get_representatives(self,clusters: List[List[str]]) -> List[str]:
        """
        Extract one representative from each cluster.
        
        Args:
            clusters: Dictionary mapping cluster IDs to lists of sentences
        
        Returns:
            List of representative sentences (one from each cluster)
        """
        representatives = [cluster[0] for cluster in clusters]
        return representatives

    def _process(self, input_data: List[List[str]]) -> List[str]:
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
        assert len(distance_matrix_np.shape)==2 and\
            distance_matrix_np.shape[0]==distance_matrix_np.shape[1], "Not symmetric"
        condensed_distance = scipy.spatial.distance.squareform(distance_matrix_np, checks=False)
        # for checks cf: https://stackoverflow.com/a/73428402
        
        Z = sch.linkage(condensed_distance, method='average') #clustering part
        
        
        clusters = self.get_clusters(Z, flat_concepts, self.threshold) #get clusters
        clusters = self.filter_clusters(clusters)
        
        representatives = self.get_representatives(clusters) #get representatives
        
        return representatives
    
    def filter_clusters(self,clusters: Dict[int, List[str]]):
        n = self.context["mcq_number"]
        cluster_couples = sorted([v for v in clusters.values()], key=lambda x: len(x),
                                 reverse=True)
        if len(cluster_couples)>n:
            return cluster_couples[:n]
        return cluster_couples
    
    
    def _validate(self, input_data, output_data):
        
        # Check that output is a list of strings
        if not isinstance(output_data, list):
            return False
        
        # Check that all elements are strings
        if not all(isinstance(item, str) for item in output_data):
            return False
        
        # Check that output has fewer or equal elements than flattened input
        flat_input = [item for sublist in input_data for item in sublist]
        if len(output_data) > len(flat_input):
            return False
        
        len_input = sum(len(lst) for lst in input_data)
        self.logger.info(f"{len(output_data)}/{len_input}")
            
        return True
            

def main():
    from index import WikiTestDataIndex
    from test_helpers import TEST_DATA_PATH, measure_time
    from concept_extractor import ConceptExtractor

    index = WikiTestDataIndex(TEST_DATA_PATH)

    ccc = ConceptClusterCombiner(threshold=0.4)
    index.ensure_pipeline_dir(ccc.title)

    from json import dumps, loads

    input_file_path = TEST_DATA_PATH.joinpath("pipelines").joinpath(
        ConceptExtractor.title).joinpath("out1.json")

    with open(str(input_file_path), "r", encoding="utf8") as f:
        paragraphs = loads(f.read())
    @measure_time
    def p(input_: List[List[str]]):
        return ccc.process(input_)
    
    
    print(len(paragraphs), "----------------")
    pipeline_output = p(paragraphs)
    
    print(f"Number of input concepts: {sum(len(lst) for lst in paragraphs)}")
    print(f"Number of output concepts: {len(pipeline_output)}")
    print(f"Validation result: {ccc._validate(paragraphs, pipeline_output)}")
    
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(ccc.title, pipeline_output, "out1.json")

if __name__=="__main__":
    main()