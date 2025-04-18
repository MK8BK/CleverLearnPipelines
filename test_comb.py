from sentence_transformers import SentenceTransformer, util
from huggingface_hub import hf_hub_download as cached_download
import torch
import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)

similarities = util.cos_sim(embeddings, embeddings)

#conversion distance en similarité 
distance_matrix = 1 - similarities 

distance_matrix_np = distance_matrix.numpy() 

#mettre des 0 sur la diag 
np.fill_diagonal(distance_matrix_np, 0.0)

print(distance_matrix_np)


# Partie triangulaire supérieure
condensed_distance = scipy.spatial.distance.squareform(distance_matrix)


Z = sch.linkage(condensed_distance, method='average')

# Function to get clusters based on a threshold
def get_clusters(linkage_matrix, sentences, threshold):
    cluster_labels = sch.fcluster(linkage_matrix, t=threshold, criterion='distance')
    
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])
    
    return clusters



treshold = 0.4
clusters= get_clusters(Z, sentences, treshold)
print(clusters)

l=[]
for cluster_id in clusters:
    l.append(clusters[cluster_id][0])
    
print(l)


# Afficher les clusters
#
#
#for cluster_id, cluster_sentences in clusters.items():
#    print(f"\nCluster {cluster_id}:")
#    for sentence in cluster_sentences:
#        print(f"  - {sentence}")
#



