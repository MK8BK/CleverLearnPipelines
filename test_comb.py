from sentence_transformers import SentenceTransformer, util

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)

similarities = util.cos_sim(embeddings, embeddings)
print(similarities)