from sentence_transformers import SentenceTransformer, util
import json
import matplotlib.pyplot as plt


# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def mean_similarity(mcq_path: str, treshold=0.5):
    """Plots each MCQ's average cosine similarity score of distractors
    versus the correct answer, and gives the percentage of MCQs that
    have an average score higher than the threshold.

    Args:
        mcq_path (str): Path to the MCQ's JSON file.
        treshold (float, optional): Average score threshold. Defaults to 0.5.
    """

    with open(mcq_path, 'r') as file:  
        mcqs = json.load(file)

    avg_similarities = []
    similar_mcqs = 0
    n = len(mcqs["mcqs"])

    for i in range(n):
        answer = mcqs["mcqs"][i]["answer"]
        distractors = mcqs["mcqs"][i]["distractors"]

        
        answer_emb = model.encode(answer)
        distractors_emb = model.encode(distractors)

        # Compute cosine similarities of distractors versus the answer
        similarities = util.pytorch_cos_sim(answer_emb, distractors_emb)  
        similarities = similarities.tolist()
        similarities = similarities[0]

        avg_sim = sum(similarities) / len(distractors)  # Averaging similarity scores
        if avg_sim >= treshold:
            similar_mcqs += 1
        avg_similarities.append(avg_sim)

    print(f"The ratio of MCQs that have an average similarity score over {treshold} is: {round(similar_mcqs / n, 2)}")

    plt.bar([i + 1 for i in range(len(avg_similarities))], avg_similarities)
    plt.axhline(y=treshold, color='r', linestyle='--', label=f'y = {treshold}')
    plt.title(f"{round(similar_mcqs * 100 / n, 2)}% ({similar_mcqs}) of MCQs are over {treshold}")
    plt.xlabel("MCQ Index")
    plt.ylabel("Average Similarity")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    treshold = 0.6
    test_path = "../../test_data/quizzes/1.json"  # Relative path to the JSON file
    mean_similarity(test_path, treshold)