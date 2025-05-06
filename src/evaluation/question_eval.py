from json import loads
import matplotlib.pyplot as plt
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    pipeline
)

MODEL_NAME = "deepset/roberta-base-squad2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

qa_model = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

def evaluate_question_with_model(question, article_paragraphs):
    best_result = {}
    best_index = -1

    for i, para in enumerate(article_paragraphs):
        context = para.strip()
        if context == "":
            continue
        try:
            result = qa_model(
                question=question.strip(),
                context=context
            )
            if (best_result == {}) or (result["score"] > best_result["score"]):
                best_result = result
                best_index = i + 1  # Use 1-based index
        except Exception:
            continue

    if best_result == {}:
        return {"match": False, "score": 0.0, "model_answer": "", "chunk_index": -1}

    return {
        "score": best_result["score"],
        "model_answer": best_result["answer"],
        "chunk_index": best_index
    }

# ----------------------------------------
if __name__ == "__main__":
    with open("../../test_data/quizzes/26.json", "r") as f:
        quizz = loads(f.read())
        print(len(quizz["mcqs"]))

    with open("../../test_data/pipelines/semantic_text_chunker_pipeline/out2.json", "r") as g:
        chunks = loads(g.read())
        print(len(chunks))

    scores = []
    labels = []
    chunk_indices = []

    for idx, mcq in enumerate(quizz["mcqs"]):
        question = mcq["question"]
        result = evaluate_question_with_model(question, chunks)

        scores.append(result["score"])
        labels.append(f"{idx+1}")
        chunk_indices.append(result["chunk_index"])

        print(f"Question {idx+1}: {question}")
        print(f"Score: {result['score']:.2f}")
        print(f"Answer: {result['model_answer']}")
        print(f"Chunk index: {result['chunk_index']}")
        print("------")

    # Plot 1: Score per question (clean)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, scores)
    plt.ylim(0, 1)
    plt.title("Score per Question")
    plt.xlabel("Question")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()

    # Plot 2: Chunk usage frequency
    chunk_usage = Counter(chunk_indices)
    sorted_chunks = sorted(chunk_usage.keys())
    usage_counts = [chunk_usage[i] for i in sorted_chunks]

    plt.figure(figsize=(12, 6))
    plt.bar([f"Chunk {i}" for i in sorted_chunks], usage_counts)
    plt.title("Usage Frequency of Chunks")
    plt.xlabel("Chunk Index")
    plt.ylabel("Number of Questions Answered Using This Chunk")
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Pie chart of chunk usage percentage (relative to total number of chunks)
    total_chunks = len(chunks)
    chunk_usage = Counter(chunk_indices)

    # Créer une liste complète de counts (1-based index)
    usage_counts = [chunk_usage.get(i, 0) for i in range(1, total_chunks + 1)]
    labels_chunks = [f"Chunk {i}" for i in range(1, total_chunks + 1)]

    # Pourcentage par rapport au nombre total de questions
    percentages = [(count / len(quizz["mcqs"])) * 100 for count in usage_counts]

    # Afficher uniquement les chunks utilisés dans le pie chart
    used_labels = [label for label, count in zip(labels_chunks, usage_counts) if count > 0]
    used_percentages = [pct for pct in percentages if pct > 0]

    # Calcul du taux global d'utilisation des chunks
    used_chunks = sum(1 for count in usage_counts if count > 0)
    usage_rate = (used_chunks / total_chunks) * 100
    comment_text = f"{used_chunks} / {total_chunks} chunks used ({usage_rate:.1f}%)"

    # Affichage du pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(used_percentages, labels=used_labels, autopct='%1.1f%%', startangle=140)
    plt.title("Chunk Usage (% of questions per chunk)")

    # Ajouter un commentaire dans le graphique
    plt.text(0, -1.3, comment_text, ha='center', fontsize=12, style='italic', color='gray')

    plt.tight_layout()
    plt.show()

