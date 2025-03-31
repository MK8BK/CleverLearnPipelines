from sentence_transformers import SentenceTransformer
from corpus.quiz import MultipleChoiceQuestion
import math 

class Metrics:
    def __init__(self)
        self.answer = MultipleChoiceQuestion.answer
        self.distractors = MultipleChoiceQuestion.distractors
        

    def entropy(self):
        """
        This method calculates the entropy of choosing a proposition
        based on its semantic similarity to the correct answer using
        a pseudo-probabilistic approach.

        Input : 
        Output : entropy (float) 

        """
        probabilities = []
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Compute embeddings for both answer and distractors
        answer_emb = model.encode(self.answer)
        distractors_emb = model.encode(self.distractors)

        # Compute cosine similarities of distractors versus the answer
        similarities = model.similarity(answer_emb, distractors_emb)

        scores_sum = similarities.sum()

        for score_i in range(len(similarities)):

        # Compute the the probability from the scores ratio

            probabilities.append(score_i/scores_sum)

        entropy = 0

        for p_i in probabilities:

            entropy += -(p_i*math.log10(p_i))

        print(f"The MCQ's entropy is :   {entropy}")
            

        




