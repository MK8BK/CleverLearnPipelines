from sentence_transformers import SentenceTransformer, util
from quiz import MultipleChoiceQuestion
from math import *

class Metrics:
    def __init__(self, mcq: MultipleChoiceQuestion):
     
        self.answer = mcq.answer
        self.distractors = mcq.distractors
        

    def entropy(self) -> float:
        """
        calculates the entropy of choosing a proposition based
        on its semantic similarity to the correct answer using
        a pseudo-probabilistic approach.

        Output : entropy (float) 

        """
        probabilities = []
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Compute embeddings for both answer and distractors
        answer_emb = model.encode(self.answer)
        distractors_emb = model.encode(self.distractors)

        # Compute cosine similarities of distractors versus the answer
        similarities = util.pytorch_cos_sim(answer_emb, distractors_emb)        # returns Tensor([[float, float, ... , 1]])

        print(similarities)

        similarities = similarities.tolist()
        similarities = similarities[0]
        
        scores_sum = sum(similarities)

        for i in range(len(similarities)):
            # Compute the the probability of choosing a distractor
            p_i = similarities[i]/scores_sum
            probabilities.append(p_i)

        entropy = 0

        for p_i in probabilities:
            entropy += -(p_i*log2(max(p_i, 1e-10)))            # the max function is used to avoid log(0)

        return probabilities, entropy


# test

easy_mcq = {'mcq_1':{'question': "Which of the following statements about Lithuania is true?",
                'answer': "The capital of Lithuania is Vilnius.",
                'distractors': ["Lithuania is located in Scandinavia.",
                                "The official language of Lithuania is Latvian.",
                                "Lithuania became independent from the Soviet Union in 2004.",
                                "Lithuania is an island nation in the Baltic Sea.",
                                "The capital of Lithuania is Vilnius."]},

            'mcq_2':{'question': "Which of the following statements about Albert Einstein is true?",
                'answer': "Albert Einstein developed the theory of relativity.",
                'distractors': ["Albert Einstein was born in the United States.",
                                "Albert Einstein wrote the 'Harry Potter' series.",
                                "Albert Einstein was the first man on the moon.",
                                "Albert Einstein developed the theory of relativity."]},

            'mcq_3':{'question': "Which of the following is the capital of Japan?",
                'answer': "Tokyo is the capital of Japan.",
                'distractors': ["Beijing is the capital of Japan.",
                                "Seoul is the capital of Japan.",
                                "Bangkok is the capital of Japan.",
                                "Tokyo is the capital of Japan."]},

            'mcq_4':{'question': "Which of the following statements about the Eiffel Tower is true?",
                'answer': "The Eiffel Tower is located in Paris, France.",
                'distractors': ["The Eiffel Tower is located in London, England.",
                                "The Eiffel Tower was designed by Leonardo da Vinci.",
                                "The Eiffel Tower was built for the 2000 Olympics.",
                                "The Eiffel Tower is located in Paris, France."]}
            }

hard_mcq = {'mcq_1':{'question': "Which of the following is the primary function of the human liver?",
                'answer': "The liver filters toxins from the blood.",
                'distractors': ["The liver stores bile for digestion.",
                                "The liver produces insulin to regulate blood sugar.",
                                "The liver stores glycogen and converts it to glucose.",
                                "The liver breaks down red blood cells to produce hemoglobin.",
                                "The liver filters toxins from the blood."]},

            'mcq_2':{'question': "Which of the following is the capital of Switzerland?",
                'answer': "Bern is the capital of Switzerland.",
                'distractors': ["Zurich is the capital of Switzerland.",
                                "Geneva is the capital of Switzerland.",
                                "Basel is the capital of Switzerland.",
                                "Bern is the capital of Switzerland."]},

            'mcq_3':{'question': "Which of the following is a type of bird?",
                'answer': "Penguins are a type of bird.",
                'distractors': ["Puffins are a type of bird.",
                                "Albatrosses are a type of bird.",
                                "Seagulls are a type of bird.",
                                "Penguins are a type of bird."]},
            
            'mcq_4':{'question': "Which of the following composers is known for composing symphonies?",
                'answer': "Ludwig van Beethoven is known for composing symphonies.",
                'distractors': ["Johannes Brahms is known for composing symphonies.",
                                "Wolfgang Amadeus Mozart is known for composing symphonies.",
                                "Franz Schubert is known for composing symphonies.",
                                "Ludwig van Beethoven is known for composing symphonies."]},
            }

print("""printing the easy mcq's entropies: 
       """)

for i in range(1,5):
    mcq = MultipleChoiceQuestion(**easy_mcq[f"mcq_{i}"])
    entropy = Metrics(mcq).entropy()
    print(entropy[0])
    print("")
    print(f"{entropy[1]}               {log2(5)}")

print("""printing the hard mcq's entropies: 
       """)

for i in range(1,5):
    mcq = MultipleChoiceQuestion(**hard_mcq[f"mcq_{i}"])
    entropy = Metrics(mcq).entropy()
    print(entropy[0])
    print("")
    print(f"{entropy[1]}               {log2(5)}")

print("Done.")


