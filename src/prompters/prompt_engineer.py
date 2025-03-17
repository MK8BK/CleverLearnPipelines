from .gpt_4o_mini import Gpt_4o_mini_client, OpenAI_role
from corpus.quiz import Quiz
from corpus.corpus import Corpus
from json import dumps




class PromptEngineer:
    def __init__(self, llmclient: Gpt_4o_mini_client, corpus: Corpus):
        self.llmclient = llmclient
        self.corpus = corpus
        self.min_number_mcq = 10
    def build_dev_prompt_question(self):
        prompt = f"""
        The user is a student and I want you to generate a multiple-choice quiz based strictly on the following Material. The quiz must:
        - Contain exactly {self.min_number_mcq} unique questions, ensuring no repetition.
        - Cover all key points of the text (except the references ..) comprehensively, without omitting any important information.
        - Stick exclusively to the information provided, without adding external knowledge or interpretations.
        - Offer four distinct answer choices per question, with only one correct answer.
        - Ensure that both questions and answers are clearly formulated, precise, and unambiguous.
        """
        return prompt
    def build_quiz(self):
        dev_prompt = self.build_dev_prompt_question()
        self.llmclient.add_message(OpenAI_role.DEVELOPER, dev_prompt)
        self.llmclient.add_message(OpenAI_role.USER, self.corpus.clean_text)
        self.chat_completion = self.llmclient.submit_messages(response_format=Quiz)
        self.quiz = self.chat_completion.choices[0].message.parsed
        self.dict_quiz = self.quiz.model_dump()
        return self.quiz
    def save_quiz(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(dumps(self.dict_quiz))
    #




        
    