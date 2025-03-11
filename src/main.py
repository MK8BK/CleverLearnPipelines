from prompters.gpt_4o_mini import Gpt_4o_mini_client, OpenAI_role
from prompters.prompt_engineer import PromptEngineer
from corpus.corpus import Corpus
from corpus.quiz import Quiz

client = Gpt_4o_mini_client()

def make_quiz(article_name):
    corpus = Corpus(f"scraping/data/wiki/{article_name}.md")
    prompt_engineer = PromptEngineer(client, corpus)
    prompt_engineer.build_quiz()
    prompt_engineer.save_quiz(f"corpus/generated/quiz/{article_name}.json")

# make_quiz("Microeconomics")
# make_quiz("Paraguay")
make_quiz("Napoleon")