from models.gpt_4o_mini import Gpt_4o_mini_client, OpenAI_role

from corpus.corpus import Corpus
from corpus.quiz import Quiz

text = Corpus("scraping/data/wiki/Paraguay.md")
client = Gpt_4o_mini_client()

developer_instructions = "the user is a student and I want you to generate a quizz \
    consisting of multiple choice questions to test his understanding of the\
    following material."

client.add_message(OpenAI_role.DEVELOPER, developer_instructions)
client.add_message(OpenAI_role.USER, text.text)
chat_response = client.submit_messages(response_format=Quiz)
quiz = chat_response.choices[0].message.parsed.model_dump()

from json import dumps
quiz_str = dumps(quiz)
with open("corpus/generated/quiz/Paraguay.json", "w") as f:
    f.write(quiz_str)






#question = "What color is the sky?"
#client = Gpt_4o_mini_client()
#client.add_message(OpenAI_role.USER, question)
#print(client.submit_messages())