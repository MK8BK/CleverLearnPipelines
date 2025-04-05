# CleverLearnPipelines

Pipelines NLP et RAG pour extraire des compétences de textes pédagogiques et générer automatiquement des questions et QCM.

# Installation
Be sure to use python `>=3.11` and `<=3.12` 
(spacy does not yet support `3.13`)

Create a python virtual environment

Activate it


```bash
pip install sentence-transformers

# possibly useless now
pip install langchain-text-splitters
pip install langchain_experimental langchain_openai
```

install the requirements specified in the 
`requirements.txt` file

create a file named `.env` in `src/prompters`
OPENAI_API_KEY=\<our-top-secret-api-key>

## Docs
[OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)

[langchain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)


[Spacy 101](https://spacy.io/usage/spacy-101/)


