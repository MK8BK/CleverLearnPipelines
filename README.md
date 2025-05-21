# CleverLearnPipelines

Pipelines NLP et RAG pour extraire des compétences de textes pédagogiques et générer automatiquement des questions et QCM.


# running examples on 
https://en.wikipedia.org/wiki/Mainland_Australia

# Installation
Be sure to use python `>=3.11`

Create a python virtual environment

Activate it

```bash 
pip install -r requirements.txt
pip install sentence-transformers # os specific
```

create a file named `.env` in `src/llms`
OPENAI_API_KEY=\<our-top-secret-api-key>

If libraries are missing when using a different non default pipeline set:

```bash
# possibly useless now
pip install nltk
pip install langchain_experimental langchain_openai
```
```python
>>> import nltk
>>> nltk.download('punkt_tab')
```

# Usage

```bash
# in the src/ directory
python -u main.py -g <https-link-to-wikipedia-article>
```

## Docs
Useful 
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [langchain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)



