# CleverLearnPipelines
Pipelines NLP et RAG pour extraire des compétences de textes pédagogiques et générer automatiquement des questions et QCM.

Voici une clé API confidentielle à usage strictement réservé à vous et au projet :

Cette clé vous permettra de réaliser les calls API avec les différents modèles openAI. La documentation de l'API est disponible à cette adresse : https://openai.com/api/




Necessite python>=3.11(type hints pas dispo en dessous)
<=3.12(spacy pas possible sous 3.13)
 

[Spacy 101](https://spacy.io/usage/spacy-101/)

https://spacy.io/usage

specify virtualenv, cpu, english && french, accuracy

✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_trf')
✔ Download and installation successful
You can now load the package via spacy.load('fr_dep_news_trf')




## Notes
precisez le nombre de MCQ a generer
subdiviser le corpus


in .env in src/models
OPENAI_API_KEY=<la-cle-api-envoyee-par-mail>