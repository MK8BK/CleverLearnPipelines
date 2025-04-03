import re
from typing import List
from sentence_transformers import SentenceTransformer, util


def split_into_sentences(text: str) -> List[str]:
    
    def markdown_to_text(markdown_list):
        clean_list = []
    # Regex pour matcher le format [texte](lien)
        pattern = re.compile(r'\[([^\]]+)\]\([^)]+\)')
        for item in markdown_list:
            # Remplace les [texte](lien) par juste "texte"
            clean_text = pattern.sub(r'\1', item)
            clean_list.append(clean_text)
        return clean_list
    
    # Regex : capture les fins de phrases classiques
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZÉÈÂÎÔÙ])')
    sentences = sentence_endings.split(text.strip())
    return markdown_to_text([s.strip() for s in sentences if s.strip()])
def chunk_text_intelligently_with_bert(
    sentences: List[str],
    model,
    similarity_threshold: float = 0.43,
    max_chunk_size: int = 3000
) -> List[str]:
    """
    Regroupe les phrases en chunks selon :
      - un seuil de similarité  (cosine similarity)
      - une limite de taille max en caractères

    :param sentences: Liste de phrases déjà découpées
    :param model: Modèle SentenceTransformer chargé
    :param similarity_threshold: Seuil de similarité (0 < x < 1)
    :param max_chunk_size: Longueur max d'un chunk en caractères
    :return: Liste de chunks
    """
    chunks = []
    current_chunk = []
    current_length = 0
    embeddings = model.encode(sentences, convert_to_tensor=True)

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        sentence_length = len(sentence)
        if not sentence:
            continue
        if not current_chunk:
            current_chunk = [sentence]
            current_length = sentence_length
            continue
        new_sentence_embedding = embeddings[i]
        last_sentence_embedding = embeddings[sentences.index(current_chunk[-1])]

        # Calcul de la similarité cosinus
        similarity = util.cos_sim(new_sentence_embedding, last_sentence_embedding).item()

        
        if similarity >= similarity_threshold and (current_length + sentence_length) <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def clean_chunk(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def process(texte:str)->list[str]:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name)
    splitted_text=split_into_sentences(texte)
    chunks=chunk_text_intelligently_with_bert(splitted_text,model)
    cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]
    final_chunks=[cleaned for cleaned in cleaned_chunks if len(cleaned)>150]
    
    return final_chunks

    
