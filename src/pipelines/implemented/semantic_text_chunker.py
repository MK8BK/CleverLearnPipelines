from typing import List
from itertools import chain
import re

# Temporary -----------------------------------------------------
# TODO: uncomment to test this particular pipeline
import sys
sys.path.insert(0, '../..')
# ---------------------------------------------------------------

from pipelines.base_pipeline import Pipeline
from sentence_transformers import SentenceTransformer, util


class SemanticTextChunker(Pipeline):
    """
    """
    title = "semantic_text_chunker_pipeline"

    def __init__(self, *args, **kwargs):
        super().__init__(SemanticTextChunker.title)

    def split_into_sentences(self, text: str) -> List[str]:
        # assumes all markdown links have been removed
        # regex to capture classic end of sentences, TODO: review later
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZÉÈÂÎÔÙ])')
        sentences = sentence_endings.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text_intelligently_with_bert(
        self,
        sentences: List[str],
        model,
        # TODO: test these defaults later and document how they were established
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
            last_sentence_embedding = embeddings[sentences.index(
                current_chunk[-1])]
            # cosine similarity evaluation
            similarity = util.cos_sim(
                new_sentence_embedding, last_sentence_embedding).item()
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

    def clean_chunk(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _process(self, input_data: str) -> List[str]:
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model = SentenceTransformer(model_name)
        splitted_text = self.split_into_sentences(input_data)
        chunks = self.chunk_text_intelligently_with_bert(splitted_text, model)
        cleaned_chunks = [self.clean_chunk(chunk) for chunk in chunks]
        # TODO: check 150 here
        final_chunks = [
            cleaned for cleaned in cleaned_chunks if len(cleaned) > 150]
        return final_chunks

    def _validate(self, input_data, output_data):
        return True


if __name__ == "__main__":

    from corpus.corpus import Corpus
    from test_helpers import TEST_DATA_PATH, measure_time
    from index import WikiTestDataIndex

    index = WikiTestDataIndex(TEST_DATA_PATH)

    stc = SemanticTextChunker()
    index.ensure_pipeline_dir(stc.title)

    with open(str(TEST_DATA_PATH.joinpath("wiki").joinpath("1.md")),
              "r", encoding="utf8") as f:
        corpus = Corpus(f.read())

    from json import dumps

    @measure_time
    def p(t):
        return stc.process(t)
    pipeline_output = p(corpus.clean_text)
    # print(len(pipeline_output))
    pipeline_output = dumps(pipeline_output)
    index.store_pipeline_output(stc.title, pipeline_output, "out1.json")
