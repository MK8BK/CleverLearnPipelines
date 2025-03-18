"""
index.json structure
{
    "quiz_count": q,
    "doc_count": d,
    "docs": {
        url1:1,
        url2:2,
        url3:3,
        ...
    },
    "quizzes":{
        doc_id:[n1, n3, ...],
    }

}
"""

import os
import pathlib
import shutil
import json
import itertools
from typing import Dict, List


class WikiTestDataIndex:

    def clear(self):
        self.quiz_count = 0
        self.doc_count = 0
        self.docs = {}
        self.quizzes = {}

    def empty_dirs(self):
        if os.path.isdir(self.wiki_path):
            shutil.rmtree(self.wiki_path)
            os.mkdir(self.wiki_path)
        if os.path.isdir(self.quizzes_path):
            shutil.rmtree(self.quizzes_path)
            os.mkdir(self.quizzes_path)

    def create_dirs(self):
        if not os.path.exists(self.quizzes_path):
            os.mkdir(self.quizzes_path)
        if not os.path.exists(self.quizzes_path):
            os.mkdir(self.wiki_path)

    def __init__(self, data_path: pathlib.Path):
        self.data_path = data_path
        self.index_path = self.data_path.joinpath("index.json")
        self.quizzes_path = self.data_path.joinpath("quizzes")
        self.pipelines_path = self.data_path.joinpath("pipelines")
        self.wiki_path = self.data_path.joinpath("wiki")
        if not all(os.path.exists(p) for p in [self.index_path, self.quizzes_path, self.wiki_path]):
            self.clear()
            self.persist()
            self.create_dirs()
            self.empty_dirs()
            return

        with open(str(self.index_path), "r", encoding="utf8") as f:
            self.index = json.loads(f.read())

        self.quiz_count: int = self.index["quiz_count"]
        self.doc_count: int = self.index["doc_count"]
        self.docs: Dict[str, int] = self.index["docs"]
        self.quizzes: Dict[str, List[int]] = self.index["quizzes"]

    def already_scraped(self, url: str):
        return url in self.docs

    def get_quizzes(self, url: str):
        if url not in self.docs or self.docs[url] not in self.quizzes:
            raise RuntimeError(
                f"{url} has never been scraped or invalid class state.")
        return self.quizzes[self.doc[url]]

    def ensure_pipeline_dir(self, pipeline_title: str):
        pipeline_dir_path = self.pipelines_path.joinpath(pipeline_title)
        if not os.path.exists(pipeline_dir_path):
            os.mkdir(pipeline_dir_path)

    def store_pipeline_output(self, pipeline_title: str, output, file_name: str):
        self.ensure_pipeline_dir(pipeline_title)
        pipeline_dir_path = self.pipelines_path.joinpath(pipeline_title)
        file_path = pipeline_dir_path.joinpath(file_name)
        with open(file_path, "w", encoding="utf8") as f:
            f.write(output)

    def add_document(self, url, content: str):
        if url in self.docs:
            doc_path = self.wiki_path.joinpath(str(self.docs[url])+".md")
        else:
            self.doc_count += 1
            doc_path = self.wiki_path.joinpath(str(self.doc_count)+".md")
            self.docs[url] = self.doc_count
            self.quizzes[self.docs[url]] = []
        with open(doc_path, "w", encoding="utf8") as f:
            f.write(content)
        self.persist()

    def add_quiz(self, url, quiz_content: str):
        if url not in self.docs or url not in self.quizzes:
            raise RuntimeError(
                "Quiz has no associated corpus or invalid class state.")
        self.quiz_count += 1
        self.quizzes[self.docs[url]].append(self.quiz_count)
        quiz_path = self.quizzes_path.joinpath(self.quiz_count, ".json")

        with open(quiz_path, "w", encoding="utf8") as f:
            f.write(quiz_content)
        self.persist()

    def remove_quiz_versions(self):
        """
            Keep only the last quiz for each document.
        """
        registered_to_be_dropped = [quizzes[:-1]
                                    for quizzes in self.quizzes.values()]
        to_be_dropped = itertools.chain.from_iterable(registered_to_be_dropped)
        # print(to_be_dropped)

    def persist(self):
        with open(str(self.index_path), "w", encoding="utf8") as f:
            f.write(json.dumps(
                {
                    "quiz_count": self.quiz_count,
                    "doc_count": self.doc_count,
                    "docs": self.docs,
                    "quizzes": self.quizzes,
                }, indent=2,
            ))

    def retrieve_doc(self, url):
        if url not in self.docs:
            raise RuntimeError(f"{url} has never been scraped.")
        doc_path = self.wiki_path.joinpath(str(self.docs[url])+".md")
        if not os.path.exists(doc_path):
            raise RuntimeError("Invalid index state.")
        with open(doc_path, "r", encoding="utf8") as f:
            return f.read()
