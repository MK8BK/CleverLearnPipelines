import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
from pipelines.base_pipeline import Pipeline
from llms.openai import OpenAI_client, OpenAI_role
from pydantic import BaseModel

class DistractorResponse(BaseModel):
    distractors: list[str]
    
class DistractorGenerator(Pipeline):
    
    def __init__(self, *args, **kwargs):
        super().__init__("distractor_generator_pipeline")
        self.client = OpenAI_client()  

    def _process(self, input_data: dict) -> str:
        question = input_data.get("question")
        answer = input_data.get("answer")
        if not question or not answer:
            raise ValueError("Input must include both 'question' and 'answer' fields.")
        
        prompt = f"""You are an expert in educational content creation. Your task is to generate challenging multiple-choice distractors.

                    Given a question and its correct answer, return exactly 3 incorrect options (called 'distractors') in valid JSON format.
                    Guidelines:
                    - Distractors must be thematically related to the correct answer.
                    - Avoid distractors that are obviously incorrect or completely unrelated.
                    - Avoid distractors that are too close in meaning or wording to the correct answer.
                    - Make sure there's no ambiguity: only one correct answer should be possible.

                    

                    Format:
                    {{ "distractors": ["...", "...", "..."] }}

                    Question: {question}
                    Correct Answer: {answer}"""

        self.client.clear_messages()
        self.client.add_message(OpenAI_role.USER, prompt)

        raw_output = self.client.submit_messages(response_format=DistractorResponse)

        distractors = raw_output.distractors if raw_output else []

        result = {
            "question": question,
            "answer": answer,
            "distractors": distractors
        }

        return json.dumps(result, indent=2, ensure_ascii=False)


    def _validate(self, input_data, output_data) -> bool:
        try:
            parsed = json.loads(output_data)
            return "question" in parsed and "answer" in parsed and isinstance(parsed.get("distractors"), list)
        except Exception:
            return False
        
dg=DistractorGenerator()
print(dg._process({
            "question": "What is the Capital of Spain",
            "answer": "Madrid"
        }))