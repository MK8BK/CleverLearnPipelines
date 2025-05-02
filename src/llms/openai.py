from typing import List
from openai import OpenAI
from pydantic import BaseModel
from enum import StrEnum
from dataclasses import dataclass
import concurrent
import os

"""
We use the 4o-mini model by default, saves costs.
[gpt-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini)
"""
GPT_4O_MINI = "gpt-4o-mini-2024-07-18"


class OpenAI_role(StrEnum):
    """
        Enumeration of the different open ai message roles
        see 
        [OpenAI api roles](https://platform.openai.com/docs/guides/text?api-mode=chat#message-roles-and-instruction-following)
        for more details.
    """
    USER = "user"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    A message helper class, encodes a message as a str and the openai_role
    [prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)
    first section on messages and roles.
    """
    role: OpenAI_role
    text: str

    def to_dict(self):
        """
        Helper method to produce a compatible dict structure to feed to the
        OpanAI api.
        """
        return {"role": f"{self.role}", "content": self.text}


class OpenAI_client:
    NTHREADS: int = 10
    """
    Class to interact with remote openai models.
    """

    """
    A static OpenAI _client to limit the overhead of OpenAI_client instance 
    construction.
    """
    _client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    def __init__(self, model_id: str = GPT_4O_MINI):
        """
        Constructor.

        model_id: a unique openai model identifier, 
            see snapshots section of models at 
            [openai models](https://platform.openai.com/docs/models).
        """
        self.model_id = model_id


    def concurrent_submit_messages(self, messages: List[List[Message]], max_workers: int = NTHREADS, **kwargs) -> List:
        """
            UNTESTED
        """
        cf = concurrent.futures
        with cf.ThreadPoolExecutor(max_workers=OpenAI_client.NTHREADS) as executor:
            results = executor.map(lambda m: self.submit_messages(m, **kwargs), messages)
            return list(results)

    def submit_messages(self, messages: List[Message], **kwargs):
        """Makes the actual api call to the model specified in model_id.
            Submits all messages in the message queue.

            response_format: pydantic.BaseModel.__class__, optional

                the expected response json format as a pydantic class

            returns a parsed chat completion (a.k.a instance of 
                openai.types.chat.parsed_chat_completion.ParsedChatCompletion)
        """
        # See the documentation of the class
        # openai.types.chat.parsed_chat_completion.ParsedChatCompletion
        # for more details.
        chat_completion = OpenAI_client._client.beta.chat.completions.parse(
            messages=[message.to_dict() for message in messages],
            model=self.model_id,
            **kwargs
        )
        message_content = chat_completion.choices[0].message.parsed
        return message_content


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI_client()
