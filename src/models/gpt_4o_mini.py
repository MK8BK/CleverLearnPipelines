from typing import List
from openai import OpenAI
from pydantic import BaseModel
from enum import StrEnum
from dataclasses import dataclass
import os

# https://platform.openai.com/docs/models/gpt-4o-mini
GPT_4O_MINI = "gpt-4o-mini-2024-07-18"


class OpenAI_role(StrEnum):
    USER = "user"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: OpenAI_role
    text: str

    def to_dict(self):
        return {"role": f"{self.role}", "content": self.text}


class Gpt_4o_mini_client:
    def __init__(self):
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.messages: List[Message] = []

    def add_message(self, role: OpenAI_role, message: str):
        self.messages.append(Message(role, message))
        return self  # builder design pattern

    def clear_messages(self):
        self.messages = []

    def submit_messages(self, **kwargs):
        """Makes the actual api call to the gpt-4o-mini model
            Parameters ----
            response_format: pydantic.BaseModel.__class__, optional
                the expected response json format as a pydantic class
        """
        # TODO: see the documentation of the class
        # openai.types.chat.parsed_chat_completion.ParsedChatCompletion
        chat_completion = self._client.beta.chat.completions.parse(
            messages=[message.to_dict() for message in self.messages],
            model=GPT_4O_MINI,
            **kwargs
        )
        return chat_completion 

