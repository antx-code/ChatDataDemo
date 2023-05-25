from pydantic import BaseModel


class ChatModel(BaseModel):
    question: str
