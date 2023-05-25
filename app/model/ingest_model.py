from pydantic import BaseModel


class IngestModel(BaseModel):
    case_title: str
    case_content: str
    case_uuid: str
    publish_time: str
    chat_name: str
