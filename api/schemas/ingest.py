from pydantic import BaseModel


class IngestResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    message: str
