from pydantic import BaseModel


class DestinationModel(BaseModel):
    queue: str
    event: str


class RouteModel(BaseModel):
    source: str
    destination: DestinationModel | None


class QuestionModel(BaseModel):
    message: str
    filters: dict | None = None


class ResponseModel(BaseModel):
    message: str


class AMQPPayload(BaseModel):
    communityId: str
    route: RouteModel
    question: QuestionModel
    response: ResponseModel
    metadata: dict | None


class HTTPPayload(BaseModel):
    communityId: str
    question: QuestionModel
    response: ResponseModel | None = None
    taskId: str
