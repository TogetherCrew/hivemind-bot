from pydantic import BaseModel


class InputModel(BaseModel):
    message: str | None = None
    community_id: str | None = None


class OutputModel(BaseModel):
    destination: str | None = None


class FiltersModel(BaseModel):
    username: list[str] | None = None
    resource: str | None = None
    dataSourceA: dict[str, list[str] | None] | None = None


class PayloadModel(BaseModel):
    input: InputModel
    output: OutputModel
    metadata: dict
    session_id: str | None = None
    filters: FiltersModel | None = None
