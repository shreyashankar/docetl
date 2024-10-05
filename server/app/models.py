from pydantic import BaseModel


class PipelineRequest(BaseModel):
    yaml_config: str
