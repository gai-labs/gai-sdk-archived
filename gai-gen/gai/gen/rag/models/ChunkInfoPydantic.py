from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

# The ChunkInfo class is a subset of the Chunk model.
# 
class ChunkInfoPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str = Field(...)  # The ellipsis here indicates a required field with no default value
    IsDuplicate: bool
    IsIndexed: bool
    class Config:
        from_attributes = True
