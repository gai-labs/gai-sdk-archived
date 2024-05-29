from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

class IndexedDocumentChunkPydantic(BaseModel):
    Id: str = Field(...)  # The ellipsis here indicates a required field with no default value
    ChunkHash: str
    ChunkGroupId: str
    ByteSize: int
    IsDuplicate: bool
    IsIndexed: bool
    Content: Optional[str] = None

    # If you need to represent the relationship to IndexedDocumentChunkGroup as a nested model:
    # ChunkGroup: Optional[IndexedDocumentChunkGroupPydantic] = None

    class Config:
        from_attributes = True
