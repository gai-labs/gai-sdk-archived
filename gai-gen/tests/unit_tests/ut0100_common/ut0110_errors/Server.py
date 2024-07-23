# prettier-ignore
import asyncio
import os, sys
import uuid
from fastapi import FastAPI, Response, WebSocket, Header, Body, UploadFile, File
from typing import Optional
from fastapi.responses import JSONResponse
from gai_common.errors import ApiException, ContextLengthExceededException, GeneratorMismatchException, InternalException, MessageNotFoundException, UserNotFoundException
from pydantic import BaseModel
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "..", "..", ".."))

router = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
router.add_middleware(
    CORSMiddleware,
    allow_origins="http://localhost:12031",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@router.get("/ApiException")
async def raise_ApiException():
    raise ApiException(status_code=409, code='server_error', message='Server Error')

@router.get("/JSONResponse")
async def return_JSONResponse():
    return JSONResponse(status_code=200, content={"text":'output'})

@router.get("/MessageNotFoundException/{message_id}")
async def raise_MessageNotFoundException(message_id:str):
    raise MessageNotFoundException(message_id=message_id)

@router.get("/InternalException")
async def raise_InternalException():
    id = str(uuid.uuid4())
    raise InternalException(id)

@router.get("/UserNotFoundException/{user_id}")
async def raise_UserNotFoundException(user_id:str):
    raise UserNotFoundException(user_id=user_id)

@router.get("/ContextLengthExceededException")
async def raise_ContextLengthExceededException():
    raise ContextLengthExceededException()

@router.get("/GeneratorMismatchException")
async def raise_GeneratorMismatchException():
    raise GeneratorMismatchException()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=12031)
