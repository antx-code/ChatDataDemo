from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger
import asyncio
import uvloop
from handler.query_data import QueryData
from model.chat_model import ChatModel
from utils.recv_file import get_col_name
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
uvloop.install()


router = APIRouter()


@logger.catch
@router.post("/chat")
async def query_and_chat(chat: ChatModel):
    collection_name = get_col_name()
    try:
        qd = QueryData(id_worker=collection_name)
        result = qd.chat_data(question=chat.question)
        return JSONResponse(status_code=200, content={'answer': result})
    except Exception as e:
        logger.error(f'[-] Chat failed: {e}')
        return JSONResponse(status_code=400, content={'answer': 'Chat failed!'})


@logger.catch
@router.post("/chat_with_case")
async def query_and_chat_with_case(chat: ChatModel):
    collection_name = get_col_name()
    try:
        qd = QueryData(id_worker=collection_name)
        result = qd.chat_data_with_case(question=chat.question)
        return JSONResponse(status_code=200, content={'answer': result})
    except Exception as e:
        logger.error(f'[-] Chat failed: {e}')
        return JSONResponse(status_code=400, content={'answer': 'Chat failed!'})
