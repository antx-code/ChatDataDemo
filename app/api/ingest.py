from fastapi import APIRouter, BackgroundTasks, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
import asyncio
import uvloop
import json
from handler.ingest_data import auto_ingest_data, ingest_data_case
from model.ingest_model import IngestModel
from utils.recv_file import receive_file
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
uvloop.install()


router = APIRouter()


@logger.catch
@router.post("/upload")
async def create_upload_file(chat_file: UploadFile, background_tasks: BackgroundTasks):
    logger.success(f'[+] Received upload file: {chat_file.filename}')
    collection_name = await receive_file(chat_file)
    if not collection_name:
        return JSONResponse(status_code=400, content={"message": f"Upload {chat_file.filename} failed!"})
    background_tasks.add_task(auto_ingest_data, collection_name=collection_name)
    return JSONResponse(status_code=200, content={"message": f"Upload {chat_file.filename} successfully!"})


@logger.catch
@router.post("/ingest")
async def create_ingest_from_api(ingest_model: IngestModel, background_tasks: BackgroundTasks):
    logger.success(f'[+] Received docs from api: {ingest_model.case_uuid}')
    collection_name = 'cases'
    docs = f'"{ingest_model.chat_name}"在 {ingest_model.publish_time} 发布了:标题为"{ingest_model.case_title}"的公众号文章,' \
           f'内容为:\n{ingest_model.case_content}'
    payload = {
        'case_uuid': ingest_model.case_uuid,
        'case_title': ingest_model.case_title,
        'case_content': ingest_model.case_content,
        'publish_time': ingest_model.publish_time,
        'chat_name': ingest_model.chat_name
    }
    logger.success(f'[+] {ingest_model.case_uuid}')
    logger.success(f'[+] {json.dumps(payload, indent=4, ensure_ascii=False)}')
    background_tasks.add_task(ingest_data_case, docs=docs, payload=payload)
    return JSONResponse(status_code=200, content={"message": f"Received {ingest_model.case_uuid} successfully!"})