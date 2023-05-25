from fastapi import UploadFile
from loguru import logger
import os
from __init__ import conf, wid


@logger.catch
def get_col_name():
    collection_name = wid() if not conf['qdrant']['collection_name'] \
        else conf['qdrant']['collection_name']
    return collection_name


@logger.catch
async def receive_file(chat_file: UploadFile):
    try:
        collection_name = get_col_name()
        work_dir = f'{conf["work_dir"].rstrip("/")}/{collection_name}'
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        filepath = f'{work_dir}/{chat_file.filename}'
        with open(filepath, 'wb') as buffer:
            content = await chat_file.read()
            buffer.write(content)
        logger.success(f'[+] Received upload file: {chat_file.filename}')
        return collection_name
    except Exception as e:
        logger.error(f'[-] Received upload file: {chat_file.filename} failed: {e}')
        return False


def save2csv(collection_name: str, filename: str, content: str):
    work_dir = f'{conf["work_dir"].rstrip("/")}/{collection_name}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    filepath = f'{work_dir}/{filename}'
    with open(filepath, 'w') as buffer:
        buffer.write(f'{content}\n')
    return True
