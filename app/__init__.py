import os
from loguru import logger
from yaml import load, Loader
from utils.snow_flake import IdWorker


@logger.catch
def load_yaml(path: str):
    """load yaml file"""
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    logger.info(f'path: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        data = load(f, Loader=Loader)
    return data


conf = load_yaml('config.yaml')
id_worker = IdWorker(0, 0)


@logger.catch
def wid():
    return str(id_worker.get_id())


openai_key = conf['openai']['api_key']
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["HTTP_PROXY"] = conf['openai']['proxy']
os.environ["HTTPS_PROXY"] = conf['openai']['proxy']
