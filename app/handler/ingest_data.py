from loguru import logger
import os
import re
import json
from typing import List
from qdrant_client import QdrantClient
from llama_index import SimpleDirectoryReader, GPTQdrantIndex, LLMPredictor, PromptHelper, StringIterableReader
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from utils.misc import is_file_or_directory, is_private_ip
from __init__ import conf, wid


class IngestData:
    def __init__(self, id_worker: str = None, debug: bool = False):
        if not id_worker:
            self.qd_collection = wid()
        else:
            self.qd_collection = id_worker
        self.work_path = f'{conf["work_dir"]}/{self.qd_collection}'
        self.qd_host = conf['qdrant']['host']
        self.qd_port = conf['qdrant']['port']
        self.qd_url = conf['qdrant']['url']
        self.qd_api_key = conf['qdrant']['api_key']
        self.embeddings = OpenAIEmbeddings()
        self.chat_model = conf['openai']['chat_model']
        self.embedding_model = conf['openai']['embedding_model']
        self.other_model = conf['openai']['other_model']
        self.temperature = conf['openai']['temperature']
        self.max_input_size = conf['llama']['max_input_size']
        self.num_outputs = conf['llama']['num_outputs']
        self.max_chunk_overlap = conf['llama']['max_chunk_overlap']
        self.chunk_size_limit = conf['llama']['chunk_size_limit']
        self.max_token = conf['langchain']['max_token']
        self.k = conf['langchain']['similarity_search_k']
        self.chain_type = conf['langchain']['chain_type']
        if (not self.qd_api_key or is_private_ip(self.qd_host)) and not self.qd_url:
            # if qdrant is local
            logger.success('[+] Connecting to qdrant local...')
            self.client = QdrantClient(host=self.qd_host, port=self.qd_port)
        else:
            # if qdrant is cloud.qdrant.tech
            logger.success('[+] Connecting to qdrant cloud...')
            self.client = QdrantClient(url=self.qd_url, api_key=self.qd_api_key)
        logger.success('[+] Ingest data process initialized successfully!')
        if debug or not id_worker:
            logger.debug(f'[*] Collection name: {self.qd_collection}')

    @logger.catch
    def load_data(self, mode: str = 'langchain'):
        """

        :param mode: the mode of the data.
        :return: the unstructured documents.

        """

        logger.warning(f"[*] Loading data from {self.work_path}...")
        logger.warning(f'[*] Loading data with {mode} mode...')
        if not os.path.exists(self.work_path):
            logger.error(f"[-] {self.work_path} does not exist")
            return None
        if not is_file_or_directory(self.work_path):
            logger.error(f"[-] {self.work_path} is not a file or directory")
            return None
        if mode == 'langchain':
            documents = SimpleDirectoryReader(self.work_path).load_langchain_documents()  # for langchain
        else:
            documents = SimpleDirectoryReader(self.work_path).load_data()  # for llama_index
        logger.success(f'[+] Loaded {self.work_path} successfully!')
        return documents

    @logger.catch()
    def load_text_data(self, texts: List[str]):
        """

        :param texts: the unstructured texts.
        :return: the unstructured documents.

        """
        logger.warning(f"[*] Loading text data...")
        documents = StringIterableReader().load_langchain_documents(texts=texts)
        logger.success(f'[+] Loaded text data successfully!')
        return documents


    @logger.catch
    def split_text(self, documents = None, texts = None):
        """

        :param documents: the unstructured documents.
        :param mode: the parsing mode.
        :return: the split text.

        """
        if documents:
            logger.warning('[*] Starting parsing documents...')
            separators = ["。\n\n", "」\n\n", "\n\n", "\n", ""]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_limit,
                                                           chunk_overlap=self.max_chunk_overlap, separators=separators)
            docs = text_splitter.split_documents(documents)
            self._rm_redundant_newline(docs=docs)
            logger.success('[+] Parsed documents successfully!')
            return docs
        elif texts:
            logger.warning('[*] Starting parsing texts...')
            separators = ["。\n\n", "」\n\n", "\n\n", "\n", ""]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_limit,
                                                           chunk_overlap=self.max_chunk_overlap, separators=separators)
            docs = text_splitter.split_text(texts)
            self._rm_redundant_newline(texts=docs)
            logger.success('[+] Parsed texts successfully!')
            return docs
        else:
            logger.error("[-] No documents or texts to parse!")
            return None

    @logger.catch
    def _rm_redundant_newline(self, docs: List[Document] = None, texts: List[str] = None):
        """

        del "\n\n", "\n" "\x0c"(page breaker), but keep "。\n\n"
        :param docs: the list of split text.

        """
        logger.warning('[*] Starting removing redundant newline...')
        if docs:
            pattern = r"(?<!。)\n\n|(?<!\n)\n(?!\n)|\x0c"
            for doc in docs:
                doc.page_content = re.sub(pattern, "", doc.page_content)
            logger.success('[+] Removed redundant newline successfully!')
        elif texts:
            pattern = r"(?<!。)\n\n|(?<!\n)\n(?!\n)|\x0c"
            for i, text in enumerate(texts):
                texts[i] = re.sub(pattern, "", text)
            logger.success('[+] Removed redundant newline successfully!')
        else:
            logger.error("[-] No documents or texts to remove redundant newline!")

    @logger.catch
    def data2qdrant(self, documents, mode: str = 'langchain'):
        """

        Load Data to vector database.
        :param documents: the embedding documents.
        :param mode: the mode of the data.

        """
        logger.warning('[*] Starting embedding and storing...')
        try:
            if mode == 'langchain':  # M1 use langchain
                if not self.qd_api_key:
                    Qdrant.from_documents(documents, self.embeddings, url=self.qd_url,
                                          collection_name=self.qd_collection)
                else:
                    Qdrant.from_documents(documents, self.embeddings, url=self.qd_url,
                                          collection_name=self.qd_collection, api_key=self.qd_api_key)
            elif mode == 'llama':  # M2 use llama_index
                # define LLM
                LLMPredictor(llm=OpenAI(temperature=self.temperature, model_name=self.other_model,
                                                        max_tokens=self.max_token))
                PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap,
                                             chunk_size_limit=self.chunk_size_limit)
                qclient = QdrantClient(url=self.qd_url, api_key=self.qd_api_key)
                GPTQdrantIndex.from_documents(documents, collection_name=self.qd_collection, client=qclient)
            else:
                logger.error('[-] Mode must be langchain or llama!')
                return False
            logger.success('[+] Embedded and stored successfully!')
            return self.qd_collection
        except Exception as e:
            logger.error(f'[-] Embedded and stored failed: {e}')
            return False

    @logger.catch
    def count_collection(self, collection: str):
        """

        Count the number of documents in the collection.
        :param client: the qdrant client.
        :param collection: the collection name.
        :return: the number of documents in the collection.

        """
        try:
            counts = self.client.count(collection_name=collection)
            logger.success(f'[+] Counted collection: {collection}')
            return counts
        except Exception as e:
            logger.error(e)
            return 0

    @logger.catch
    def get_all_collections(self):
        """

        Get all collections.

        """
        cols = []
        for col in self.client.get_collections().collections:
            cols.append(col.name)
        logger.success(f'[+] Got all collections: {json.dumps(cols, indent=4, ensure_ascii=False)}')
        return cols

    @logger.catch
    def delete_collection(self, collection):
        """

        Delete the collection.
        :param client: the qdrant client.
        :param collection: the collection name.
        :return: True if the collection is deleted successfully, otherwise False.

        """
        try:
            if isinstance(collection, str):
                self.client.delete_collection(collection_name=collection)
                logger.success(f'[+] Deleted collection: {collection}')
            elif isinstance(collection, list):
                for col in collection:
                    self.client.delete_collection(collection_name=col)
                    logger.success(f'[+] Deleted collection: {col}')
            else:
                logger.error('[-] Error type of collection name.')
                return False
            return True
        except Exception as e:
            logger.error(e)
            return False


@logger.catch
def auto_ingest_data(collection_name: str = None, mode: str = 'langchain'):
    """

    Auto ingest data to vector.
    :param mode: The mode to run.
    :param collection_name: The collection name.

    """
    try:
        logger.warning(f'[*] Starting auto ingest data with {mode} mode...')
        id_data = IngestData() if not collection_name else IngestData(collection_name)
        document = id_data.load_data(mode)
        if mode == 'langchain':
            docs = id_data.split_text(documents=document)
        else:
            docs = document
        if not id_data.data2qdrant(docs, mode):
            raise Exception
        logger.success(f'[+] Auto ingest data successfully! your collection name is {id_data.qd_collection}')
        return id_data.qd_collection
    except Exception:
        logger.error('[-] Auto ingest data failed')
        return False


@logger.catch
def ingest_data_case(docs, payload, mode: str = 'langchain'):
    """

    Ingest data case.
    :param docs: the documents.
    :param payload: the payload.
    :param mode: the mode of the data.

    """
    try:
        logger.warning(f'[*] Starting ingest data cases with {mode} mode...')
        idc = IngestData('cases', debug=True)
        document = idc.load_text_data([docs])
        if mode == 'langchain':
            docs = idc.split_text(documents=document)
        if not idc.data2qdrant(docs, mode):
            raise Exception
        logger.success(f'[+] Ingest data cases successfully!')
        return idc.qd_collection
    except Exception:
        logger.error('[-] ingest data cases failed')
        return False


if __name__ == '__main__':
    auto_ingest_data(collection_name='llama_law', mode='llama')
    # idd = IngestData(id_worker='langchain_law', debug=True)

    # collections = idd.get_all_collections()
    # idd.delete_collection(collections)
