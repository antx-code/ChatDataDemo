from loguru import logger
import openai
from qdrant_client import QdrantClient
from llama_index.readers.qdrant import QdrantReader
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from utils.misc import is_private_ip
from __init__ import conf

KEYWORD_TEMPLATE = """你是一个AI，你的工作是回答关于网络安全法律和执法案例的问题。请根据之前的讨论为我的问题提供详细的关键词。在关键词中不需要特别包含 网络安全法律或者执法案例。
问题: {question}
关键词:"""

KEYWORD_PROMPT = PromptTemplate(template=KEYWORD_TEMPLATE, input_variables=["question"])

QA_TEMPLATE = """你是一个AI，你的工作是回答关于网络安全法律和执法案例的问题。
你收到的输入是一个网络安全法律问题和一些相关的上下文，请提供一个对话式的回答，回答请尽量详细并提供原文的依据位置。
如果你不知道答案，请不要试着自己编造一个答案，直接说“我不确定应该如何回答这个问题”就可以了。
如果这个问题不是关于网络安全法律的，请礼貌地回答你只解答关于网络安全的问题, 比如“很抱歉，这个问题似乎与网络安全法律没有直接关系。作为AI，我的职责是回答关于网络安全法律和执法案例的问题。如果您有相关的问题，请随时向我提出。”。
问题: {question}
开示节选：
{context}
回答:"""

QA_PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["question", "context"])


class QueryData:
    def __init__(self, id_worker: str = None, debug: bool = False):
        if not id_worker:
            raise ValueError('id_worker is None!')
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
            logger.success('[+] Connecting to qdrant local...')
            self.client = QdrantClient(self.qd_host)
        else:
            logger.success('[+] Connecting to qdrant cloud...')
            self.client = QdrantClient(url=self.qd_url, api_key=self.qd_api_key)
        logger.success('[+] Query data process initialized successfully!')
        if debug or not id_worker:
            logger.debug(f'[*] Collection name: {self.qd_collection}')

    @logger.catch
    def _text2vec(self, text: str):
        """

        Convert text to vector.
        :param text: the text to be converted

        """
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text,
            max_tokens=1,
            stop=["\n"],
        )
        logger.success(f'[+] Embedding text: "{text}" successfully!')
        return response['data'][0]['embedding']

    @logger.catch
    def langchain_chat_once(self):
        llm = OpenAI(temperature=self.temperature, max_tokens=self.max_token)
        chain = load_qa_chain(llm, chain_type=self.chain_type, prompt=QA_PROMPT)
        return chain

    @logger.catch
    def langchain_chat(self):
        llm = ChatOpenAI(temperature=self.temperature, model_name=self.chat_model, max_tokens=self.max_token)
        chain = load_qa_chain(llm, chain_type=self.chain_type, prompt=QA_PROMPT)
        return chain

    @logger.catch
    def chat_data(self, question: str, chat: bool = True, mode: str = 'langchain'):
        if mode == 'langchain':
            qclient = Qdrant(client=self.client, collection_name=self.qd_collection,
                             embedding_function=self.embeddings.embed_query)
            docs = qclient.similarity_search(question, k=self.k)
            chain = self.langchain_chat() if chat else self.langchain_chat_once()
            response = chain.run(input_documents=docs, question=question)
            logger.success(f'[+] Langchain Chat response: {response}')
            return response
        elif mode == 'llama':
            LLMPredictor(llm=OpenAI(temperature=self.temperature, model_name=self.other_model,
                                                    max_tokens=self.max_token))
            PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap,
                                         chunk_size_limit=self.chunk_size_limit)
            qclient = QdrantReader(host=self.qd_url, port=self.qd_port, api_key=self.qd_api_key) if self.qd_api_key \
                else QdrantReader(host=self.qd_url, port=self.qd_port)
            vectors = self._text2vec(question)
            docs = qclient.load_data(collection_name=self.qd_collection, query_vector=vectors, limit=self.k)
            index = GPTSimpleVectorIndex.from_documents(docs)
            # index = GPTListIndex.from_documents(docs)
            response = index.query(question)
            logger.success(f'[+] Llama Chat response: {response}')
            return response
        else:
            raise ValueError('[-] Mode must be langchain or llama!')

    @logger.catch
    def chat_data_with_case(self, question: str, chat: bool = True, mode: str = 'langchain'):
        if mode == 'langchain':
            qclient = Qdrant(client=self.client, collection_name=self.qd_collection,
                             embedding_function=self.embeddings.embed_query)
            docs = qclient.similarity_search(question, k=self.k)
            chain = self.langchain_chat() if chat else self.langchain_chat_once()
            response = chain.run(input_documents=docs, question=question)
            logger.success(f'[+] Langchain Chat response: {response}')
            return response
        elif mode == 'llama':
            LLMPredictor(llm=OpenAI(temperature=self.temperature, model_name=self.other_model,
                                                    max_tokens=self.max_token))
            PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap,
                                         chunk_size_limit=self.chunk_size_limit)
            qclient = QdrantReader(host=self.qd_url, port=self.qd_port, api_key=self.qd_api_key) if self.qd_api_key \
                else QdrantReader(host=self.qd_url, port=self.qd_port)
            vectors = self._text2vec(question)
            docs = qclient.load_data(collection_name=self.qd_collection, query_vector=vectors, limit=self.k)
            index = GPTSimpleVectorIndex.from_documents(docs)
            # index = GPTListIndex.from_documents(docs)
            response = index.query(question)
            logger.success(f'[+] Llama Chat response: {response}')
            return response
        else:
            raise ValueError('[-] Mode must be langchain or llama!')


if __name__ == '__main__':
    question = '都有哪些网络安全的法律法规'

    qd = QueryData(id_worker='langchain_law', debug=True)  # for langchain
    qd.chat_data(question=question, mode='langchain')

    # qd = QueryData(id_worker='llama_law', debug=True)
    # qd.chat_data(question=question, mode='llama')  # for llama
