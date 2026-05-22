# chatbot.py
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
from threading import Thread
import time
from datetime import timedelta
import numpy as np
import torch
import logging

# 設置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.collected_tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)
        self.collected_tokens.append(token)

    def on_llm_end(self, **kwargs):
        self.queue.put(None)  # Signal the end of streaming

    def get_complete_response(self):
        return ''.join(self.collected_tokens)

class ChatBot:
    def __init__(self, model_type='smart-factory', gpt_model='phi3', log_path="./chatlog", isref=0):
        self.model_type = model_type
        self.gpt_model = gpt_model
        self.log_path = log_path
        self.isref = isref
        self.setup_model()

    def setup_model(self):
        logger.debug(f"Setting up model: {self.model_type}")
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )

        # Set database paths
        db_paths = {
            'cmp-opt': "./db/cmpOptimization",
            'smart-factory': "/media/r300/1T/A30335/Disc/db/medicalDevice",
            'cmp-assistant': "./db/cmp_agent2",
        }

        # Load vector database
        try:
            self.vectordb = FAISS.load_local(
                db_paths[self.model_type],
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectordb.as_retriever()
            logger.debug("Vector database loaded successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise Exception(f"Database initialization failed: {str(e)}")

        # Setup prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', 'Answer the user\'s questions in Traditional Chinese, based on the context provided below:\n\n{context}'),
            ('user', 'Question: {input}'),
        ])

    def get_streaming_response(self, user_input):
        logger.debug(f"Getting streaming response for: {user_input}")
        # Create a queue for streaming tokens
        queue = Queue()
        callback = StreamingCallbackHandler(queue)

        # Initialize streaming LLM with callback
        llm = Ollama(
            model=self.gpt_model,
            callbacks=[callback],
            # Remove the streaming parameter as it's not supported in the current version
        )

        # Create chains
        document_chain = create_stuff_documents_chain(llm, self.prompt)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)

        # Set context based on model type
        contexts = {
            'cmp-opt': ['Assuming you are an expert in CMP process optimization.'],
            'smart-factory': ['Assuming you are an expert in smart factory and operation communications.'],
            'cmp-assistant': ['Assuming you are an expert in CMP process and manufacturing.']
        }
        context = contexts.get(self.model_type)

        t1 = time.time()
    

        # Run chain in a separate thread to avoid blocking
        def run_chain():
            try:
                logger.debug("Starting chain execution")
                result = retrieval_chain.invoke({
                    'input': user_input,
                    'context': context
                })
                logger.debug(f"Chain execution completed: {result}")
                t2 = time.time()
                td = timedelta(seconds=np.round(t2-t1, 2))
                queue.put(f"\n處理時間：{str(td)}")
                queue.put(None)  # Signal completion
            except Exception as e:
                logger.error(f"Chain execution error: {e}")
                queue.put(f"Error: {str(e)}")
                queue.put(None)

        Thread(target=run_chain).start()
        return queue, callback
    
