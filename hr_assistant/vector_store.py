"""Step4 store embeddings in FAISS so we can search them  """

import os 
from langchain_community.vectorstores import FAISS

from hr_assistant import config 
from hr_assistant.embeddings import get_embedding_model 
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def build_vector_store(chunks):
    logger.info("Embeddings %d chunk(s) and building FAISS index..")
    embeddings_model = get_embedding_model()
    vector_store= FAISS.from_documents(chunks,embeddings_model)
    logger.info("FAISS index built in memory")
    return vector_store
## saving data
def save_vector_store(vector_store,path:str =config.VECTOR_STORE_PATH )-> None:
    """Save the FAISS index to disk, we don't have to rebuild it every time """
    vector_store.save_local(path)
    logger.info("Saved FAISS index to '%s",path)

def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """LLM needs the store data to generate answers """
    logger.info("Loading FAISS index from '%s",path)
    embeddings_model = get_embedding_model()



    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization = True)


def vector_store_exists(path:str = config.VECTOR_STORE_PATH)-> bool:
    """check if the saved faiss index is already present in the disk"""
    return os.path.exists(os.path.join(path,"index.faiss"))


def get_retriever(vector_store,k:int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that returns top k results"""
    logger.info("Creating retriever with top_k= %d",k)

    return vector_store.as_retriever(search_kwargs={"k":k}  )
