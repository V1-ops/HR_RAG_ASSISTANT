"""Step4 store embeddings in FAISS so we can search them  """

import os 
from langchain_community.vectorstores import FAISS

from hr_assistant import config 
from hr_assistant.embeddings import get_embedding_model 

def build_vector_store(chunks):
    embeddings_model = get_embedding_model()
    return FAISS.from_documents(chunks,embeddings_model)

## saving data
def save_vector_store(vector_store,path:str =config.VECTOR_STORE_PATH )-> None:
    """Save the FAISS index to disk, we don't have to rebuild it every time """
    vector_store.save_local(path)

def load_vector_store(path: str = config.VECTOR_STORE_PATH):
    """LLM needs the store data to generate answers """
    embeddings_model = get_embedding_model()


    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization = True)


def vector_store_exists(path:str = config.VECTOR_STORE_PATH)-> bool:
    """check if the saved faiss index is already present in the disk"""
    return os.path.exists(os.path.join(path,"index.faiss"))


def get_retriever(vector_store,k:int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever that returns top k results"""

    return vector_store.as_retriever(search_kwargs={"k":k}  )
