import os 
from dotenv import load_dotenv

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

#Guard model
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

#Langsmith

LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT", "default")


DATA_FILE_PATH = os.path.join("data","hr_policy.txt")

VECTOR_STORE_PATH = os.path.join("data","faiss_index")


LLM_MODEL_NAME= "openai/gpt-oss-20b"

EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K_RESULTS = 3 


SYSTEM_PROMPT= (
    
    "You are an HR assistant that helps employees with their questions regarding HR policies. "
    "You have access to the company's HR policy documents and can provide accurate and helpful information. "
    "If you don't know the answer, please respond with 'I'm sorry, I don't have that information.'"
)


def check_api_keys()-> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please set it in the .env file.")
    if not JINA_API_KEY:
        raise ValueError("Jina API key is not set. Please set it in the .env file.")
