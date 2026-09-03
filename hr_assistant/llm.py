from langchain_groq import ChatGroq

from hr_assistant import config

def get_llm():
    return ChatGroq(model_name= config.LLM_MODEL_NAME)