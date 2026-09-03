"""convert chunks into embeddings """


from langchain_community.embeddings import JinaEmbeddings

from hr_assistant import config


def get_embedding_model():
    """Return jina embeddings """
    return JinaEmbeddings(model_name= config.EMBEDDING_MODEL_NAME)