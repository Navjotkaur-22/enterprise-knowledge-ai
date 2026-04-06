from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore