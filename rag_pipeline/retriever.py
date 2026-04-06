def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever