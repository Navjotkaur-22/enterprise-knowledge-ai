from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
import pandas as pd
from langchain_core.documents import Document


def load_documents(file_paths):
    documents = []

    for file_path in file_paths:

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
            documents.extend(loader.load())

        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")

            for _, row in df.iterrows():
                text = " ".join([str(value) for value in row.values])
                documents.append(Document(page_content=text, metadata={"source": file_path}))

        else:
            print(f"Unsupported file type: {file_path}")

    return documents