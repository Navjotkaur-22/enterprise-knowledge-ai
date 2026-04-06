from langchain_openai import ChatOpenAI


def compare_documents(query, context_docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are an expert data analyst.

Your task is to compare information from multiple datasets.

IMPORTANT:
- The context contains data from different files (employee, sales, etc.)
- Even if datasets are different, you must still extract and compare relevant information

Instructions:
1. Identify different datasets in the context
2. Extract key information from each dataset
3. Present a comparison in a structured way

Rules:
- Do NOT say "comparison not possible" unless absolutely no data exists
- Do NOT use outside knowledge
- Use ONLY provided context

Output format:
- Clearly mention each dataset
- Use bullet points
- Provide a meaningful comparison

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content