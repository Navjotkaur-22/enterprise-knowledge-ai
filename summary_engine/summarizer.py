from langchain_openai import ChatOpenAI


def summarize_documents(query, context_docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Group content by source file
    grouped_data = {}

    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")

        if source not in grouped_data:
            grouped_data[source] = []

        grouped_data[source].append(doc.page_content)

    # Build structured context
    structured_context = ""

    for source, contents in grouped_data.items():
        combined_text = "\n".join(contents[:10])  # limit per file
        structured_context += f"\n\nDataset Source: {source}\n{combined_text}"

    # Improved prompt
    prompt = f"""
You are an expert data analyst.

Your task is to summarize multiple datasets.

IMPORTANT:
- Each "Dataset Source" is a DIFFERENT dataset
- DO NOT mix data from different sources
- Summarize each dataset separately

Instructions:
1. Identify each dataset by its source
2. Summarize key points for each dataset
3. Keep output clean and structured

Rules:
- Do NOT merge datasets
- Do NOT mix employee data with sales data
- Use ONLY provided context

Output format:

Dataset: <file name>
- key points

Dataset: <file name>
- key points

Context:
{structured_context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content