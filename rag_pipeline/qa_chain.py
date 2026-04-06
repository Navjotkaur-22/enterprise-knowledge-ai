from langchain_openai import ChatOpenAI


def generate_answer(query, context_docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Combine context
    context = "\n\n".join([doc.page_content for doc in context_docs])

    # Structured + strict prompt
    prompt = f"""
You are a professional AI assistant.

Answer the question ONLY using the provided context.

Rules:
- Do NOT use outside knowledge
- If answer is not in context, say: "I don't know based on the provided documents."

Formatting Rules:
- Provide a clear and structured answer
- Use bullet points where possible
- Keep answer concise and professional

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    # Prepare citations
    sources = []
    for doc in context_docs:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content[:200]
        })

    return response.content, sources