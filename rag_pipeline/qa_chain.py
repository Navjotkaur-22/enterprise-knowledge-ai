from langchain_openai import ChatOpenAI


def generate_answer(query, context_docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Limit context for better relevance
    context_docs = context_docs[:5]

    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a professional AI assistant.

Answer the question ONLY using the provided context.

Rules:
- Do NOT use outside knowledge
- If answer is not in context, say: "I don't know based on the provided documents."

Formatting:
- Use bullet points if needed
- Keep answer clean and concise

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    # -------------------------------
    # CLEAN SOURCE HANDLING
    # -------------------------------
    seen_sources = set()
    clean_sources = []

    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")

        if source not in seen_sources:
            seen_sources.add(source)

            clean_sources.append({
                "source": source,
                "content": doc.page_content[:150]
            })

        if len(clean_sources) >= 3:
            break

    return response.content, clean_sources