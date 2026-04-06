from langchain_openai import ChatOpenAI


def generate_answer(query, context_docs):
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Step 1: Limit context
    context_docs = context_docs[:5]

    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a professional AI assistant.

Answer the question ONLY using the provided context.

Rules:
- Do NOT use outside knowledge
- If answer is not in context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    # -------------------------------
    # SMART SOURCE FILTERING
    # -------------------------------
    query_lower = query.lower()

    filtered_sources = []
    seen = set()

    for doc in context_docs:
        source = doc.metadata.get("source", "").lower()

        # Match query intent with file type
        if ("ai" in query_lower or "artificial intelligence" in query_lower) and "ai" not in source:
            continue

        if ("salary" in query_lower or "employee" in query_lower) and "employee" not in source:
            continue

        if ("sales" in query_lower or "product" in query_lower) and "sales" not in source:
            continue

        if source not in seen:
            seen.add(source)

            filtered_sources.append({
                "source": source,
                "content": doc.page_content[:150]
            })

        if len(filtered_sources) >= 3:
            break

    return response.content, filtered_sources