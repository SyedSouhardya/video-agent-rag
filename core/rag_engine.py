import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_optimized_prompt():
    """
    Returns a balanced prompt that encourages thorough analysis of the transcript
    without inducing aggressive, unhelpful refusals.
    """
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an elite corporate intelligence assistant. Your goal is to analyze the provided meeting transcript context to fully address the user's query.

Guidelines:
1. Synthesize information across the chunks completely to answer thoroughly. 
2. If the context explicitly mentions a topic but doesn't give a direct answer, use logical inference based on the meeting dialogue flow, explicitly noting it as a reasonable inference.
3. If the topic is entirely absent from the text, only then state: "I could not find this information in the meeting transcript."
4. Be structured, deep, and highly actionable. Use bullet points for multi-part answers. Highlight who said what when relevant.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ])

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    
    # Bumped k to 6 to fetch significantly more conversational context
    retriever = get_retriever(vector_store, k=6)
    llm = get_llm()
    prompt = get_optimized_prompt()

    # Full LCEL RAG pipeline 
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def load_rag_chain():
    vector_store = load_vector_store()
    
    # FIX: Passed vector_store here so the retriever knows what to query, set k=6
    retriever = get_retriever(vector_store, k=6)

    llm = get_llm()
    prompt = get_optimized_prompt()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer   : {answer}")
    return answer