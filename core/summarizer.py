import os 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest", 
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3
    )

def split_transcript(transcript: str) -> list:
    # Slightly lowered chunk size to 2500 for tighter, punchier intermediate summaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300
    )
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely, capturing key discussion points and speaker assertions."),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)

    # ── OPTIMIZATION: Async/Parallel Batching ──
    # This fires all chunk requests simultaneously instead of waiting sequentially.
    batch_inputs = [{"text": chunk} for chunk in chunks]
    chunk_summaries = map_chain.batch(batch_inputs)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Take these chronological partial summaries "
            "and synthesize them into one final, professional executive summary. Use bold headers "
            "and clean bullet points to organize sections logically.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough() 
        | RunnableLambda(lambda x: {"text": x}) 
        | combined_prompt 
        | llm 
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript_summary: str) -> str:
    """
    Generates a professional title based on the overall summary 
    rather than just the first 2000 characters of a raw transcript.
    """
    llm = get_llm()

    title_chain = (
        RunnablePassthrough() 
        | RunnableLambda(lambda x: {"text": x}) 
        | ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting summary provided, generate a short, professional, and descriptive meeting title "
                "(maximum 8 words). Do not include quotes, prefixes like 'Title:', or any filler text. Return ONLY the title.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    # Strip any accidental wrapping quotes the model returns
    return title_chain.invoke(transcript_summary).strip().replace('"', '')
