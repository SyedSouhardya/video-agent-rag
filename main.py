from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
# Note: Renamed core.summarizer import path to match your layout module naming structure
from core.summarizer import summarize, generate_title 
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source: str, language: str = "english") -> dict:
    print("\n🚀 Starting AI Video Assistant Pipeline...")

    # 1. Process and extract audio chunks
    chunks = process_input(source)

    # 2. Transcribe chunks
    transcript = transcribe_all(chunks, language)
    print(f"\n[Info] Raw transcription sample (first 300 chars): \n{transcript[:300]}...")

    # 3. ── FIX: Generate summary first ──
    print("\n✨ Generating meeting summary...")
    summary = summarize(transcript)

    # 4. ── FIX: Pass the summary to title generator instead of raw text ──
    print("✨ Creating professional title...")
    title = generate_title(summary)

    # 5. Extract structured insight matrices
    print("✨ Extracting action items...")
    action_items = extract_action_items(transcript)

    print("✨ Extracting key decisions...")
    decisions = extract_key_decisions(transcript)

    print("✨ Extracting open questions...")
    questions = extract_questions(transcript)
    
    # 6. Initialize Vector Database context framework
    print("✨ Building RAG vector space indices...")
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }

if __name__ == "__main__":
    # CLI execution lifecycle loop
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish) [default: english]: ").strip() or "english"
    
    if not source:
        print("❌ Error: Source path or URL cannot be empty.")
        exit(1)
        
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Interactive RAG chat framework
    print("\n💬 Chat session active. Ask questions about the meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    
    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Session closed. Goodbye!")
                break
            if not question:
                continue
                
            answer = ask_question(rag_chain, question)
            print(f"\n🤖 Assistant: {answer}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break