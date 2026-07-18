import os
import requests
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import whisper

# Sarvam sync API accepts up to 30 seconds.
SARVAM_PIECE_SECONDS = 25
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
# ── UPGRADE: Shifted from legacy /speech-to-text-translate to the modern endpoint
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

_model = None

def load_model():
    global _model  
    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 

def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()  
    result = model.transcribe(chunk_path, task="transcribe")  
    return result.get("text", "")  

def _send_to_sarvam(piece_data: tuple) -> tuple:
    """Sends one audio fragment to Sarvam. Returns a tuple of (index, text)."""
    index, piece_path = piece_data
    headers = {"api-subscription-key": SARVAM_API_KEY}

    try:
        with open(piece_path, "rb") as f:
            files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
            # ── UPGRADE: Pass saaras:v3 parameters with mode="translate"
            data = {
                "model": SARVAM_MODEL, 
                "mode": "translate",
                "with_diarization": "false"
            }
            
            response = requests.post(
                SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=60,
            )

        if not response.ok:
            print(f"\n❌ Sarvam returned {response.status_code}: {response.text}")
            return index, ""

        return index, response.json().get("transcript", "")
        
    except Exception as e:
        print(f"Error calling Sarvam on piece {index}: {str(e)}")
        return index, ""
    finally:
        # Cleanup temporary slice immediately after dispatching/finishing
        if os.path.exists(piece_path):
            os.remove(piece_path)

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Splits the chunk into 25-second pieces and transcribes them concurrently.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    
    pieces_metadata = []
    start_positions = list(range(0, len(audio), piece_ms))
    
    # Pre-export all temporary fragments to disk
    for i, start in enumerate(start_positions):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        pieces_metadata.append((i, piece_path))

    print(f" ── Dispatching {len(pieces_metadata)} fragments to Sarvam concurrently...")
    
    # ── OPTIMIZATION: Fire all fragment uploads simultaneously over threads ──
    # This prevents sequential network wait times
    results = [None] * len(pieces_metadata)
    with ThreadPoolExecutor(max_workers=min(len(pieces_metadata), 8)) as executor:
        for idx, text in executor.map(_send_to_sarvam, pieces_metadata):
            results[idx] = text

    # Stitch the parts back together in correct sequence chronological order
    full_text = " ".join([text for text in results if text])
    return full_text.strip()

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)

def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = "" 
    engine = "Sarvam AI (Saaras V3)" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    # Note: Whisper runs locally and completely saturates hardware threads, 
    # so we process the top-level chunk array sequentially to avoid race conditions/OOM,
    # but the internal Sarvam handling runs perfectly parallelized.
    for i, chunk in enumerate(chunks):  
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)  
        full_transcript += text + " "  

    print("Transcription complete.")
    return full_transcript.strip()