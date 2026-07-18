import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloaded_audio'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "cookiefile": "cookies.txt",  
        "quiet": True,
        # Emulates Safari desktop architecture to bypass modern bot checks
        "extractor_args": {
            "youtube": {
                "player_client": ["web_safari"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if 'requested_downloads' in info and info['requested_downloads']:
                return info['requested_downloads'][0]['filepath']
            else:
                filename = ydl.prepare_filename(info)
                return os.path.splitext(filename)[0] + ".wav"
    except Exception as e:
        raise RuntimeError("Cloud IP blocked by YouTube scanner.") from e

def convert_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) 
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 
    chunks = []
    base_name = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        wav_path = download_youtube_audio(source)
    else:
        wav_path = convert_to_wav(source)

    return chunk_audio(wav_path)