import yt_dlp
import os
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "noplaylist": True,

    "ffmpeg_location": r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin",

    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
    }],

    "quiet": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_name = ydl.prepare_filename(info).replace(".webm",".wav").replace(".m4a",".wav")
    return file_name



def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file to WAV format using pydub.

    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path) # detect the type of file
    audio=audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")

    return output_path




# for chunk Large audio files

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:  # milisection covertor
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60*1000   

    chunks =[]

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)

    return chunks


def process_audio(input_path: str) -> list:
    """
    Detect YouTube URL or local file and process automatically.
    """
    if input_path.startswith(("http://", "https://")):
        print("Detected Youtube URL. Downloading audio...")
        wav_file = download_youtube_audio(input_path)
        wav_file = convert_to_wav(wav_file)
    else:
        print("Detected local file. Converting to WAV...")
        wav_file = convert_to_wav(input_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_file)
    print(f"Audio ready - {len(chunks)} chuck(s) created.")
    return chunks


if __name__ == "__main__":
    input_path = input("Enter YouTube URL or local file path: ")
    chunks = process_audio(input_path)
    print(chunks)