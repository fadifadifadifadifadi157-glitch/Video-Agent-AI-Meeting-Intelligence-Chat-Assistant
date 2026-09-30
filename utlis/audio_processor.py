import yt_dlp
import os
import shutil
import tempfile
import time
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Optional: only needed on machines where ffmpeg isn't on PATH (e.g. some
# local Windows installs). Set FFMPEG_LOCATION in your .env to override.
FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION")


def download_youtube_audio(url: str) -> str:
    """Download audio from a YouTube URL.
    Uses temp directory for FFmpeg processing to avoid Windows WinError 32 file-locking
    conflicts (common in OneDrive and cloud-synced folders), and safely moves the result."""
    temp_dir = tempfile.gettempdir()

    # Pre-check: if a matching wav file already exists in DOWNLOAD_DIR, reuse it!
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl_info:
            info = ydl_info.extract_info(url, download=False)
            if info:
                title = info.get("title", "")
                video_id = info.get("id", "")
                for fname in os.listdir(DOWNLOAD_DIR):
                    if fname.endswith(".wav") and not fname.endswith("_converted.wav") and "_chunk_" not in fname:
                        # Match by ID or matching title
                        if (video_id and video_id in fname) or (title and title[:25].lower() in fname.lower()) or (title and fname[:25].lower() in title.lower()):
                            candidate = os.path.join(DOWNLOAD_DIR, fname)
                            if os.path.exists(candidate) and os.path.getsize(candidate) > 1000:
                                print(f"Reusing already downloaded audio: {candidate}")
                                return candidate
    except Exception as e:
        print(f"Pre-download cache check notice: {e}")

    # Download to %TEMP% first (outside OneDrive sync engine to prevent WinError 32)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "%(title).100s_%(id)s.%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
        "overwrites": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": True,
    }

    if FFMPEG_LOCATION:
        opts["ffmpeg_location"] = FFMPEG_LOCATION

    temp_wav = None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            temp_file = ydl.prepare_filename(info)
            temp_wav = os.path.splitext(temp_file)[0] + ".wav"
    except Exception as e:
        # If WinError 32 occurred during postprocessing/rename, check if wav was created
        for f in os.listdir(temp_dir):
            if f.endswith(".wav") and os.path.getmtime(os.path.join(temp_dir, f)) > time.time() - 180:
                temp_wav = os.path.join(temp_dir, f)
                break
        else:
            # Check in DOWNLOAD_DIR in case it completed there
            for f in os.listdir(DOWNLOAD_DIR):
                if f.endswith(".wav") and os.path.getmtime(os.path.join(DOWNLOAD_DIR, f)) > time.time() - 180:
                    return os.path.join(DOWNLOAD_DIR, f)
            raise e

    if not temp_wav or not os.path.exists(temp_wav):
        # Fallback search for recently modified .wav in temp_dir
        for f in os.listdir(temp_dir):
            if f.endswith(".wav") and os.path.getmtime(os.path.join(temp_dir, f)) > time.time() - 180:
                temp_wav = os.path.join(temp_dir, f)
                break

    if not temp_wav or not os.path.exists(temp_wav):
        raise FileNotFoundError("Could not find the extracted audio WAV file.")

    # Safely copy into DOWNLOAD_DIR with retry logic for Windows file locks
    dest_filename = os.path.basename(temp_wav)
    dest_path = os.path.join(DOWNLOAD_DIR, dest_filename)

    for attempt in range(5):
        try:
            shutil.copy2(temp_wav, dest_path)
            try:
                os.remove(temp_wav)
            except Exception:
                pass
            return dest_path
        except (PermissionError, OSError):
            time.sleep(0.5)

    # If copying to DOWNLOAD_DIR was locked, reuse dest_path if valid, or return temp_wav
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return dest_path

    return temp_wav


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    # Reuse if already converted and non-empty
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"Reusing already converted audio: {output_path}")
        return output_path

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split large audio file into smaller chunks for transcription."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"

        # Reuse chunk if already exported
        if not (os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 1000):
            chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_audio(input_path: str) -> list:
    """Detect YouTube URL or local file and process automatically."""
    if input_path.startswith(("http://", "https://")):
        print("Detected Youtube URL. Downloading audio...")
        wav_file = download_youtube_audio(input_path)
        wav_file = convert_to_wav(wav_file)
    else:
        print("Detected local file. Converting to WAV...")
        wav_file = convert_to_wav(input_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_file)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks


if __name__ == "__main__":
    input_path = input("Enter YouTube URL or local file path: ")
    chunks = process_audio(input_path)
    print(chunks)