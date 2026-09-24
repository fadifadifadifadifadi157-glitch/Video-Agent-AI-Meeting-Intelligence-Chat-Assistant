from dotenv import load_dotenv

load_dotenv()

from utlis.audio_processor import process_audio
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)


input_path = input("Enter YouTube URL or local file path: ")

chunks = process_audio(input_path)

print("DEBUG 1: Starting transcription...", flush=True)

transcript = transcribe_all(chunks, translate=True)

print("DEBUG 2: Transcription function returned", flush=True)

print("\n" + "\n" * 5)

print("Transcript")
print("=" * 60)

print(
    transcript[:500] + "..."
    if len(transcript) > 500
    else transcript
)

print("=" * 60, flush=True)


print("DEBUG 3: Starting title generation...", flush=True)

title = generate_title(transcript)

print("DEBUG 4: Title generated", flush=True)


print("DEBUG 5: Starting summary generation...", flush=True)

summary = summarize(transcript)

print("DEBUG 6: Summary generated", flush=True)


print("\n" + "=" * 60)

print(f"📝 TITLE: {title}")

print("=" * 60)

print("\n📋 SUMMARY")

print("=" * 60)

print(summary)


print("\nDEBUG 7: Extracting action items...", flush=True)

action_items = extract_action_items(transcript)

print("DEBUG 8: Extracting decisions...", flush=True)

decisions = extract_key_decisions(transcript)

print("DEBUG 9: Extracting questions...", flush=True)

questions = extract_questions(transcript)


print("\n" + "=" * 60)

print("ACTION ITEMS")

print("=" * 60)

print(action_items)


print("\n" + "=" * 60)

print("KEY DECISIONS")

print("=" * 60)

print(decisions)


print("\n" + "=" * 60)

print("OPEN QUESTIONS")

print("=" * 60)

print(questions)