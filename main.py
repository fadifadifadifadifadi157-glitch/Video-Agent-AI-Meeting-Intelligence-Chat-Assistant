print("1 - Starting main.py")

from dotenv import load_dotenv
print("2 - dotenv imported")


# Import audio processing functions
from utlis.audio_processor import process_audio
print("3 - audio_processor imported")


# Import Whisper transcription function
from core.transcriber import transcribe_all
print("4 - transcriber imported")


# Import summary, title, translation and shortening functions
from core.summarize import (
    summarize,
    generate_title,
    translate_summary,
    shorten_summary
)
print("5 - summarize imported")


# Import information extraction functions
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
print("6 - extractor imported")


# Import RAG / Video Assistant functions
from core.rag_engine import build_rag_chain, ask_question
print("7 - rag_engine imported")


# Load variables from .env file
load_dotenv()
print("8 - dotenv loaded")


# ---------------------------------------------------------
# MAIN VIDEO PROCESSING PIPELINE
# ---------------------------------------------------------

def run_pipeline(source: str) -> dict:

    print("Starting AI Video Assistant")

    # Download/process the video audio
    chunks = process_audio(source)

    # Convert the audio into English text using Whisper
    transcript = transcribe_all(chunks)

    print(
        f"Raw transcription (first 300 characters): "
        f"{transcript[:300]}"
    )

    # Generate a title using the Groq LLM
    title = generate_title(transcript)

    # Generate a summary using the Groq LLM
    summary = summarize(transcript)

    # Extract action items from the transcript
    action_items = extract_action_items(transcript)

    # Extract important decisions from the transcript
    decisions = extract_key_decisions(transcript)

    # Extract questions/open questions from the transcript
    questions = extract_questions(transcript)

    # Build the RAG chain for chatting with the video
    rag_chain = build_rag_chain(transcript)

    # Return all generated information
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ---------------------------------------------------------
# PROGRAM START
# ---------------------------------------------------------

if __name__ == "__main__":

    # Ask the user for a YouTube URL or local video/audio file
    input_path = input(
        "Enter YouTube URL or local file path: "
    ).strip()

    # Run the complete video processing pipeline
    result = run_pipeline(input_path)

    # -----------------------------------------------------
    # DISPLAY VIDEO INFORMATION
    # -----------------------------------------------------

    print("\n" + "=" * 60)

    print(f"📌 Title: {result['title']}")

    print("\n📋 Summary:")
    print("=" * 60)
    print(result["summary"])

    print("\n✅ Action Items:")
    print("=" * 60)
    print(result["action_items"])

    print("\n🔑 Key Decisions:")
    print("=" * 60)
    print(result["key_decisions"])

    print("\n❓ Open Questions:")
    print("=" * 60)
    print(result["open_questions"])

    print("=" * 60)

    # -----------------------------------------------------
    # CHAT WITH THE VIDEO
    # -----------------------------------------------------

    print(
        "\n💬 Chat with your video"
        "\nYou can ask the assistant to:"
        "\n- Answer questions"
        "\n- Explain topics"
        "\n- Make quizzes"
        "\n- Create MCQs"
        "\n- Give important points"
        "\n- Make answers shorter"
        "\n- Translate information"
        "\n\nSpecial summary commands:"
        "\n- translate summary to <language>"
        "\n- shorten summary"
        "\n\nType 'exit' to quit.\n"
    )

    # Get the RAG chain created from the video transcript
    rag_chain = result["rag_chain"]

    # Get the generated summary
    summary = result["summary"]

    # Start continuous chat
    while True:

        # Get user's request
        question = input("You: ").strip()

        # Exit the program
        if question.lower() in ["exit", "quit", "q"]:

            print("👋 Goodbye!")
            break

        # Ignore empty input
        if not question:
            continue

        # -------------------------------------------------
        # TRANSLATE THE GENERATED SUMMARY
        # -------------------------------------------------

        # Example:
        # translate summary to French
        # translate summary to Arabic
        # translate summary to Urdu
        #
        # The language is taken from whatever the user writes.

        if question.lower().startswith("translate summary to "):

            # Extract the language from the user's request
            language = question[len("translate summary to "):].strip()

            if not language:
                print(
                    "\n🤖 Assistant: Please specify a language.\n"
                )
                continue

            print(
                f"\n🌐 Translating summary into {language}...\n"
            )

            translated_summary = translate_summary(
                summary,
                language
            )

            print(
                f"\n🤖 Assistant ({language}): "
                f"{translated_summary}\n"
            )

            continue

        # -------------------------------------------------
        # SHORTEN THE GENERATED SUMMARY
        # -------------------------------------------------

        if question.lower() == "shorten summary":

            print("\n🤖 Shortening summary...\n")

            short_summary = shorten_summary(summary)

            print(
                f"\n🤖 Assistant:\n{short_summary}\n"
            )

            continue

        # -------------------------------------------------
        # GENERAL VIDEO ASSISTANT
        # -------------------------------------------------

        # Any other request is sent to the RAG assistant.
        #
        # Examples:
        #
        # "Make a quiz from this video"
        # "Give me 10 MCQs"
        # "Explain this topic simply"
        # "What are the important points?"
        # "Make this explanation shorter"
        # "What is the main idea?"
        #
        # The RAG system retrieves relevant information
        # from the video transcript before sending it to Groq.

        answer = ask_question(
            rag_chain,
            question
        )

        print(
            f"\n🤖 Assistant: {answer}\n"
        )