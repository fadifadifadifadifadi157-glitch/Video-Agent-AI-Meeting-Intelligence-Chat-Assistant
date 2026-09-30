print("SUMMARY 1 - Starting summarize.py")

from groq import Groq
print("SUMMARY 2 - Groq imported")

from langchain_text_splitters import RecursiveCharacterTextSplitter
print("SUMMARY 3 - Text splitter imported")

import os
import re
from dotenv import load_dotenv

load_dotenv()
print("SUMMARY 4 - os, re, and dotenv loaded")


# Create Groq client
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(
        api_key=api_key
    )


# Split long transcript into smaller chunks
def split_transcript(transcript: str) -> list:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitter.split_text(transcript)


# Send a prompt to Groq and get the response
def ask_groq(prompt: str) -> str:

    client = get_llm()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=6000 
    )

    return response.choices[0].message.content


# Generate a summary from the complete transcript
def summarize(transcript: str) -> str:

    chunks = split_transcript(transcript)

    chunk_summaries = []

    for chunk in chunks:

        prompt = f"""
Summarize this portion of a video transcript concisely.

Focus on the important information, ideas, facts, and explanations.

Transcript:
{chunk}
"""

        chunk_summaries.append(ask_groq(prompt))

    # Combine all partial summaries
    combined = "\n\n".join(chunk_summaries)

    # Create the final summary from the partial summaries
    final_prompt = f"""
You are an expert video summarizer who writes clean, well-structured
Markdown notes — similar in style to a well-organized study guide.

Combine the following partial summaries into one polished, easy-to-read
video summary written in Markdown, following these formatting rules:

- Start with a short title line formatted as a Markdown heading, e.g.
  "## Main Video Topic".
- Organize the content into clearly numbered sections using
  Markdown headings, e.g. "### 1. Key Concepts", "### 2. Core Methodologies".
- Under each heading, use concise bullet points ("- ") and **bold** the
  key terms, names, or labels (the way a ChatGPT-style study note would).
- If any part of the content naturally compares items or lists structured
  data (e.g. steps with a description and an example), format it as a
  Markdown table with a header row.
- End with a short "### Bottom line" section giving the key takeaway in
  1-2 sentences.
- Do not add information that is not present in the summaries.
- Use plain Markdown only — never use raw HTML tags (no <br>, <p>, etc.).

Partial summaries:
{combined}
"""

    return ask_groq(final_prompt)


# Generate a short title for the video
def generate_title(transcript: str) -> str:

    prompt = f"""
Based on the video transcript below, generate a short
professional video title.

Maximum 8 words.
Return only the title.

Transcript:
{transcript[:2000]}
"""

    return ask_groq(prompt)


# Translate the generated summary into ANY language completely without truncation
def translate_summary(summary: str, language: str) -> str:
    """Translate the complete video summary into the target language.
    Guarantees 100% complete translation by processing in section-aware chunks
    so long summaries are never truncated or summarized halfway."""
    if not summary or not summary.strip():
        return ""

    # Split by markdown headers (#, ##, ###) if present
    sections = re.split(r'(?=(?:^|\n)#{1,3}\s)', summary.strip())
    sections = [s.strip() for s in sections if s.strip()]

    # Helper function to translate a single block with strict completeness rules
    def _translate_block(block: str) -> str:
        prompt = f"""You are a professional expert translator.
Translate the following Markdown content of a video summary into {language}.

CRITICAL RULES:
1. Translate the ENTIRE content accurately and completely from beginning to end.
2. Do NOT summarize, shorten, abbreviate, or omit any sentence, heading, bullet point, or word.
3. Every single point, label, and detail in the source text MUST appear fully translated in the output.
4. Preserve the exact Markdown structure (headings #, ##, ###, bold **labels**, bullet points -, and tables).
5. Use natural, high-quality, fluent {language}.
6. Use plain Markdown only — never use raw HTML tags (no <br>, <p>, etc.).
7. Return ONLY the translated Markdown without any introductory or concluding comments.

Content to translate:
{block}
"""
        return ask_groq(prompt).strip()

    if len(sections) > 1:
        # Group sections into reasonable chunks (up to ~2000 chars each) to keep context together
        chunks = []
        current_chunk = []
        current_len = 0

        for sec in sections:
            if current_len + len(sec) > 2200 and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [sec]
                current_len = len(sec)
            else:
                current_chunk.append(sec)
                current_len += len(sec)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        translated_chunks = [_translate_block(chunk) for chunk in chunks]
        return "\n\n".join(translated_chunks)

    # If no markdown headings, check length
    if len(summary) > 2200:
        paragraphs = summary.strip().split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            if current_len + len(p) > 2000 and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
            else:
                current_chunk.append(p)
                current_len += len(p)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        translated_chunks = [_translate_block(chunk) for chunk in chunks]
        return "\n\n".join(translated_chunks)

    # Short single block
    return _translate_block(summary)


# Make an existing summary shorter
def shorten_summary(summary: str) -> str:

    prompt = f"""
Make the following video summary shorter.

Rules:
- Keep only the most important information.
- Do not remove the main ideas.
- Do not add new information.
- Preserve the Markdown structure (headings, bold labels, bullet points)
  where it still makes sense for a shorter summary.
- Use plain Markdown only — never use raw HTML tags (no <br>, <p>, etc.).
- Return only the shorter summary.

Summary:
{summary}
"""

    return ask_groq(prompt)