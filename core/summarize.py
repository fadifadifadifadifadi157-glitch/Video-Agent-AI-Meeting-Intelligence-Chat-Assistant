print("SUMMARY 1 - Starting summarize.py")

from groq import Groq
print("SUMMARY 2 - Groq imported")

from langchain_text_splitters import RecursiveCharacterTextSplitter
print("SUMMARY 3 - Text splitter imported")

import os
print("SUMMARY 4 - os imported")


# Create Groq client
def get_llm():
    return Groq(
        api_key=os.getenv("GROQ_API_KEY")
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
  "## <Short Video Topic>".
- Organize the content into a few clearly numbered sections using
  Markdown headings, e.g. "### 1. <Section Name>", "### 2. <Section Name>".
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


# Translate the generated summary into ANY language
def translate_summary(summary: str, language: str) -> str:

    prompt = f"""
Translate the following video summary into {language}.

Rules:
- Keep the original meaning unchanged.
- Do not add new information.
- Keep the important points.
- Preserve the exact Markdown structure (headings, bold labels, bullet
  points, and any tables) — translate only the text inside it.
- Use natural and easy-to-understand {language}.
- Use plain Markdown only — never use raw HTML tags (no <br>, <p>, etc.).
- Return only the translated summary.

Summary:
{summary}
"""

    return ask_groq(prompt)


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
