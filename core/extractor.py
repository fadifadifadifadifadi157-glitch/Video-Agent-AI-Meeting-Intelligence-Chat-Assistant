from groq import Groq
import os


def get_llm():
    return Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )


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
        temperature=0.2
    )

    return response.choices[0].message.content


def extract_action_items(transcript: str) -> str:

    prompt = f"""
You are an expert meeting analyst.

From the meeting transcript, extract all action items.

For each action item provide:

- Task description
- Owner
- Deadline

If the deadline is not mentioned, write:
Not specified

Format as a numbered list.

If there are no action items, say:
No action items found.

Transcript:
{transcript}
"""

    return ask_groq(prompt)


def extract_key_decisions(transcript: str) -> str:

    prompt = f"""
You are an expert meeting analyst.

From the meeting transcript, extract all key decisions made.

Format as a numbered list.

If there are no key decisions, say:
No key decisions found.

Transcript:
{transcript}
"""

    return ask_groq(prompt)


def extract_questions(transcript: str) -> str:

    prompt = f"""
You are an expert meeting analyst.

From the meeting transcript, extract all unresolved questions
or topics needing follow-up.

Format as a numbered list.

If there are no open questions, say:
No open questions found.

Transcript:
{transcript}
"""

    return ask_groq(prompt)