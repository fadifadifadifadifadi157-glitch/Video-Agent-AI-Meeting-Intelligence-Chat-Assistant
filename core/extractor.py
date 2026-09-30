from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(
        api_key=api_key
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
        temperature=0.3,
        max_tokens=6000 
    )

    return response.choices[0].message.content


def _prepare_context(transcript: str, summary: str = "") -> str:
    """Prepare context combining transcript and summary, handling long transcripts safely."""
    clean_transcript = transcript.strip() if transcript else ""
    clean_summary = summary.strip() if summary else ""

    # If transcript is very long (> 25000 chars), use summary + key transcript sections
    if len(clean_transcript) > 25000:
        if clean_summary:
            return (
                f"Summary of the video:\n{clean_summary}\n\n"
                f"Key Transcript (Opening & Core Content):\n{clean_transcript[:12000]}\n\n"
                f"[... middle section summarized above ...]\n\n"
                f"Key Transcript (Conclusion & Next Steps):\n{clean_transcript[-12000:]}"
            )
        else:
            return (
                f"Transcript (Opening):\n{clean_transcript[:14000]}\n\n"
                f"[... middle section ...]\n\n"
                f"Transcript (Conclusion):\n{clean_transcript[-14000:]}"
            )

    if clean_summary:
        return f"Video Summary for context:\n{clean_summary}\n\nFull Transcript:\n{clean_transcript}"
    return f"Transcript:\n{clean_transcript}"


def extract_action_items(transcript: str, summary: str = "") -> str:
    """Extract action items, next steps, instructions, and practical recommendations
    from any video (meetings, tutorials, lectures, tech talks, product reviews)."""
    context = _prepare_context(transcript, summary)

    prompt = f"""
You are an expert video content and meeting analyst.

Analyze the video content and extract all Action Items, Next Steps, and Practical Recommendations.

GUIDELINES:
- For meetings or collaborative discussions: Extract all tasks, assigned owners, deadlines, and agreed next steps.
- For tutorials, technical guides, how-to videos, and lectures: Extract all concrete action steps, setup/installation tasks, code or config changes, practical exercises, recommended practices to implement, tools to try, and common pitfalls to avoid.
- For product reviews, talks, podcasts, or general videos: Extract concrete recommendations, steps to take, decisions to execute, and suggested follow-ups.

FORMATTING:
Format as a clean, highly structured Markdown list:
1. **[Clear Action / Task Title]**
   - **Details / Steps**: Clear explanation of what needs to be done.
   - **Actor / Target**: Who should do it (e.g. Viewer, Developer, Specific person/role, Team).
   - **Timing / Condition**: When or condition to do it (e.g. Immediate next step, Before deploying, If applicable, Deadline if specified).

Extract at least 3 to 8 meaningful, concrete action items whenever actionable content is present.
Never return "No action items found" if there are instructions, steps, advice, recommendations, or best practices in the video.

Content:
{context}
"""

    return ask_groq(prompt)


def extract_key_decisions(transcript: str, summary: str = "") -> str:
    """Extract key decisions, architectural choices, core takeaways, and strategic conclusions
    from any video (meetings, tutorials, lectures, tech talks, product reviews)."""
    context = _prepare_context(transcript, summary)

    prompt = f"""
You are an expert video content and meeting analyst.

Analyze the video content and extract all Key Decisions, Architectural Choices, Core Takeaways, and Strategic Conclusions.

GUIDELINES:
- For meetings or collaborative discussions: Extract agreed resolutions, chosen paths, approvals, policy decisions, and consensus reached.
- For tutorials, tech talks, and engineering videos: Extract key architectural decisions, tool/technology choices (and why one was chosen over alternatives), design patterns chosen, trade-offs accepted, and best practice decisions.
- For reviews, comparisons, and analysis videos: Extract final verdicts, comparative judgments, pros/cons conclusions, and recommended choices.
- For lectures and educational talks: Extract fundamental principles established, core takeaways, and main conclusions reached.

FORMATTING:
Format as a clean, highly structured Markdown list:
1. **[Core Decision / Selection / Conclusion]**
   - **Context / Choice**: What was chosen, decided, or concluded.
   - **Rationale / Why**: The specific reasons, trade-offs, or benefits mentioned (e.g. performance, simplicity, reliability, cost).
   - **Impact / Takeaway**: How this choice affects the project, workflow, or viewer.

Extract all meaningful decisions and conclusions.
Never return "No key decisions found" if there are design choices, tool selections, rationale, or key conclusions in the video.

Content:
{context}
"""

    return ask_groq(prompt)


def extract_questions(transcript: str, summary: str = "") -> str:
    """Extract unresolved questions, open questions, discussion points, or topics for follow-up."""
    context = _prepare_context(transcript, summary)

    prompt = f"""
You are an expert video content analyst.

Analyze the video content and extract all Unresolved Questions, Open Questions, Discussion Points, or Topics for Follow-up.

GUIDELINES:
- Questions explicitly raised or asked by the speaker(s) or participants.
- Topics flagged for future videos, upcoming features, or roadmap items.
- Important open questions, trade-offs, or decisions that the viewer/team needs to think through.

Format as a clean numbered list in Markdown with a brief explanation for each item.
If no open questions or follow-up topics are mentioned, list 2-3 thoughtful discussion questions viewers or practitioners should consider based on the video.

Content:
{context}
"""

    return ask_groq(prompt)