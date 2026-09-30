import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


# Create the Groq LLM
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


# Convert retrieved documents into plain text
def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


# Build the RAG chain for the video
def build_rag_chain(transcript: str):

    # Create vector database from the video transcript
    vector_store = build_vector_store(transcript)

    # Retrieve the most relevant transcript chunks
    # k=4 means 4 relevant chunks will be provided to the LLM
    retriever = get_retriever(
        vector_store,
        k=4
    )

    # Create the Groq LLM
    llm = get_llm()

    # General-purpose video assistant prompt
    #
    # The user is not limited to asking questions.
    # They can ask the assistant to summarize, explain,
    # create quizzes, translate, shorten, etc.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an intelligent video assistant.

Use ONLY the provided video transcript context to answer
or perform the user's request.

The user may ask you to:

- Answer questions about the video
- Explain a topic simply
- Summarize information
- Make an answer shorter
- Make an answer more detailed
- Create a quiz
- Create multiple-choice questions (MCQs)
- Create study notes
- List important points
- Extract key information
- Translate information into ANY language
- Rewrite information in a different style
- Compare ideas found in the video

Follow the user's requested format and language.

FORMATTING RULES (always apply these unless the user asks for something
different, like a single short sentence):
- Respond in clean, well-structured Markdown — similar in style to a
  well-organized ChatGPT study note.
- For longer answers, break the content into sections using Markdown
  headings (e.g. "### Topic Name") instead of one long paragraph.
- Use bullet points ("- ") or numbered lists ("1.", "2.") for any list
  of items, steps, or examples.
- **Bold** key terms, names, and labels so they stand out.
- When comparing items or presenting structured data (e.g. a feature with
  its description and an example), use a Markdown table with a header row.
- For quizzes, number each question and give a clear "Answer Key" section
  at the end. For MCQs, list each option on its own line (A, B, C, D).
- Never use raw HTML tags (no <br>, <p>, <div>, etc.) — Markdown only.

IMPORTANT:
Do not invent information that is not present in the
provided video transcript context.

If the requested information cannot be found in the
provided context, say:

"I could not find this information in the video transcript."

Be clear and helpful, and format every answer according to the rules above.

Context from video transcript:
{context}
"""
        ),
        (
            "human",
            "{question}"
        ),
    ])

    # Full LCEL RAG pipeline
    #
    # 1. User request is received
    # 2. Retriever finds relevant transcript chunks
    # 3. Documents are converted into text
    # 4. Context + user request are sent to the LLM
    # 5. LLM generates the final answer
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Send any user request to the video assistant
#
# The function name remains ask_question so your existing
# main.py code does not break.
def ask_question(rag_chain, question: str) -> str:

    print("User Request:", question)

    # Send the request through the RAG chain
    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer