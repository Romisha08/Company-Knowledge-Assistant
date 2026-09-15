from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

DB_NAME = str(
    Path(__file__).parent.parent / "vector_db"
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

RETRIEVAL_K = 5

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.

You are chatting with a user about Insurellm.

If relevant, use the given context to answer any question.

If you don't know the answer, say so.

Context:

{context}
"""

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_K
    }
)

llm = ChatOpenAI(
    temperature=0,
    model=MODEL,
)

def fetch_context(question: str) -> list[Document]:
    if not isinstance(question, str):
        question = str(question)

    return retriever.invoke(question)


def combined_question(
    question: str,
    history: list[dict] | None = None,
) -> str:
    """
    Combine previous user messages with the current question.
    """

    history = history or []

    previous_questions = []

    for message in history:
        if message.get("role") == "user":
            content = message.get("content", "")

            if isinstance(content, str):
                previous_questions.append(content)
            else:
                previous_questions.append(str(content))

    previous_text = "\n".join(previous_questions)

    if previous_text:
        return previous_text + "\n" + question

    return question


def answer_question(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, list[Document]]:
    """
    Answer the given question using RAG.
    Return the answer and retrieved context documents.
    """

    history = history or []

    combined = combined_question(
        question,
        history,
    )

    documents = fetch_context(combined)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        SystemMessage(content=system_prompt)
    ]

    messages.extend(
        convert_to_messages(history)
    )

    messages.append(
        HumanMessage(content=question)
    )

    response = llm.invoke(messages)

    return response.content, documents