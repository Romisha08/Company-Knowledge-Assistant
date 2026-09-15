import gradio as gr
from dotenv import load_dotenv

from implementation.answer import answer_question


load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"

    for doc in context:
        source = doc.metadata.get("source", "Unknown source")

        result += (
            f"<span style='color: #ff7800;'>"
            f"Source: {source}"
            f"</span>\n\n"
        )
        result += doc.page_content + "\n\n"

    return result


def add_user_message(message, history):
    """
    Add the user's message to the conversation.
    """

    history = history or []

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return "", history


def chat(history):
    """
    Generate an answer for the latest user message.
    """

    if not history:
        return history, "*No context retrieved yet*"

    # The latest message must be the user's message
    last_message = history[-1]["content"]

    if not isinstance(last_message, str):
        last_message = str(last_message)

    # Previous messages are passed to the RAG pipeline
    prior_history = history[:-1]

    answer, context = answer_question(
        question=last_message,
        history=prior_history,
    )

    history.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return history, format_context(context)


def main():
    theme = gr.themes.Soft(
        font=["Inter", "system-ui", "sans-serif"]
    )

    with gr.Blocks(
        title="Insurellm Expert Assistant"
    ) as ui:

        gr.Markdown(
            "# 🏢 Insurellm Expert Assistant\n"
            "Ask me anything about Insurellm!"
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=600,
                )

                message = gr.Textbox(
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            add_user_message,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
        ).then(
            chat,
            inputs=[chatbot],
            outputs=[chatbot, context_markdown],
        )

    ui.launch(
        inbrowser=True,
        theme=theme,
    )


if __name__ == "__main__":
    main()