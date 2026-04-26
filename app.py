import chainlit as cl
from backend import llm, load_document, retrieve, build_message


@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome to **Text-Image RAG**!\n\nUpload a document (`.pdf`, `.docx`, or `.txt`) to get started."
    ).send()

    files = await cl.AskFileMessage(
        content="Upload your document:",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain"
        ],
        max_size_mb=50,
    ).send()

    file = files[0]

    processing_msg = cl.Message(content=f"Processing **{file.name}** — this may take a moment...")
    await processing_msg.send()

    try:
        vector_store, image_data_stores, n_chunks, n_images = load_document(file.path)

        cl.user_session.set("vector_store", vector_store)
        cl.user_session.set("image_data_stores", image_data_stores)

        await cl.Message(
            content=(
                f"**{file.name}** is ready!\n\n"
                f"- Text chunks: **{n_chunks}**\n"
                f"- Images: **{n_images}**\n\n"
                f"Ask me anything about your document."
            )
        ).send()

    except Exception as e:
        await cl.Message(content=f"Error processing document: `{str(e)}`").send()


@cl.on_message
async def main(message: cl.Message):
    vector_store = cl.user_session.get("vector_store")
    image_data_stores = cl.user_session.get("image_data_stores")

    if not vector_store:
        await cl.Message(content="Please upload a document first.").send()
        return

    retrieved_docs = retrieve(message.content, vector_store, k=5)
    human_message = build_message(message.content, retrieved_docs, image_data_stores)

    # Stream response token by token
    response_msg = cl.Message(content="")
    await response_msg.send()

    async for chunk in llm.astream([human_message]):
        await response_msg.stream_token(chunk.content)

    # Append sources
    sources = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "?")
        if doc.metadata.get("type") == "text":
            preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            sources.append(f"- Text (page {page}): *{preview}*")
        else:
            sources.append(f"- Image (page {page})")

    response_msg.content += "\n\n---\n**Sources retrieved:**\n" + "\n".join(sources)
    await response_msg.update()
