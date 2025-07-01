# AlanGPT – A RAG Chatbot about my Grandfather's Life

This is a **Retrieval-Augmented Generation (RAG)** chatbot that allows you to talk with the life story written by my grandfather, Alan Gough.
The project was built with:

- **LangChain** for orchestration
- **ChromaDB** for vector storage
- **text-embedding-3-small** for embeddings
- **OpenAI GPT-4.o mini** for response generation
- **Streamlit** for the chat interface

The goal: Let users interactively explore and ask questions about my grandfather’s life.

---

## How It Works

1. **Documents**: The chatbot indexes my grandfather’s life story (initial text was in PDF format).
2. **Embeddings**: It uses OpenAI's `text-embedding-3-small` model to embed the story chunks.
3. **Vector Store**: Embeddings are stored in ChromaDB for fast retrieval.
4. **Chat**: A user types a question. LangChain retrieves relevant chunks from Chroma, and GPT-4.o mini provides an answer given context from the story.
5. **Streamlit**: Displays the chat conversation.

---

## Project Structure

grandpa-gpt/
├── Grandad_Life_Story.txt # File to chunk and generate embeddings from
├── Grandad_Life_Story.pdf # File for users to download
├── app.py # Streamlit chatbot app with all RAG querying logic
├── process.py # Script to load & embed story
├── requirements.txt
└── README.md