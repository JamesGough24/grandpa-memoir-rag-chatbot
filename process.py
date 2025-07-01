import streamlit as st
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

MEMOIR_FILE = 'Grandad_Life_Story.txt'
CHROMA_DB_DIR = 'db'
COLLECTION_NAME = 'memoir_text'

def load_and_chunk(memoir_file):
    # Read in .txt file of life story
    with open(memoir_file, "r", encoding="utf-8") as f:
        memoir_text = f.read()

    # Convert text to LangChain Document object
    memoir_doc = Document(page_content=memoir_text, metadata={})

    # Split life story into chunks based on factors like number of characters and new lines
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents([memoir_doc])

def create_vectorstore(chunks):
    # Set up embedding function
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Set up ChromaDB vectorstore
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    # Push chunk data to vectorstore
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)

def set_up_db():
    chunks = load_and_chunk(MEMOIR_FILE)
    create_vectorstore(chunks)