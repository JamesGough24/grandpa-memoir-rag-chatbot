import streamlit as st
import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

# Set Streamlit app title and config page
st.set_page_config(page_title="AlanGPT", page_icon="📚", initial_sidebar_state='expanded', layout='wide')

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=4,
        return_messages=True
    )

with st.sidebar:
    st.header("About this Chatbot  📖")
    st.write("This chatbot can answer questions about Alan Gough's life using")
    st.write("• **Vector Search**: Finds relevant passages")
    st.write("• **Conversation Memory**: Remembers context")
    st.write("• **AI Understanding**: Provides natural responses")

    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()

    st.subheader("Example Prompts  💭")
    example_prompts = [
        "Tell me about Alan's childhood",
        "What did Alan do for work?",
        "Did Alan have a brother?",
        "What car did Alan once buy?",
        "Recount a funny experience from Alan's life"
    ]
    for prompt in example_prompts:
        if st.button(prompt, key=f"example_{prompt}"):
            st.session_state.pending_prompt = prompt
            st.rerun()

    st.subheader("Download Alan's Life Story  👇")
    with open("Grandad_Life_Story.pdf", "rb") as pdf_file:
        st.download_button(
            label="Click Here",
            data=pdf_file,
            file_name='Alan_Gough_Life_Story.pdf',
            mime="application/pdf"
        )

@st.cache_resource
def init_components():
    # Initialize embeddings (same model used for chunking)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Load existing Chroma vector store
    persist_directory = 'db'
    vectorstore = Chroma(
        collection_name='memoir_text',
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Intialize LLM
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=st.secrets['OPENAI_API_KEY'],
        temperature=0.3,
        streaming=True,
        max_tokens=512
    )

    return embeddings, vectorstore, llm

def expand_query_with_context(query, memory, k=4):
    # Get recent conversation history for context
    chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []

    if not chat_history:
        return query
    
    vague_indicators = ['he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'it']
    has_vague_reference = any(indicator in query.lower() for indicator in vague_indicators)

    if not has_vague_reference:
        return query

    recent_messages = []
    for msg in chat_history[-4:]:
        if isinstance(msg, HumanMessage):
            recent_messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            recent_messages.append(f"Assistant: {msg.content}")
        
    context_text = '\n'.join(recent_messages)

    return f"""Considering the previous conversation context: '{context_text}', {query}"""


def perform_similarity_search(vectorstore, query, memory, k=6):
    # Expand vague queries with conversation context
    print(f"This is the ORIGINAL prompt: {query}")
    expanded_query = expand_query_with_context(query, memory)
    print(f"This is the ADJUSTED prompt: {expanded_query}")

    # Get relevant chunks for the user query
    relevant_chunks_docs = vectorstore.similarity_search(query, k=6)
    relevant_chunks = [doc.page_content for doc in relevant_chunks_docs]
    print(f"These were the relevant chunks for the ORIGINAL prompt: {relevant_chunks}")

    if expanded_query != query:
        expanded_relevant_chunks_docs = vectorstore.similarity_search(query, k=4)
        expanded_relevant_chunks = [doc.page_content for doc in expanded_relevant_chunks_docs]
        print(f"These were the relevant chunks for the ADJUSTED prompt: {expanded_relevant_chunks}")
        relevant_chunks = list(set(relevant_chunks + expanded_relevant_chunks))
        print(f"These are the final relevant chunks with everything combined: {expanded_relevant_chunks}")
        query = expanded_query
    
    # Extract text content
    context_chunks = []
    for chunk in relevant_chunks:
        context_chunks.append({
            'content': chunk
        })
    
    return query, context_chunks


def create_context_aware_prompt(query, context_chunks):
    # Format context from memoir
    context_text = ""
    if context_chunks:
        context_text = "Relevant information from grandfather's memoir:\n"
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"[Context {i}]: {chunk}\n\n"

    # Create the full prompt
    prompt = f"""You are a helpful, informative assistant answering questions about someone's grandad Alan Gough based on his memoir.
Only use the provided context and conversation history to give accurate, personal responses.

{context_text}

Instructions:
- Alan's memoir is written from his perspective. When it says "I", "me", "my", it means Alan!
- Answer from YOUR perspective (like 3rd-person style)
- Answer based primarily on the memoir context provided
- Use the conversation history to understand context (like "he" or "it", or if they ask a follow-up question)
- If the question refers to someone/something mentioned earlier in conversation, maintain that context
- If you cannot find relevant information in the memoir, say so politely
- Key details: Sue is Alan's wife. Les is Alan's brother. Trevor and Lindsay and Alan's kids.

The user just asked: {query}. Your job is to answer them."""
    print(f"This is the CONTEXT AWARE PROMPT: \n{prompt}")

    return prompt

def query_llm_with_context(llm, prompt):
    try:
        response = llm.predict(prompt)
        return response
    except Exception as e:
        st.error(f"Error querying LLM: {str(e)}")
        return "I'm sorry, I encountered an error while processing your question."

def main():
    st.title("Welcome to AlanGPT 📚")

    # Init components
    try:
        embeddings, vectorstore, llm = init_components()
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        st.stop()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.write(message['content'])

    user_input = st.chat_input("Ask a question about Alan Gough's life!")
    prompt = None
    
    # Check whether the user has clicked one of the sidebar prompts
    if 'pending_prompt' in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt
    else:
        prompt = user_input
    
    # Chat input
    if prompt:
         # Display user message in chat container
        with st.chat_message('user'):
            st.write(prompt)

        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })

        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                # Perform similarity search with context-aware query
                context_chunks, expanded_query = perform_similarity_search(
                    vectorstore, prompt, st.session_state.memory, k=4
                )

                # Create context-aware prompt
                context_prompt = create_context_aware_prompt(
                    context_chunks, expanded_query
                )

                # Query LLM
                response = query_llm_with_context(llm, context_prompt)

                # Display response
                st.write(response)

                st.session_state.memory.chat_memory.add_user_message(prompt)
                st.session_state.memory.chat_memory.add_ai_message(response)

                st.session_state.chat_history.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()