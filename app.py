import streamlit as st
import os
import shutil
from ingest import load_local_docs, load_website, ingest_documents
from agent import query_agent

st.set_page_config(page_title="Agentic RAG", page_icon="🤖")

st.title("🤖 Agentic RAG System")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    
    st.header("Data Ingestion")
    
    # Local File Upload (Simulating folder selection by uploading multiple files)
    uploaded_files = st.file_uploader("Upload PDF/TXT files", accept_multiple_files=True, type=["pdf", "txt"])
    
    if st.button("Process Uploaded Files"):
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
        elif uploaded_files:
            # Save uploaded files to a temporary directory
            temp_dir = "temp_docs"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            for uploaded_file in uploaded_files:
                with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Ingesting documents..."):
                docs = load_local_docs(temp_dir)
                ingest_documents(docs, api_key)
                # Cleanup
                shutil.rmtree(temp_dir)
            st.success("Documents ingested successfully!")
        else:
            st.warning("Please upload files first.")

    st.divider()

    # Website URL
    url = st.text_input("Enter Website URL")
    if st.button("Process Website"):
        if not api_key:
            st.error("Please enter your OpenAI API Key.")
        elif url:
            with st.spinner("Ingesting website..."):
                docs = load_website(url)
                ingest_documents(docs, api_key)
            st.success("Website content ingested successfully!")
        else:
            st.warning("Please enter a URL.")

# Chat Interface
st.header("Chat with your Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not api_key:
                st.error("Please enter your OpenAI API Key in the sidebar.")
                response = "I need an API Key to answer."
            else:
                response = query_agent(prompt, api_key)
            st.markdown(response)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
