from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prommpts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import streamlit as st
import time
import os
from dotenv import load_dotenv
load_dotenv()

## About Groq: https://console.groq.com/docs/quickstart

groq_api_key = os.environ['GROQ_API_KEY']

if "vector_db" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loaders = WebBaseLoader('')
    st.session_state.web_document = st.session_state.loaders.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.chunk_document = st.session_state.text_splitter.split_documents(st.session_state.web_document)
    st.session_state.vector_db = FAISS.from_documents(st.session_state.chunk_document, st.session_state.embeddings)

st.title("Chatbot using Groq API")

llm_model = ChatGroq(groq_api_key = groq_api_key, model = "Gemma-7b-It")

prompt = ChatPromptTemplate.from_template(
    """
    Answer
    {context}
    Question: {input}
    """)

document_chain = create_stuff_documents_chain(llm_model, prompt)

retriever = st.session_state.vector_db.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response Time: ", time.process_time() - start_time)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
