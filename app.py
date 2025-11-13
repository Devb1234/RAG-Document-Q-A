import streamlit as st
import os
import ssl

# ---- FIX SSL Certificate Errors (Corporate Networks, Windows) ----
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# ------------------ IMPORTS --------------------
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings     # ✅ UPDATED
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# ------------------ API KEYS --------------------
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# ------------------ LLM --------------------
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# ------------------ PROMPT TEMPLATE --------------------
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Provide the most accurate response.

<context>
{context}
</context>

Question: {input}
""")

# ------------------ Create Vector Embeddings --------------------
def create_vector_embedding():
    if "vectors" not in st.session_state:

        # ✅ Updated embedding class + explicit model
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )

# ------------------ UI --------------------
st.title("RAG Document Q&A With Groq and Llama3")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector Database is ready!")

import time

# ------------------ Query Handling --------------------
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embedding' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f}s")

        st.subheader("Answer")
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("---")
