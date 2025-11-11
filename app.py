import streamlit as st
import os
from dotenv import load_dotenv

# ==== Load environment ====
load_dotenv()

# Remove broken SSL cert env
for var in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]:
    if var in os.environ:
        del os.environ[var]

# ==== LangChain Imports ====
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from pypdf import PdfReader


# ==== Streamlit UI ====
st.set_page_config(page_title="RAG Q&A", page_icon="📄")
st.header("📄 RAG Document Q&A System")


# ==== API Key Check ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("No GROQ_API_KEY found!")
    st.stop()


# ==== LLM ====
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# ==== Embeddings ====
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==== File Upload ====
uploaded_files = st.file_uploader(
    "Upload PDF / Text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

query = st.text_input("Enter your question:")


# ==== Load & Index Documents ====
def process_docs(uploaded):
    text = ""
    for f in uploaded:
        if f.name.endswith(".txt"):
            text += f.read().decode("utf-8")
        elif f.name.endswith(".pdf"):
            pdf = PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

    if not text.strip():
        st.error("❌ No readable text found in uploaded files")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=c) for c in splitter.split_text(text)]

    vs = FAISS.from_documents(docs, embedding_model)
    return vs


vectorstore = None
if uploaded_files:
    vectorstore = process_docs(uploaded_files)
    if vectorstore:
        st.success("✅ Documents indexed successfully!")


# ==== RAG ====
if query:
    if not vectorstore:
        st.warning("Upload documents first!")
    else:
        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Use the following documents to answer the question.
        If answer is not found, say 'I don't know'.

        Documents:
        {context}

        Question: {input}
        """)

        combine_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        rag_chain = create_retrieval_chain(
            retriever,
            combine_chain
        )

        response = rag_chain.invoke({"input": query})

        st.write("### ✅ Answer")
        st.write(response.get("answer", "No answer found"))
