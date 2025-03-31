import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.graph import StateGraph
from typing import TypedDict, List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pdfplumber  # Alternative to PyMuPDF

# Set USER_AGENT to avoid warnings
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Define the State Schema
class AgentState(TypedDict):
    input: str
    documents: List[str]
    analysis: str
    output: str

# Initialize Streamlit app
st.set_page_config(page_title="Multi-Agent LangGraph", layout="wide")
st.title("🚀 Advanced Multi-Agent LangGraph Streamlit UI")

# Sidebar for API Key and PDF Upload
with st.sidebar:
    st.header("Settings")
    HF_API_KEY = st.text_input("🔑 Enter Hugging Face API Key", type="password")
    uploaded_pdf = st.file_uploader("📂 Upload a PDF", type=["pdf"])
    dark_mode = st.toggle("🌙 Dark Mode")
    
if HF_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY
    st.sidebar.success("Hugging Face API Key Set!")

# PDF File Processing
def extract_text_from_pdf(pdf_file):
    """Extract text using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Session state for persistence
if "docs_split" not in st.session_state:
    st.session_state.docs_split = []
    st.session_state.vectorstore = None

if uploaded_pdf:
    text = extract_text_from_pdf(uploaded_pdf)
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.docs_split = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        st.session_state.vectorstore = FAISS.from_texts(st.session_state.docs_split, embeddings)
        st.sidebar.success(f"Loaded {len(st.session_state.docs_split)} document chunks.")

# Multi-Agent LangGraph Setup
st.subheader("💡 Multi-Agent System")
if HF_API_KEY:
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", huggingfacehub_api_token=HF_API_KEY)

    def agent_retriever(state: AgentState):
        if not st.session_state.vectorstore:
            return {"documents": []}
        k_value = min(10, len(st.session_state.docs_split))  # Increased k for better retrieval
        search_results = st.session_state.vectorstore.max_marginal_relevance_search(state['input'], k=k_value)
        relevant_docs = [doc.page_content[:500] for doc in search_results]
        return {"documents": relevant_docs if relevant_docs else []}

    def agent_reasoner(state: AgentState):
        if state['documents']:
            summary_prompt = f"Summarize these key points related to '{state['input']}': {' '.join(state['documents'])}"
            analysis = llm.invoke(summary_prompt)
            return {"analysis": analysis}
        else:
            general_knowledge_prompt = f"Provide a detailed answer for: {state['input']}"
            analysis = llm.invoke(general_knowledge_prompt)
            return {"analysis": analysis}

    def agent_responder(state: AgentState):
        return {"output": state['analysis']}

    builder = StateGraph(AgentState)
    builder.add_node("retriever", agent_retriever)
    builder.add_node("reasoner", agent_reasoner)
    builder.add_node("responder", agent_responder)
    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "reasoner")
    builder.add_edge("reasoner", "responder")
    graph = builder.compile()

    def run_multi_agent(input_text: str):
        try:
            state = {"input": input_text, "documents": [], "analysis": "", "output": ""}
            with st.spinner("Processing..."):
                result = graph.invoke(state)
            return result.get("output", "No output generated")
        except Exception as e:
            return f"Error: {str(e)}"

    input_text = st.text_input("📝 Enter query")
    if st.button("Run System"):
        result_output = run_multi_agent(input_text)
        with st.expander("📌 Final Output"):
            st.write(result_output)

        if st.session_state.vectorstore and any(input_text.lower() in doc.lower() for doc in st.session_state.docs_split):
            st.markdown("### 📖 Retrieved from PDF")
        else:
            st.markdown("### 🌍 Answer Generated from LLM")
else:
    st.warning("Upload a PDF first.")
