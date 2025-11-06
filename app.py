import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from dotenv import load_dotenv 
import os
import streamlit as st
from streamlit_community_navigation_bar import st_navbar
import warnings
import tempfile

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings('ignore')
load_dotenv()

# API environment setup
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "store" not in st.session_state:
    st.session_state["store"] = {}

# Chroma directory
CHROMA_DB_DIR = "/tmp/chroma_db"  # For Streamlit Cloud write access

def create_vectorstore_from_docs(docs, embedding):
    return Chroma.from_documents(docs, embedding_function=embedding)

try:
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", convert_system_message_to_human=True)
except Exception as e:
    st.error(f"Error initializing Gemini models: {e}")
    st.stop()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

# Streamlit page config
st.set_page_config(page_title="RAG with Conversational Memory", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
body, .stApp {
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
    color: #e0e6f0 !important;
}
.stNavigationBar {
    background: #23243a !important;
    border-radius: 14px;
    box-shadow: 0 8px 32px rgba(30,40,80,0.25), 0 1.5px 6px rgba(0,0,0,0.12);
    margin-top: 18px !important;
    margin-bottom: 36px !important;
    padding: 0.7rem 0;
    min-height: 60px;
    display: flex;
    align-items: center;
    z-index: 100;
    border-bottom: 2px solid #3e68ff;
}
.stNavigationBar span {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    padding: 12px 36px !important;
    border-radius: 10px !important;
    margin: 0 10px !important;
    transition: background 0.2s, color 0.2s;
    color: #e0e6f0 !important;
}
.stNavigationBar span.active {
    background: #3e68ff !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(62,104,255,0.18);
}
.stNavigationBar span:hover {
    background: #35365a !important;
    color: #fff !important;
}
.heading-box {
    background-color:#35365a;
    color:#fff;
    border-radius:16px;
    padding:18px 28px 14px 28px;
    margin-bottom:18px;
    font-size:2.1rem;
    font-weight:700;
    text-align:left;
    letter-spacing:1px;
}
.history-box {
    background-color:#35365a;
    color:#fff;
    border-radius:12px;
    padding:12px 22px 10px 22px;
    margin-top:24px;
    margin-bottom:10px;
    font-size:1.1rem;
    font-weight:500;
    text-align:left;
    letter-spacing:0.5px;
}
.main-content {
    padding: 0 32px 24px 32px;
}
.stTextInput>div>div>input { 
    background-color: #2d2d44; color: white; 
    border-radius: 8px; border: 1px solid #444;
}
.stButton>button {
    background-color: #3e68ff; color: white;
    border-radius: 8px; border: none;
    padding: 10px 20px;
}
.stButton>button:hover { background-color: #5e78ff; }
.chat-history {
    background-color: #2d2d44; border: 1px solid #444; border-radius: 8px;
    padding: 15px; margin-top: 10px; max-height: 400px; overflow-y: auto;
}
[data-testid="stSidebar"] {
    background-color: #1e1e2e;
    border-right: 1px solid #383850;
}
.contact-box {
    background-color: #ccccc6;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 10px;
    color: #040760;
}
.contact-title {
    font-size: 20px; font-weight: bold; color: #1a73e8; margin-bottom: 10px;
}
.contact-item { margin-bottom: 8px; }
.contact-icon { margin-right: 8px; }
a.contact-link { color: #1a73e8; text-decoration: none; }
a.contact-link:hover { text-decoration: underline; }
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1e1e2e;
    border-top: 1px solid #383850;
    color:#FFC300;
    text-align: center;
    padding: 10px 0;
    z-index: 100;
}
.stApp {
    padding-bottom: 50px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Navigation ---
selected_page = st_navbar(["Home", "How to Use", "About Us", "Team", "Contact Us", "Future Enhancements"])

# --- Footer (fixed and correctly placed) ---
st.markdown("""
<div class='footer'>
    © 2025 ASHI JAIN — All rights reserved.
</div>
""", unsafe_allow_html=True)
