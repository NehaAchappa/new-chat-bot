import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize session state
if "qa_chain" not in st.session_state:
    # Load and process documents
    loader = TextLoader("data/docs.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Set up retriever and LLM
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    # Create RetrievalQA chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG Chatbot with Gemini")
st.write("Ask anything related to the knowledge base!")

query = st.text_input("Your question:", placeholder="Ask a question...")

if query:
    response = st.session_state.qa_chain.run(query)
    st.markdown("### 🤖 Answer")
    st.write(response)
