import streamlit as st
from loader import load_and_index_pdf
from chatbot import create_chatbot

st.title("📚 RAG Chatbot using PDF + Gemini")
st.markdown("Ask anything based on the uploaded PDF document!")

if "chatbot" not in st.session_state:
    with st.spinner("Indexing the PDF..."):
        vectorstore = load_and_index_pdf("data/your_doc.pdf")
        st.session_state.chatbot = create_chatbot(vectorstore)
        st.success("Chatbot is ready!")

query = st.text_input("Enter your question:")

if query:
    response = st.session_state.chatbot({"question": query})
    st.write("**Answer:**", response["answer"])
