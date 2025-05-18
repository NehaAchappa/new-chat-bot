import streamlit as st
import os 
os.environ["GOOGLE_API_KEY"]=st.secrets["GOOGLE_API_KEY"]
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage 

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Set page title
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖")

st.title("🤖 Gemini 2.0 Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a helpful assistant.")]

# User input box
user_input = st.chat_input("Type your message...")

# If the user submits a message
if user_input:
    # Append human message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Display user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the LLM
    result = llm.invoke(st.session_state.chat_history)

    # Append AI response
    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(result.content)

# Display full chat history
for msg in st.session_state.chat_history[1:]:  # Skip the initial SystemMessage
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
