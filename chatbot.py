from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

def create_chatbot(vectorstore):
    retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
    )

    return qa_chain
