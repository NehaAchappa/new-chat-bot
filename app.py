import os
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

# Load and split the documents
loader = TextLoader("data/docs.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings and FAISS vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(docs, embeddings)

# Setup Retriever and QA chain
retriever = vectorstore.as_retriever()
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = qa_chain.run(query)
    print("Bot:", response)
