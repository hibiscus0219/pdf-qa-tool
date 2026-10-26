from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.agents import create_structured_chat_agent
from langchain_classic.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.tools import Tool

def qa_agent(deepseek_api_key,memory,uploaded_file,question):
    model = ChatOpenAI(model="deepseek_chat",deepseek_api_key=deepseek_api_key,base_url="https://api.deepseek.com")
    file_content=uploaded_file.read()
    temp_file_path="temp.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n","。","！","？","，","、",""]
    )
    texts=text_splitter.split_documents(docs)
    embeddings=OpenAIEmbeddings()
    db=FAISS.from_documents(texts,embeddings)
    retriever=db.as_retriever()
    qa=ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({"chat_history":memory,"question":question})
    return response