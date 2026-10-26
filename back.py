from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
import jieba

def qa_agent(moonshot_api_key,memory,uploaded_file,question):
    model = ChatOpenAI(model="kimi-k2.6",api_key=moonshot_api_key,base_url="https://api.moonshot.cn/v1")
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
    retriever = BM25Retriever.from_documents(
        texts,
        preprocess_func=lambda s: list(jieba.cut_for_search(s))
    )
    qa=ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response=qa.invoke({"chat_history":memory,"question":question})
    return response