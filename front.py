import streamlit as st
from back import qa_agent
from langchain_classic.memory import ConversationBufferMemory

st.title("PDF问答AI")
with st.sidebar:
    deepseek_api_key=st.text_input("Enter your DEEPSEEK API key",type="password")
if "memory" not in st.session_state:
    st.session_state["memory"]=ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
    )
uploaded_file=st.file_uploader("Upload your file",type="pdf")
question=st.text_input("Enter your question",disabled=not uploaded_file)
if uploaded_file and question and deepseek_api_key:
    response=qa_agent(deepseek_api_key,st.session_state["memory"],uploaded_file,question)
    st.write("解答：")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]
else:
    st.info("Exist empty content!")
if "chat_history" in st.session_state:
    with st.expander("历史消息："):
        for i in range(0,len(st.session_state["chat_history"]),2):
            st.write(st.session_state["chat_history"][i])
            st.write(st.session_state["chat_history"][i+1])
