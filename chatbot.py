import threading
import uvicorn
import streamlit as st
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import logging

app = FastAPI()

logging.basicConfig(filename="chat_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

llm = OllamaLLM(model="llama3")

prompt_template = PromptTemplate(
    input_variables=["chat_history", "user_input"],
    template="You are a helpful AI assistant. Here is the conversation so far:\n{chat_history}\nUser: {user_input}\nAI:"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = LLMChain(prompt=prompt_template, llm=llm, memory=memory)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = chain.run({"user_input": request.user_input})
        logging.info(f"User: {request.user_input}")
        logging.info(f"AI: {response}")
        return {"response": response}
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="LLaMA 3 Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Chat with LLaMA 3 using FastAPI!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = requests.post(API_URL, json={"user_input": user_input})
    
    if response.status_code == 200:
        bot_response = response.json()["response"]
    else:
        bot_response = "⚠️ Error: Unable to get response from the server."

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    with st.chat_message("assistant"):
        st.markdown(bot_response)
