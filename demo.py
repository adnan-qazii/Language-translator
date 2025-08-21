from flask import Flask
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


messages = [
    SystemMessage(content="translate the following English text to hinglish"),
    HumanMessage(content="Hello, how are you?")
]


result= model.invoke(messages)

print(result.content)

