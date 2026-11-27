import os
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

llm = OpenAI(model="gpt-6-luna")
result = llm.invoke("What is django rest framework?")
print(result)
