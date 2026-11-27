import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

model = ChatOpenAI(
    model="gpt-6-luna",
    max_completion_tokens=90,
)
result = model.invoke("Write a 5 line poem on glasses")
# print(result)
print(result.content)
