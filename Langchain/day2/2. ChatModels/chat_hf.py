import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not api_key:
    raise ValueError("API key kaha hai bhai")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    huggingfacehub_api_token=api_key,
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is HuggingFace")
print(result.content)
