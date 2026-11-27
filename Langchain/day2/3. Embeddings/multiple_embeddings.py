from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
documents = [
    "KTM is the capital of Nepal",
    "Kolkata is the capital of West Bangal",
    "Paris is the capital of France",
]
result = embedding.embed_documents(documents)
print(str(result))
