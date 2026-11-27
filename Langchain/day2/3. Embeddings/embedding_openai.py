from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
result = embedding.embed_query("Ktm is the capital of Nepal.")
print("Dimension: ",len(result))
print("Dimension: ",result)

