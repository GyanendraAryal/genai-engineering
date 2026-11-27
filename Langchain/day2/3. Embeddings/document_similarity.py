from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)
documents = [
    "Lionel Messi is an Argentine footballer who has won multiple Ballon d'Or awards.",
    "Cristiano Ronaldo is a Portuguese footballer known for his goal-scoring records.",
    "Virat Kohli is an Indian cricketer known for his batting consistency and aggressive playing style.",
    "LeBron James is an American basketball player who has won multiple NBA championships.",
    "Serena Williams is an American tennis player who has won 23 Grand Slam singles titles.",
]
query = "Tell me about Williams"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("Similarity score is: ",score)
