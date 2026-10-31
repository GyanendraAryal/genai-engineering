from pathlib import Path
import os
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
pdf_path = Path(__file__).parent / "doc.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
split_docs = text_splitter.split_documents(documents=docs)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key kaha xa bhai??")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectore_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embeddings,
# )
# vectore_store.add_documents(documents=split_docs)
# print("Injection done ✅")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embeddings,
)

relevant_chunks = retriever.similarity_search(query="What is in the week 3?")

SYSTEM_PROMPT = f"""
Your are an helpful AI assistant who responds base of the available context.

Context:{relevant_chunks.page, relevant_chunks.page}

"""
