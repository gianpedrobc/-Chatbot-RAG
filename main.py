import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# 1. Configura os mesmos embeddings usados na criação
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Carrega o banco existente
db = Chroma(
    persist_directory="./rag_db", 
    embedding_function=embeddings,
)

# 3. Faz uma pergunta de teste
query = "Qual o principal tema dos documentos carregados?"
docs = db.similarity_search(query, k=3) # Busca os 3 trechos mais parecidos

print(f"\nResultados da busca para: '{query}'")
for i, doc in enumerate(docs):
    print(f"\n--- Trecho {i+1} ---")
    print(doc.page_content[:300] + "...") # Exibe os primeiros 300 caracteres