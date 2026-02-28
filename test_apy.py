import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

def listar_modelos():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    print("--- Modelos que suportam Embeddings ---")
    try:
        for m in client.models.list():
            # No SDK mais recente, usamos 'supported_actions'
            if 'embedContent' in m.supported_actions:
                print(f"Nome: {m.name}")
    except Exception as e:
        print(f"Erro ao conectar com a API: {e}")

if __name__ == "__main__":
    listar_modelos()