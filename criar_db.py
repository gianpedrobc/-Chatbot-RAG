import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
PASTA_BASE = 'base'

def criar_db():
    documentos = carregar_documentos()
    chucks = dividir_chucks(documentos)
    vetrizar_chucks(chucks)

def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE)
    documentos = carregador.load()
    print(f"{'='*50}")
    print(f'Foram carregados {len(documentos)} documentos!')
    return documentos

def dividir_chucks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)    
    chucks = separador_documentos.split_documents(documentos)
    print(f"{'='*50}")
    print(f'Foram criados {len(chucks)} chucks!')
    return chucks

def vetrizar_chucks(chucks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Reduzimos para 25 para ser mais conservador
    tamanho_lote = 25 
    
    print(f"Iniciando vetorização de {len(chucks)} chunks...")

    # Criamos o banco inicial
    db = Chroma.from_documents(
        documents=chucks[:tamanho_lote],
        embedding=embeddings,
        persist_directory="./rag_db"
    )
    
    for i in range(tamanho_lote, len(chucks), tamanho_lote):
        # Aumentamos a pausa para 20-30 segundos se o erro persistir
        print(f"Aguardando 20 segundos para respeitar a cota...")
        time.sleep(20) 
        
        fim = min(i + tamanho_lote, len(chucks))
        lote_atual = chucks[i:fim]
        
        print(f"Enviando lote: chunks {i} até {fim}...")
        try:
            db.add_documents(lote_atual)
        except Exception as e:
            print(f"Erro no lote {i}: {e}")
            print("Tentando novamente em 60 segundos...")
            time.sleep(60)
            db.add_documents(lote_atual)

    print("DB criada com sucesso!")


criar_db()
