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
    
    # Definimos um tamanho de lote menor que o limite de 100 do Google
    tamanho_lote = 50
    
    print(f"Iniciando vetorização de {len(chucks)} chunks em lotes de {tamanho_lote}...")

    # Criamos o banco inicial com o primeiro lote
    db = Chroma.from_documents(
        documents=chucks[:tamanho_lote],
        embedding=embeddings,
        persist_directory="./rag_db"
    )
    
    # Processamos os demais lotes com uma pausa entre eles
    for i in range(tamanho_lote, len(chucks), tamanho_lote):
        print(f"Aguardando 15 segundos para evitar limite de cota do Google...")
        time.sleep(15) # Pausa necessária para o plano gratuito
        
        fim = min(i + tamanho_lote, len(chucks))
        lote_atual = chucks[i:fim]
        
        print(f"Enviando lote: chunks {i} até {fim}...")
        db.add_documents(lote_atual)

    print(f"{'='*50}")
    print(f'DB criada com sucesso na pasta ./chroma_db!')


criar_db()
