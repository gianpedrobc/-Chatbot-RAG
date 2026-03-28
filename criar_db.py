import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- RECURSOS ---
@st.cache_resource
def carregar_recursos():
    # Se o criar_db usou 'models/gemini-embedding-001', mantemos igual aqui
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(persist_directory="./rag_db", embedding_function=embeddings)
    return db

# --- INTERFACE ---
st.title("🤖 ChatBot RAG com Gemini")
st.markdown("---")

db = carregar_recursos()

st.sidebar.header("⚙️ Configurações")
num_docs = st.sidebar.slider("Trechos a buscar:", 1, 10, 3)
temp = st.sidebar.slider("Temperatura:", 0.0, 1.0, 0.7, 0.1)

pergunta_usuario = st.text_input("📝 Faça sua pergunta:")

if pergunta_usuario:
    with st.spinner("🔍 Buscando e Gerando..."):
        try:
            # 1. Busca no banco que você criou
            docs = db.similarity_search(pergunta_usuario, k=num_docs)
            contexto = "\n\n".join([d.page_content for d in docs])

            # 2. Configura o modelo (AQUI ESTAVA O ERRO 404)
            # No plano free, NÃO use version="v1beta" nem caminhos complexos
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=temp
            )

            # 3. Prompt
            template = "Use o contexto: {contexto}\n\nPergunta: {pergunta}"
            prompt = PromptTemplate.from_template(template)
            
            # 4. Execução
            chain = prompt | llm
            resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta_usuario})

            st.subheader("💬 Resposta")
            st.markdown(resposta.content)

            with st.expander("📄 Fontes"):
                for d in docs:
                    st.write(d.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Erro: {e}")
            st.info("Tente rodar: pip install -U langchain-google-genai")