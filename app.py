import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ChatBot RAG com Gemini")
st.markdown("---")

# Inicializa o banco de dados Chroma
@st.cache_resource
def carregar_banco_dados():
    """Carrega o banco de dados Chroma uma única vez"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    db = Chroma(
        persist_directory="./rag_db", 
        embedding_function=embeddings,
    )
    return db

@st.cache_resource
def carregar_gemini():
    """Carrega o modelo Gemini uma única vez"""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",version="v1", temperature=0.7)

# Carrega recursos
db = carregar_banco_dados()
llm = carregar_gemini()

# Interface da sidebar
st.sidebar.header("⚙️ Configurações")
num_documentos = st.sidebar.slider(
    "Número de trechos a buscar:",
    min_value=1,
    max_value=10,
    value=3,
    help="Quantos trechos similares usar na resposta"
)

temperatura = st.sidebar.slider(
    "Temperatura da IA:",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Quanto menor, mais determinística; maior, mais criativa"
)

# Template do prompt
template_prompt = """Você é um assistente útil que responde perguntas baseado em documentos.

Use os seguintes trechos dos documentos para responder a pergunta do usuário:

TRECHOS DO BANCO DE DADOS:
{contexto}

PERGUNTA DO USUÁRIO:
{pergunta}

RESPOSTA:"""

prompt = PromptTemplate(
    input_variables=["contexto", "pergunta"],
    template=template_prompt
)

# Histórico de perguntas
if "historico" not in st.session_state:
    st.session_state.historico = []

# Área principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Faça sua pergunta")
    pergunta_usuario = st.text_input(
        "Digite sua pergunta aqui:",
        placeholder="Ex: Qual é o tema principal dos documentos?",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("📚 Informações")
    st.info(f"Trechos a buscar: {num_documentos}")

# Processar pergunta
if pergunta_usuario:
    with st.spinner("🔍 Buscando trechos relevantes..."):
        # Busca trechos similares no banco de dados
        docs_similares = db.similarity_search(pergunta_usuario, k=num_documentos)
    
    # Formata os trechos como contexto
    contexto_formatado = "\n\n".join([
        f"[Documento {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs_similares)
    ])
    
    # Exibe os trechos encontrados
    with st.expander("📄 Trechos encontrados", expanded=False):
        for i, doc in enumerate(docs_similares):
            st.markdown(f"**Trecho {i+1}:**")
            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            st.divider()
    
    # Gera resposta do Gemini
    with st.spinner("✨ Gerando resposta com Gemini..."):
        try:
            # Cria uma nova instância do LLM com a temperatura ajustada
            llm_ajustado = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                temperature=temperatura
            )
            
            # Cria a chain usando LCEL (LangChain Expression Language)
            chain = prompt | llm_ajustado
            resposta = chain.invoke({
                "contexto": contexto_formatado, 
                "pergunta": pergunta_usuario
            }).content
            
            # Exibe a resposta
            st.subheader("💬 Resposta da IA")
            st.markdown(resposta)
            
            # Adiciona ao histórico
            st.session_state.historico.append({
                "pergunta": pergunta_usuario,
                "resposta": resposta,
                "trechos": len(docs_similares)
            })
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar resposta: {str(e)}")

# Exibe histórico
if st.session_state.historico:
    st.markdown("---")
    with st.expander("📋 Histórico de Perguntas", expanded=False):
        for i, item in enumerate(reversed(st.session_state.historico[-5:]), 1):
            st.markdown(f"**Pergunta {i}:** {item['pergunta']}")
            st.markdown(f"**Trechos usados:** {item['trechos']}")
            st.divider()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🔗 Powered by Chroma + Gemini + Streamlit</p>
    <p><small>Faça perguntas sobre os documentos carregados no banco de dados</small></p>
</div>
""", unsafe_allow_html=True)
