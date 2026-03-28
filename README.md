# Chatbot RAG

Este projeto é um chatbot com Recuperação de Texto (RAG) em desenvolvimento. Ele utiliza uma base de dados local (SQLite) e integrações para processamento de texto e geração de respostas.

> **Atenção:** o projeto está em desenvolvimento. Algumas funcionalidades podem estar instáveis ou incompletas.

## Estrutura do Repositório

- `main.py` - Script principal para iniciar o chatbot.
- `criar_db.py` - Utilitário para criar/popular o banco de dados.
- `base/` - Diretório para arquivos auxiliares (se houver).
- `rag_db/` - Contém o banco de dados `chroma.sqlite3` e diretórios relacionados.
- `test_apy.py` - Arquivo de testes.
- `requirements.txt` - Dependências do projeto.

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/gianpedrobc/-Chatbot-RAG.git
   cd chatbot_RAG
   ```
2. Crie e ative um ambiente virtual (por exemplo, `venv`):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso Básico

1. Configure seu banco de dados local, se necessário, executando:
   ```bash
   python criar_db.py
   ```
2. Execute o chatbot:
   ```bash
      streamlit.exe run app.py
   ```

