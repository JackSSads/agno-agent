from agno.os import AgentOS
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

from agno.vectordb.chroma import ChromaDb
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.knowledge.reader.csv_reader import CSVReader
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.embedder.openai import OpenAIEmbedder

from os import getenv
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
TAVILY_APY_KEY = getenv("TAVILY_APY_KEY")

db=SqliteDb(db_file="temp/db_agno_agent.db")

# === Rag ===
vector_db = ChromaDb( # Criação do bando vetorial
    collection="empresas_relatorios",
    path="temp/chromadb",
    embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
    persistent_client=True
)

# Create Knowledge base
knowledge = Knowledge( # Passando o bando de dados para o Knowledge - conhecimento
    vector_db=vector_db
)

knowledge.add_content(  # Passando arquivos para o Knowledg
    path="data_to_rag/PETR/", # O path pode ser um caminho ou URL
    reader=PDFReader(
        chunking_strategy=SemanticChunking() # Aqui passa a estratégia do chunking
    ),
    skip_if_exists=True # Faz com que o chinking e emdeadding todo novamente caso já exista na base
)

""" knowledge.add_content(
    path="data_to_rag/VALE/",
    reader=PDFReader(
        chunking_strategy=SemanticChunking()
    ),
    metadata={ 
        "company": "Vale",
        "sector": "Mineração",
        "country": "Brazil"
    },
    skip_if_exists=True
) """

agno_agent = Agent(
    name="analista_financeiro",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[YFinanceTools()],
    instructions="""
        Você é um analista financeiro.
        Sempre que o usuário informar preferências (formato, nível de detalhe),
        salve essas informações na memória do usuário.
    """,
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    enable_user_memories=True, # Habilitando salvamento do de dados na memória
    enable_agentic_memory=True,
    knowledge=knowledge,
    add_memories_to_context=True, # Habilitando salvamento do contexto da conversa na memória
)

agno_agent.print_response(
    input="Gere um resumo do documento explicando de forma susinta os principais pontos.",
    session_id="petrobras_session_6",
    user_id="petrobras_analitic"
)

""" agno_agent.print_response(
    input="Gere um resumo da conversa recente sobre a VALE, incluindo a cotação atual da ação e as últimas notícias relevantes.",
    session_id="vale_session_6",
    user_id="vale_analitic"
) """