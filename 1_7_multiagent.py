from agno.team import Team
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.chroma import ChromaDb
from agno.tools.hackernews import HackerNewsTools
from agno.tools.yfinance import YFinanceTools
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools

from os import getenv
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

db=SqliteDb(db_file="temp/db_agno_agent.db")

vector_db = ChromaDb(
    collection="multiagent_knowledge",
    path="temp/chromadb",
    embedder=OpenAIEmbedder(api_key=OPENAI_API_KEY),
    persistent_client=True
)

knowledge = Knowledge(
    vector_db=vector_db
)

analista_noticias_agent = Agent(
    name="Analista de Notícias",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGoTools(enable_search=False, enable_news=True)],
    role="Você é um pesquisador de notícias.",
    instructions="Use usas toos para encontrar informações na web sobre empreas listadas na B3.",
    markdown=True
)

analista_cotacoes_agent = Agent(
    name="Analista de Cotações",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    tools=[YFinanceTools()],
    instructions="Você é um analista de cotações. Use as ferramentas disponíveis para fornecer análises detalhadas sobre ações e tendências de mercado.",
    markdown=True
)

analista_relatorios_agent = Agent(
    name="Analista de Relatórios",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    instructions="Você é um analista de relatórios de empresas listadas na B3.",
    knowledge=knowledge,
    add_knowledge_to_context=True,
    markdown=True
)

analista_team = Team(
    name="Team analstas",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    members=[
        analista_noticias_agent,
        analista_cotacoes_agent,
        analista_relatorios_agent
    ],
    instructions=[
        "Você deve entender as informações solicitadas pelo usuário e fornecer uma resposta adequada.",
        "Para obter informações sobre o DRE, utilize o analista de relatórios.",
        "Para obter informações sobre cotações, utilize o analista de cotações.",
        "Para obter informações sobre notícias, utilize o analista de notícias.",
    ],
    db=db,
    add_history_to_context=True,
    num_history_runs=3,
    show_members_responses=True,
    get_member_information_tool=True,
    add_datetime_to_context=True,
    markdown=True,
    debug_mode=True
)

analista_team.print_response(
    "Olá! Qual foi o lucro liquido da Petrobrás em 2T25, segundo o relatório publicado? Utilize o analista notícias para verificar se houve alguma notícia relevante sobre o assunto e o analista de cotações para verificar a cotação atual da ação.",
    session_id="petrobras_session_11",
    user_id="analista_petrobras"
)