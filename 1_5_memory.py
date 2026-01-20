from agno.os import AgentOS
from agno.agent import Agent
from agno.db.sqlite import SqliteDb

from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

from os import getenv
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
TAVILY_APY_KEY = getenv("TAVILY_APY_KEY")

db=SqliteDb(db_file="temp/db_agno_agent.db")

agno_agent = Agent(
    name="analista_financeiro",
    model=OpenAIChat(id="gpt-4.1-mini", api_key=OPENAI_API_KEY),
    db=db,
    tools=[YFinanceTools()],
    instructions="""
        Você é um analista financeiro.
        Sempre que o usuário informar preferências (formato, nível de detalhe),
        salve essas informações na memória do usuário.
    """,
    add_history_to_context=True,
    enable_user_memories=True, # Habilitando salvamento do de dados na memória
    add_memories_to_context=True, # Habilitando salvamento do contexto da conversa na memória
    num_history_runs=3,
    enable_agentic_memory=True
)

agno_agent.print_response(
    input="Olá, prefiro as resposta em formato de tabela, gosto de poucas informações",
    session_id="petrobras_session_1",
    user_id="petrobras_analitic"
)

agno_agent.print_response(
    input="Olá, prefiro as resposta em formato de texto, gosto de bastante detalhes.",
    session_id="vale_session_1",
    user_id="vale_analitic"
)

agno_agent.print_response(
    input="Qual é a cotação da Petrobras?",
    session_id="petrobras_session_2",
    user_id="petrobras_analitic"
)

agno_agent.print_response(
    input="Qual é a cotação da Vale?",
    session_id="vale_session_2",
    user_id="vale_analitic"
)

""" agent_os = AgentOS(agents=[agno_agent])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="1_4_analitic:app", reload=True) """