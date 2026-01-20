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

db=SqliteDb(db_file="temp/data.db")

agno_agent = Agent(
    name="analista_financeiro",
    model=OpenAIChat(id="gpt-4.1-nano", api_key=OPENAI_API_KEY),
    db=db,
    tools=[YFinanceTools()],
    instructions=[
        "Use tabelas para mostrar a informação final.",
        "Não inclua nem um outro texto.",
        "Não tente apresentar cotações de empresas que não foral solicitadas."
    ],
    add_history_to_context=True,
    num_history_runs=10
)

""" agno_agent.print_response(
    input="Qual é a cotação da Petrobras?",
    session_id="petrobras",
    user_id="petrobras_analitic"
)

agno_agent.print_response(
    input="Qual é a cotação da Vale?",
    session_id="vale",
    user_id="vale_analitic"
)

agno_agent.print_response(
    input="Quais empresas já foram consultadas a cotação?",
    session_id="empresas_session",
    user_id="empresas_analitic"
) """

agent_os = AgentOS(agents=[agno_agent])
app = agent_os.get_app()

if __name__ == "__main__":
    agent_os.serve(app="1_4_analitic:app", reload=True)