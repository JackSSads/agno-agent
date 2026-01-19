from agno.agent import Agent

from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

from os import getenv
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(
        api_key=OPENAI_API_KEY,
        id="gpt-4.1-nano"
    ),
    tools=[
        YFinanceTools()
    ],
    instructions=["Use tabelas para mostrar a informação final com duas colunas: Data e Valor da cotação. Não inclua nem um outro texto."],
)

agent.print_response("Qual a cotação atual da APPLE?", stream=True)