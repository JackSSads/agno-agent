from agno.agent import Agent

from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools

from os import getenv
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
TAVILY_APY_KEY = getenv("TAVILY_APY_KEY")

def c_to_fh(temperature_celsius: float):
    """
    Converte uma temperatura em graus celsius em Fahrenheit.

    Args:
        temperature_celsius: (float): Temperatura em graus Celsius

    Return:
        float: Temperatura convertida em graus Fahrenheit
    """
    return (temperature_celsius * 9/5) + 32


agent = Agent(
    model=OpenAIChat(
        api_key=OPENAI_API_KEY,
        id="gpt-4.1-nano"
    ),
    tools=[
        TavilyTools(
            api_key=TAVILY_APY_KEY
        ),
        c_to_fh,
    ],
    instructions=["Use tabelas para mostrar a informação final com duas colunas: Data e Temperatura. Não inclua nem um outro texto."],
    debug_mode=True
)

agent.print_response("Use suas ferramentas para pesquisar a temperatura dos ultimos 10 dias, dia a dia, em Natal em Fahrenheit.", stream=True)