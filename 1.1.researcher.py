from agno.agent import Agent

from agno.tools.tavily import TavilyTools
from agno.models.openai import OpenAIChat

from os import getenv
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
TAVILY_APY_KEY = getenv("TAVILY_APY_KEY")

agent = Agent(
    model=OpenAIChat(
        id="gpt-4.1-nano",
        api_key=OPENAI_API_KEY
    ),
    tools=[
        TavilyTools(
            api_key=TAVILY_APY_KEY
        )
    ],
    instructions=[
        "Use uma liguagem de facil compreenção para qualquer faixa etária de pessoas",
        "O resoltado deve ser aprenntado em forma de tabela"
    ]
)

agent.print_response("Me conte mais sobre os passeios do site Parrachos Náutica, de Maracajaú.")
