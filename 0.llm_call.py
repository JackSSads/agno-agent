from agno.models.openai import OpenAIChat
# from agno.models.groq import Groq
from agno.models.message import Message
from os import getenv

OPENAI_API_KEY = getenv("OPENAI_API_KEY")

from dotenv import load_dotenv
load_dotenv()

model = OpenAIChat(
    id="gpt-4.1-nano",
    api_key=OPENAI_API_KEY
)

msg = Message(
    role="user",
    content=[{
        "type": "text",
        "text": "Hello, my name is Jackson"
    }]
)

response = model.invoke(messages=[msg], assistant_message=msg)

print(response.content)