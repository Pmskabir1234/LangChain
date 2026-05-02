from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-5-sonnet-2201033848')

result = model.invoke("How to score marks in exam without studying?")

print(result.content)