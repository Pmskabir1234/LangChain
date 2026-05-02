from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.5)

result = model.invoke("What is the use of Agentic ai?")

print(result)

#since result gives content & extra kwargs, so we will take the content only
print(result.content)