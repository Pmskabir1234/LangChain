'''
LLM Models are general purpose models and used for raw text generation only.

'''

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result= llm.invoke('Who is Sharukh Khan?')

print(result)


""""purely text-in text-out workflow"""
