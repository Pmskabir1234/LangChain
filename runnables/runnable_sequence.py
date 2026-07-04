from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='tell me a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='exaplain the joke {joke} in short',
    input_variables=['joke']
)

parser = StrOutputParser()

chain_old = prompt1 | model | parser
print(chain_old.invoke({'topic':'Ai'}))

chain = RunnableSequence(prompt1, model, parser,prompt2, model,parser)
print(chain.invoke({'topic':'Ai'}))