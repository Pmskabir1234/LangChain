from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

joke_gen = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser )
})

final_chain = RunnableSequence(joke_gen, chain)




print(final_chain.invoke({'topic':'Ai'}))