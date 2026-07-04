# this helps you to convert any python function to runnable, which is pretty useful and easy to add in workflow

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
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

# def word_counter(text):
#     return len(text.split())

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt1, model, parser)

chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen, chain)

print(final_chain.invoke({'topic':'K-pop'}))