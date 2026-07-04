from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()


llm1 = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task = 'text-generation'
)
model1 = ChatHuggingFace(llm=llm1)

llm2  = HuggingFaceEndpoint(
    model = 'Qwen/Qwen2.5-1.5B-Instruct',
    task = 'text-generation'
)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template= 'Generate a {length} article on {topic} to post in Medium',
    input_variables=['length','topic']
)

prompt2 = PromptTemplate(
    template= 'Generate a {length} article on {topic} to post in Twitter',
    input_variables=['length','topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'blog':RunnableSequence(prompt1, model1, parser),
    'tweet':RunnableSequence(prompt2, model2, parser)
})

res = chain.invoke({'length':'Moderately Lengthy','topic':'Evolution of Ai Agents'})
print(res.get('blog','\n'))
print(res.get('tweet'))