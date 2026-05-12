from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='test-generation'
)

model = ChatHuggingFace(llm=llm)

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

template1 = PromptTemplate(
    template="Create a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="""summarise the given text to extract the important notes:
     topic: {text}""",
     input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

res = chain.invoke({'topic':'Ironman'})

print(res,"\n")

# let's visualise the chain
chain.get_graph().print_ascii()