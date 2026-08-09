from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint,  ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    model = 'google/gemma-4-31B-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template='summarize the given text : {text}',
    input_variables=['text']
)

# a document contains two parts - page_content, meta_data, we can access like attribute
loader = TextLoader('external_knowledge.txt',autodetect_encoding=True)
docs = loader.load()


parser = StrOutputParser()
chain = template | model | parser

print(docs, '\n')
print(type(docs), '\n')
print(docs[0].page_content, '\n', docs[0].metadata, '\n')

print(chain.invoke({'text': docs[0].page_content}))