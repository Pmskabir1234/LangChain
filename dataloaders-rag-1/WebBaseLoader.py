# WebBaseLoader uses beatifulsoup under the hood to extract info among HTML tags
# for js-heavy webpages consider using SeleniumUrlLoader for better experience

from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

url = "https://docs.langchain.com/oss/python/integrations/document_loaders"

loader = WebBaseLoader(url)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Answer the {question} from the following {doc}",
    input_variables=['question','doc']
)

docs = loader.load()
chain = prompt | model | parser

print(chain.invoke({"question":'What are the different type of PDF loaders in Langchain with their use cases?',"doc":docs[0].page_content}))



