from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='test-generation'
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("Generate notes on OutputParser in Langchain, usage, short notes P")
print(res.content)