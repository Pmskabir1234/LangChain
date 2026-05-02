from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

emb = OpenAIEmbeddings(model='text-embedding-ada-002', dimensions=32)

#single query embedding
res = emb.embed_query("I enjoy learning Ai")
print(str(res))


documents = [
    "SRK is the most famous celeb in the world",
    "Ai technology evovlves each day",
    "Portugal will win FIFA 2026"
]

result = emb.embed_documents(documents)
print(str(result))