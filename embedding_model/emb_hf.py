from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "I love India"
documents = [
    "SRK is the most famous celeb in the world",
    "Ai technology evovlves each day",
    "Portugal will win FIFA 2026"
]

res = embedding.embed_query(text)
print(str(res))

result = embedding.embed_documents(documents)
print(str(result))