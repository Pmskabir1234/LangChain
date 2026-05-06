from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', output_dimensionality=300)

docs = [
    "MS Dhoni is known for his cool captainship & magnificant wicket keeping",
    "Virat Kohli is known for his aggresive celebration & milestones",
    "Shreyas is cool and talented batsman of India",
    "Shami is a wizard of pace bowling"
]

query = "Who Virat Kohli?"

doc_embed = [embedding.embed_query(doc) for doc in docs]
query_embed = embedding.embed_query(query)


#cosine_similairty -> 2d list as parameters and returns a 2d list
scores = cosine_similarity([query_embed], doc_embed)[0]
print(query)

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print(docs[index])
print("Similiarity score is: ", score)