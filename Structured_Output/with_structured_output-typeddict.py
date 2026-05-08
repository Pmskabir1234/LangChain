#we will pass a review and generate a structured output using typeddict

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema
class Review(TypedDict):
    summary : Annotated[str,'A brief summary of the review']
    sentiment : Literal['pos','neg']
    recommendation: Annotated[Optional[str], 'if the review says to watch the movie']

#instance creation example
# new_review : Review = {'summary':'The hardware was great but very disgusting heating issue',
#                        'sentiment':'negative'}

structured_model = model.with_structured_output(Review)

res = structured_model.invoke("""
                            shutter Island is a gripping psychological thriller with an intense atmosphere
                            and a brilliant performance by Leonardo DiCaprio. The movie keeps you
                            questioning reality until the very end, and the twist completely changes
                            how you see the story. Dark, smart, and emotionally heavy — definitely worth
                            watching if you enjoy mind-bending films.
                            """)

print(res,"\n")
print(res['summary'])
print(res['sentiment'])
