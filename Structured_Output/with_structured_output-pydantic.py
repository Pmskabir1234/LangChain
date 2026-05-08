#we will extract details of a person from resume type paragraph

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema
class Details(BaseModel):
    name: str
    age: Optional[int]
    skills: list[str]
    email: Optional[EmailStr]
    experience: Optional[float] = Field(gt=0, default=0, description='Number of years for working in any company')
    imperssion: Literal['Good', 'Average', 'Bad']

structured_model = model.with_structured_output(Details)
res = structured_model.invoke("""John Anderson is a software engineer with 4 years of experience
                               in full-stack web development and cloud-based applications. He is skilled
                               in Python, JavaScript, React, Node.js, and PostgreSQL, with hands-on 
                              experience deploying scalable services using Docker and AWS. John completed his
                               Bachelor of Science in Computer Science from the University of California,
                               Berkeley in 2021. He previously worked at TechNova Solutions as a Backend Developer,
                               where he optimized API response times by 35% and collaborated with cross-functional
                               teams on microservice architecture. He is also certified in AWS Certified Developer –
                               Associate and has built several personal projects involving AI chatbots, REST APIs, and
                               data visualization dashboards.""")

print(res)