#we will pass a review and generate a structured output using typeddict

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema
json_schema = {
    "title": "Student",
    "descripton": "A student record",
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
            
        },
        "age": {
            "type": "integer"
            
        },
        "grade": {
            "type": "string"
        }
    },
    "required": ["name", "grade"]
}

#instance creation example
# new_review : Review = {'summary':'The hardware was great but very disgusting heating issue',
#                        'sentiment':'negative'}

structured_model = model.with_structured_output(json_schema)

res = structured_model.invoke("""
                            Aarav Sharma is a 15-year-old student currently studying in Grade 10 at Green Valley High School.
                               He is passionate about science and mathematics and actively participates in 
                              coding competitions and robotics clubs. Aarav is known for his disciplined nature,
                               strong problem-solving skills, and curiosity about technology. Apart from academics,
                               he enjoys playing cricket, reading mystery novels, and learning Python programming in
                               his free time.

                            """)

print(res)
