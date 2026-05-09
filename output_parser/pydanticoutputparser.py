from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Character(BaseModel):
    name: str = Field(description='Name of the superhero')
    damage: int = Field(description='Attack damage of the superhero')
    ability: str = Field(description='Special ability of the superhero')

parser = PydanticOutputParser(pydantic_object=Character)

template = PromptTemplate(
    template="""Create a superhero like {theme}, give name, the damage he makes
    in attack (number), and his special ability.\n {format_instructions}""",
    input_variables=['theme'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

chain = template | model | parser

res = chain.invoke({'theme':'DC'})

print(res, "\n")
print(type(res))    #basically a pydantic object

res = dict(res)

print(res.items())
