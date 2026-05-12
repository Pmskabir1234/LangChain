from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    model = 'google/gemma-4-31B-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='sentiment of the user from feedback')

parser1 = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()

prompt = PromptTemplate(
    template='Classify the feedback based on sentiment either positive or negative: \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser1.get_format_instructions()}
)


sentiment_chain = prompt | model | parser1

prompt2 = PromptTemplate(
    template='Write a appropriate positive feedback to the given feedback: {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a appropriate negative feedback to the given feedback: {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt2 | model | parser2 ),
    (lambda x: x.sentiment=='negative', prompt3 | model | parser2),
    RunnableLambda(lambda x: 'Could not find sentiment')
)

chain = sentiment_chain | branch_chain

res = chain.invoke({'feedback':'samsung book 4 is terrible product due its slow motherboard'})
print(res)