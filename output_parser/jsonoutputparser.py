# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = JsonOutputParser()

template = PromptTemplate(
    template="""Give me 5 facts about the existance of {topic} \n {format_instruction}""",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'Karma'})
print(result)
# print(type(result))   -> dict
