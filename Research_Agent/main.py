# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from tools import search_tool, wiki_tool, save_to_txt

load_dotenv()


# llm = HuggingFaceEndpoint(
#     repo_id = "google/gemma-4-31B-it",
#     task='text-generation'
# )

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', max_output_tokens=1500)

class ResearchResponse(BaseModel):
    topic: str
    summary : str
    sources : list[str]
    tools_used : list[str]


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """You are a helpful research assistant that will help to generate research paper.
#             Answer the query and use necessary tools.
#             wrap the output in this format and provide no other text \n {format_instructions}""",
#         ),
#         ("placeholder","{chat_history}"),
#         ("human","{query}"),
#         ("placeholder","{agent_scratchpad}")
#     ]
# ).partial(format_instructions = parser.get_format_instructions())

tools = [search_tool, wiki_tool]

agent = create_agent(
    model=model,
    system_prompt="""You are a helpful research assistant.

Your job is to research the user's question using the available tools.

Follow these rules:

1. Use the web search tool when current information is required.
2. Use Wikipedia when useful for background information.
3. Give a clear and concise summary.
4. Include the sources you used.
5. Include the names of the tools you actually used.
6. Do not invent sources.
7. If the user asks you to save the research, use the save_to_txt tool.
""",
    tools=tools,
    response_format=ToolStrategy(ResearchResponse)
)

query = "How GPT Astra model replace different job roles. Save it"

raw_response = agent.invoke({
    'messages':[
        {
            'role':'user',
            'content':query
        }
    ]
})

print("\nAvailable response keys:")
print(raw_response.keys())


if "structured_response" not in raw_response:
    print("\nStructured response was not generated.")
    print(raw_response["messages"][-1])
else:

    parsed_response = raw_response["structured_response"]

    print("\n========== RESEARCH RESPONSE ==========\n")

    print(f"Topic:\n{parsed_response.topic}")

    print(f"\nSummary:\n{parsed_response.summary}")

    print("\nTools Used:")
    for tool in parsed_response.tools_used:
        print(f"- {tool}")

    print("\nSources:")
    for source in parsed_response.sources:
        print(f"- {source}")

# parsed_response = raw_response['structured_response']

# print(raw_response)
# print(f"Topic: {parsed_response.topic} \nSummary: {parsed_response.summary} \nTools : {parsed_response.tools_used} \nSource : {parsed_response.sources}")

# {'messages': [HumanMessage(content="Langchain's current use in industry",
#              additional_kwargs={}, response_metadata={}, id='9f4295d0-fe65-45df-86bc-5a7af5e421d4'),
#             AIMessage(content='```json\n{\n  "title', additional_kwargs={},
#              response_metadata={'finish_reason': 'MAX_TOKENS', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'},
#              id='lc_run--01a07215-6036-7ac0-b8f6-e3204a15ea26-0', tool_calls=[], invalid_tool_calls=[], 
#             usage_metadata={'input_tokens': 54, 'output_tokens': 246, 'total_tokens': 300, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 238}})]}