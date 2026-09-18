from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import create_agent
# from langchain.agents.structured_output import ToolStrategy
from tools import search_tool, wiki_tool, save_to_txt
import json
import re

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-4-31B-it",
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', max_output_tokens=2500)

class ResearchResponse(BaseModel):
    topic: str
    paper_name : str
    summary : str
    sources : list[str]
    tools_used : list[str]


tools = [search_tool, wiki_tool, save_to_txt]

agent = create_agent(
    model=model,
    system_prompt="""You are a helpful research assistant.

Your job is to get the names of the latest research papers the given topic using the available tools.

Follow these rules:

1. Use the web search tool when current information is required.
2. Use Wikipedia when useful for background information.
3. Give a clear and concise summary.
4. Include the sources you used.
5. Include the names of the tools you actually used.
6. Do not invent sources.
7. If the user asks you to save the research, use the save_to_txt tool.

just return the final answer as json with the structure, no extra words:
{
    "topic" : "...",
    "paper_name" : "...",
    "summary" : "...",
    "sources" : ["..."],
    "tools_used" : ["..."]
}
""",
    tools=tools,
    # response_format=ToolStrategy(ResearchResponse) #this was for google gemini
)

query = "The use multicloud providers in cloud computing."

def run_agent():
    user_ip = input(">> Just paste the topic you want research for....\n")
    try:
        raw_response = agent.invoke({
                'messages':[
                            {
                                'role':'user',
                                'content':user_ip
                            }
                        ]
                })
    except Exception as e:
        print(f"Error Generating response: {e}")

    # print("\nAvailable response keys:")
    # print(raw_response.keys())
    # print("\nRaw Response: ", raw_response,"\n")


    data = raw_response['messages'][-1].content

    # removing markdown fences
    # data = data.strip()
    # if data.startswith("```json"):
    #     data = data[7:]
    # if data.endswith("```"):
    #     data = data[:-3]
    data = re.sub(r"^```(?:json)?\s*","",data.strip())
    data = re.sub(r"\s*```$","", data)


    response = json.loads(data)
    # print("Response: ", response)
    # print(type(response))


    print("\n========== RESEARCH RESPONSE ==========\n")

    print(f"Topic:\n{response["topic"]}")

    print(f"Paper name:\n{response["paper_name"]}")
    print(f"\nSummary:\n{response["summary"]}")

    print("\nTools Used:")
    for tool in response["tools_used"]:
        print(f"- {tool}")

    print("\nSources:")
    for source in response["sources"]:
        print(f"- {source}")

run_agent()
