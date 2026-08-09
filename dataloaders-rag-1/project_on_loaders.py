# This project is a little hands-on on dataloaders where u will compare your github activity with other using usernames
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


load_dotenv()

llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Analyze the given {doc} carefully and do the following {task}',
    input_variables=['doc','task']
)

# Custom data loader for Github
class GithubProfileLoader(BaseLoader):

    def __init__(self, username: str, token: str | None = None):
        self.username = username
        self.token = token

    def lazy_load(self):
        url = f"https://api.github.com/users/{self.username}"

        headers = {
                "Accept":"application/vnd.github+json"
            }

        if self.token:
                headers['Authorization'] = f" Bearer {self.token}"

        response = requests.get(
                url,
                headers=headers,
                timeout=5
            )
        response.raise_for_status()
        data = response.json()
        yield Document(
                page_content=self._format_profile(data),
                metadata = {
                    "source":url,
                    "username":self.username,
                    "type":"github_profile"
                }
            )
    def _format_profile(self, data:dict) -> str:
                return f"""
Github Profile

Username : {data.get("login")}
Name : {data.get("name")}
Bio : {data.get("bio")}
Location : {data.get("location")}
Followers : {data.get("followers")}

Public repositories : {data.get('public_repos')}
Public gists: {data.get("public_gists")}

Created: {data.get("created_at")}
Updated: {data.get("updated_at")}
"""

