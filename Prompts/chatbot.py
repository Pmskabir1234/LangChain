from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

messages = [
    SystemMessage(content='You are a helpful assistant')
]

#this is multi turn coversational chatbot with static propmt design

while(True):
    user_input = input("You: ")
    if user_input == 'exit':
        break
    messages.append(HumanMessage(content=user_input))
    res = model.invoke(messages)
    messages.append(AIMessage(content=res.content))
    print('Ai: ', res.content)

print(messages)