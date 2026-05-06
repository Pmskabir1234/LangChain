from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_template = ChatPromptTemplate([
    ('system','you are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

with open(r'Prompts\customer_support.txt', 'r') as f:
    chat_history.extend(f.readlines())

# print(chat_history)

user_input = input('You: ') #ask reefund related stuff

prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query' : user_input
})

res = model.invoke(prompt)
print(res.content)