from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv


load_dotenv()

# text generation model 
llm = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# embedding model 
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
embed_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# load knowledge data path and use data loader
DOC_PATH = r"C:\Users\saaad kabir\Desktop\Research Papers\Plant_Disease_Detection_and_Classification_by_Deep_LearningA_Review.pdf"
loader = PyPDFLoader(DOC_PATH)
docs = loader.load()

# split doc in chunks
splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150,
            separators=''
)

splitted_text = splitter.split_documents(docs)

# vector database
vector_store = Chroma(
    embedding_function=embed_model,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

vector_store.add_documents(splitted_text)


prompt = PromptTemplate(
    template="You are a helpful assistant. Answer {question} only with the context of {context}. In if case if you don't know the answer, just say I don't know the answer.",
    input_variables=['question','context']

)

while True:
    question = input("Ask question: ")

    if question == 'q':
        break
    similiarity = vector_store.similarity_search(query=question, k=1)
    context = similiarity[0].page_content

    parser = StrOutputParser()

    chain = prompt | model | parser

    res = chain.invoke({'question':question, 'context':context})

    print(res)