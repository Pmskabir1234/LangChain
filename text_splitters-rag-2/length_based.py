# text splitting and doc splitting

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import CharacterTextSplitter

PDF_PATH = 'Developer Memory OS.pdf'

text = """
Classical machine learning (ML) focuses on algorithms like linear regression,
 decision trees, and support vector machines, which rely heavily on feature 
 engineering and statistical assumptions to perform tasks such as classification 
 or prediction. Modern ML, driven by deep learning and neural networks, emphasizes 
 representation learning, where models automatically extract features from raw data, 
 enabling breakthroughs in vision, language, and speech. Classical ML is efficient 
 on small datasets, while modern ML thrives on massive data and computational power, 
 reshaping AI applications across industries.
"""

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator= ''

)

result = splitter.split_text(text)

# print(result)

loader = PyPDFLoader(PDF_PATH)

docs = loader.load() #this will return the pdf data as doc objects

doc_chunks = splitter.split_documents(docs)

for chunks in doc_chunks:
    print(chunks.page_content)
