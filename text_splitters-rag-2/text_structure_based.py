# it follows a hierarchy of paragraph ('\n' '\n'), line break ('\n'), then ' ', then ''
# based on the allowed chunk_size, all these are reciursively done to make sure that
# information is not left midway
# hence it i called 'recursive character text splitter

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

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

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

result = splitter.split_text(text)
print(result)

# we can extend the workflow for code as well
# where we just need to pass a language parameter
# and it supports PYTHON,MARKDOWN,JAVA,PHP and so on