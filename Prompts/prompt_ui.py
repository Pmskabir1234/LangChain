from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate


load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header("Research Summarizer")
st.subheader("Summarize any research article you want")


# static prompt -- too much control to user, may include lack of info, typo, uncertainty in prompt
# user_input = st.chat_input('type what you want to summarize...')  
# if user_input:
#     res = model.invoke(user_input)
#     st.write(res.content)

paper_input = st.selectbox("Select Research Paper Name",
                            ['Attention is all you need',
                            'BERT: Pre-training of deep bidirectional transformers',
                            'GPT-3: Language Models are Few-shot Learners',
                            'Diffusion Models Beat GANs on Image Synthesis'])
style_input = st.selectbox("Select Explanation Style", 
                           ['Beginner Friendly',
                            'Technical',
                            'Analogical',
                            'Code Oriented',
                            'Mathematical'])
length_input = st.selectbox("Select Explanation Length",
                            ['Short(1-2 paragraphs)',
                            'Medium(3-5 paragraphs)',
                            'Long(Detailed explanation)'])


# prompt template - dynamic prompt
template = PromptTemplate(
    template= """

    Summarize the research paper titled "{paper}" with the following specifications:
    Explantion style : {style},
    Explanation length : {length}
    Ensure summary is clear, accurate, and aligned with the provided style and length.


    """,
input_variables=['paper','style','length'],
validate_template=True
)

# filling placeholders
prompt = template.invoke({
    'paper': paper_input,
    'style': style_input,
    'length':length_input
})

if st.button('Summarize'):
    res = model.invoke(prompt)
    st.write(res.content)  