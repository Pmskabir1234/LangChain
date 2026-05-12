from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    model='deepseek-ai/DeepSeek-V4-Flash',
    task='text-generation'
)

llm2 = HuggingFaceEndpoint(
    model='google/gemma-4-31B-it',
    task='test-generation'
)

model1 = ChatHuggingFace(llm=llm2)
model2 = ChatHuggingFace(llm=llm1)

text = """

Here’s a detailed explanation of the **Isolation Forest model**, structured into clear sections without emojis:

---

### Overview
Isolation Forest is an ensemble-based anomaly detection algorithm that isolates outliers instead of modeling normal data. It builds multiple decision trees (called Isolation Trees) using random splits. The fundamental idea is that anomalies are few and different, so they are easier to isolate compared to normal points. This makes the algorithm efficient and well-suited for high-dimensional datasets.

---

### Use Cases
- **Fraud detection:** Spotting unusual financial transactions.  
- **Cybersecurity:** Identifying abnormal network traffic or intrusion attempts.  
- **Manufacturing:** Detecting defective items in production lines.  
- **Healthcare:** Recognizing rare medical conditions or abnormal patient data.  
- **Finance:** Flagging irregular market behaviors or trades.  

---

### Working Principle
1. **Random partitioning:** Each tree is built by randomly selecting a feature and a split value.  
2. **Isolation process:** Data points are recursively split until they are isolated.  
3. **Path length:** The number of splits required to isolate a point is recorded.  
   - Outliers are isolated quickly, resulting in shorter path lengths.  
   - Normal points require more splits, leading to longer path lengths.  
4. **Anomaly score:** The average path length across all trees is used to compute an anomaly score.  
   - Shorter average path length → higher anomaly score → more likely to be an outlier.  

---

### Key Parameters
- **n_estimators:** Number of trees in the forest. More trees improve stability but increase computation.  
- **max_samples:** Number of samples used to build each tree. Smaller samples increase randomness.  
- **contamination:** Proportion of expected anomalies in the dataset. Helps set the threshold for classification.  
- **max_features:** Number of features used per tree. Controls efficiency and diversity.  
- **random_state:** Ensures reproducibility of results.  

---

### Advantages
- Efficient for large, high-dimensional datasets.  
- Does not assume any distribution of data.  
- Outliers are naturally easier to isolate, making detection fast.  
- Works well even when anomalies are rare.  

---

### Limitations
- Sensitive to contamination parameter; mis-specification can affect results.  
- Less interpretable compared to statistical models.  
- Performance may drop if anomalies are clustered together rather than scattered.  

---

### Example in Python
```python
from sklearn.ensemble import IsolationForest

# Sample dataset
X = [[-1.1], [0.3], [0.5], [100]]

# Train model
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# Predict anomalies (-1 = anomaly, 1 = normal)
print(clf.predict(X))
```
Output: `[1, 1, 1, -1]` → The point `100` is flagged as an anomaly.

---

### Key Takeaway
Isolation Forest is a robust and scalable anomaly detection method that isolates outliers by leveraging random partitioning. It is particularly effective in domains where anomalies are rare but critical to detect, such as fraud detection, cybersecurity, and healthcare.

---

Would you like me to expand this further into a **step-by-step mathematical explanation** of how the anomaly score is calculated (including the formula for expected path length)? That would give you a deeper technical understanding.

"""

template1 = PromptTemplate(
    template="Generate Short note on the given text: \n {text}",
    input_variables=['text']
)

template2 = PromptTemplate(
    template="Generate 5 important quiz questions from the following text: \n {text}",
    input_variables=['text']
)

template3 = PromptTemplate(
    template="Merge the following text and quiz questions:\n {text} \n {quiz}",
    input_variables=['text', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'text': template1 | model1 | parser,
    'quiz': template2 | model2 | parser
    # make sure you name the chains here as per variable names in template3
})

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

res = chain.invoke({'text':text})

print(res, "\n")
chain.get_graph().print_ascii()