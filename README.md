# RAG System – AI/ML Knowledge Base Chatbot
## | Retrieval-Augmented Generation

---

## Project Structure

```
rag_assignment/
├── dataset/
│   ├── machine_learning.txt      # Wikipedia: Machine Learning
│   ├── deep_learning.txt         # Wikipedia: Deep Learning
│   ├── neural_networks.txt       # Wikipedia: Neural Networks
│   ├── nlp.txt                   # Wikipedia: NLP
│   └── llm.txt                   # Wikipedia: Large Language Models
├── rag_system.ipynb              # Main notebook (Parts 1–6)
├── app.py                        # Streamlit UI
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

##  Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API Key

1. To run the rag_system.ipynb put your API key in CELL 2: IMPORTS & API CONFIGURATION
os.environ["YOUR_API_KEY"] = "your-actual-key-here" 
Also put your key in STEP 5.1: Configure the Google Gemini Client under environment variable

2. To run the application open the app.py file and put your API key right under the imports sections. 
ANYLLM_API_KEY = "paste-your-key-here"

### 3. Run the Jupyter Notebook
```bash
jupyter notebook rag_system.ipynb
```
Run all cells top to bottom. This will:
- Load and preprocess the dataset
- Generate embeddings
- Build the FAISS index (saved as `faiss_index.bin`)
- Run the full evaluation (7 questions)

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
Open browser at http://localhost:8501

Type a question, and click "Search & Generate Answer."

