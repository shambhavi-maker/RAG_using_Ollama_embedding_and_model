# RAG using Ollama embedding and model

This repository implements **Retrieval-Augmented Generation (RAG)** using **Ollama embeddings: nomic-embed-text and model:gemma-3-4b-it-GGUF:Q4_K_M**, enabling efficient retrieval of external knowledge to enhance AI-generated responses.

## 📌 Features
- **Web-Based Document Loading:** Uses `WebBaseLoader` to fetch content dynamically.
- **PDF Processing:** Supports `PyPDFLoader` for handling document-based data.
- **Text Splitting:** Utilizes `CharacterTextSplitter` for chunking large texts.
- **Vector Storage with Chroma:** Stores embeddings using `Chroma` for efficient retrieval.
- **Context-Aware Response Generation:** Enhances responses by incorporating external knowledge via RAG.
- **Pre-RAG vs. Post-RAG Comparison:** Demonstrates the impact of retrieved context on generated responses.

### 🔍 How It Works
1. **Document Loading:**  
   - Fetches web-based data via `WebBaseLoader`.  
   - Supports PDF ingestion using `PyPDFLoader`.  

2. **Text Processing:**  
   - Splits content into manageable chunks using `CharacterTextSplitter`.  
   - Optimizes retrieval efficiency for better contextual representation.  

3. **Vector Storage & Retrieval:**  
   - Uses `OllamaEmbeddings` to convert text into vector representations.  
   - Stores embeddings in `Chroma` for fast and accurate retrieval.  

4. **Context-Aware Response Generation:**  
   - Utilizes retrieved knowledge to enrich AI responses.  
   - Demonstrates pre-RAG vs. post-RAG effectiveness.  

## 🚀 Setup Instructions
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required dependencies (`langchain`, `ollama`, `chromadb`, etc.)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-repo/RAG_using_Ollama_embedding_and_model.git
cd RAG_using_Ollama_embedding_and_model
pip install -r requirements.txt
```
### Ollama setup
```bash
ollama pull hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M
ollama run hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M
ollama pull nomic-embed-text
ollama run nomic-embed-text
```

### Usage
Run the main script to load documents, store embeddings, and generate responses:
```bash
python main.py
```

## 🔗 API Integration  
This project supports API-based interaction for external applications. You can integrate it via a **REST API** or **LangChain's callable functions**:

- **Exposing the RAG pipeline via FastAPI:**  
```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/query/")
async def rag_response(question: str):
    return after_rag_chain.invoke(question)
```
Run the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Now, external applications can send POST requests to `http://localhost:8000/query/` to retrieve responses.

## ⚙️ Custom Model Configuration
- **Change Ollama Model:** Modify the `ChatOllama` instantiation to use different models:
```python
model_local = ChatOllama(model="hf.co/unsloth/gemma-7b-it-GGUF:Q5_K_M")
```
- **Adjust Embedding Model:** Switch embedding models for different retrieval performance:
```python
embedding_model = OllamaEmbeddings(model="openai-ada")
```

## 🧠 Fine-Tuning the Retrieval Mechanism
Enhance retrieval accuracy by tweaking `retriever` parameters:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Return top 5 matches
```
Or apply **hybrid search** combining keyword and vector retrieval:
```python
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"lambda_mult": 0.3})
```

## 📊 Benchmarking & Performance Comparisons
### 🔬 Evaluation Metrics
To assess the effectiveness of RAG with Ollama, we benchmark the system using:
- **Response Accuracy:** Measures how well the model retrieves relevant information.
- **Latency:** Evaluates response time for queries.
- **Token Efficiency:** Tracks the number of tokens used per response.
- **Retrieval Success Rate:** Determines how often the correct context is fetched.

### 📈 Benchmarking Results
Recent evaluations indicate that Ollama-based RAG models outperform traditional generative models in:
- **Contextual Accuracy:** Achieves higher precision in responses compared to standard LLMs.
- **Speed & Efficiency:** Faster retrieval and response generation due to optimized embeddings.
- **Reduced Hallucination:** Significantly lowers incorrect or misleading responses.

For detailed benchmarking results, refer to [Ollama LLM Model Evaluation & Benchmarking Tool](https://github.com/valdecircarvalho/ollama_eval) and [Hugging Face RAG Evaluation](https://huggingface.co/learn/cookbook/rag_evaluation).

## 💡 Use Cases
- AI chatbots leveraging dynamic external knowledge.
- Automated research tools fetching relevant information.
- Q&A systems with improved accuracy and reduced hallucinations.

## 🔗 References
- [Ollama](https://ollama.com/)
- [Ollama Embedding](https://ollama.com/library/nomic-embed-text)
- [Ollama Model](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF)
- [ChromaDB](https://www.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

## 📜 License  
This project is licensed under the **MIT License**.

Feel free to contribute! 🚀😊

