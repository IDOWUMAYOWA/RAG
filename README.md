# 📚 RAG Q&A Application  
### Retrieval-Augmented Generation with LangChain, Chroma, OpenAI, and Gradio  

🔗 **GitHub Repository:** https://github.com/IDOWUMAYOWA/RAG  

---

## 🚀 Overview  

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions about a document and receive answers grounded in that document’s content.

The system performs the following steps:

1. Load a document  
2. Split it into chunks  
3. Generate embeddings using OpenAI  
4. Store embeddings in a persistent Chroma vector database  
5. Retrieve the most relevant chunks  
6. Use an LLM to generate a contextual answer  
7. Display results in a Gradio web interface  

This demonstrates an end-to-end production-style RAG workflow.

---

## 🖼️ App Screenshot  

> Add your screenshot image to the repository (for example: `assets/app.png`) and make sure the file path matches below.

![RAG App Screenshot](screenshot.pmgg)

---

## 🏗 Architecture  

```
Document
  ↓
Text Splitting (RecursiveCharacterTextSplitter)
  ↓
Embeddings (OpenAIEmbeddings)
  ↓
Chroma Vector Store (Persistent)
  ↓
Retriever (Top-K Similarity Search)
  ↓
LLM (ChatOpenAI)
  ↓
Answer + Sources
```

---

## 🗂 Project Structure  

```
RAG/
├── src/
│   └── utils.py
├── research/
│   └── chroma_db/            # persistent vector DB (git-ignored)
├── app.py                    # Gradio UI
├── eleven_madison_park_data.txt
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Setup  

### 1. Clone the repository

```bash
git clone https://github.com/IDOWUMAYOWA/RAG.git
cd RAG
```

### 2. Create and activate a virtual environment

Using conda:

```bash
conda create -n rag-env python=3.10
conda activate rag-env
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application  

From the project root:

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:7860
```

---

## 🧠 How to Use  

1. Enter a question in the input box.  
2. Optionally adjust:
   - Top-k documents (number of retrieved chunks)
   - Temperature (model creativity level; lower is more factual)
3. Click **Ask**.  
4. View:
   - Generated answer  
   - Sources  
   - Optional source snippets  

---

## 📦 Persistent Chroma Database  

The Chroma database is stored locally at:

```
research/chroma_db/
```

This directory is excluded via `.gitignore`.

If you change:
- The source document  
- Chunk size  
- Embedding model  

Delete the `research/chroma_db/` folder and rebuild.

---

## 🔧 Optional Configuration  

Environment variables:

- `RAG_DATA_FILE` (default: `eleven_madison_park_data.txt`)  
- `CHROMA_DIR` (default: `research/chroma_db`)  

Example:

```bash
export CHROMA_DIR="research/chroma_db"
export RAG_DATA_FILE="eleven_madison_park_data.txt"
python app.py
```

---

## 🛠 Tech Stack  

- Python 3.10  
- LangChain (modular packages)  
- OpenAI API  
- ChromaDB  
- Gradio  
- python-dotenv  

---

## 📄 License  

MIT License  

---

## 👤 Author  

GitHub: https://github.com/IDOWUMAYOWA  

If you found this project useful or interesting, feel free to ⭐ the repository.
