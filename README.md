# QA-Agent
This project builds an AI agent that answers product-related questions using your help documentation. It uses:

- FAISS for vector search  
- HuggingFace Embeddings  
- LlamaIndex for orchestration  
- Gemini 1.5 Flash for natural language responses

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/help-docs-agent.git
cd help-docs-agent
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables
Create a `.env` file and add your Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

Alternatively, you can hardcode it into the script for quick testing:
```python
GEMINI_API_KEY = "your_api_key_here"
```

---

## Dependencies

```text
llama-index
faiss-cpu
sentence-transformers
python-dotenv
bs4
selenium
streamlit
```

Install everything manually:
```bash
pip install llama-index faiss-cpu sentence-transformers python-dotenv bs4 selenium streamlit
```

---


## Design Decisions

- **FAISS Vector Store:** Chosen for fast similarity search of embedded text.
- **MiniLM-L6-v2 Embeddings:** Lightweight, fast, and high-quality sentence embeddings.
- **LlamaIndex Tools:** Allows wrapping query engines into reusable tools.
- **FunctionCallingAgent:** Empowers the Gemini model to choose tools dynamically for better relevance.
- **System Prompting:** Guides the model to use the documentation when applicable, fallback if not found.

---

## Future Improvements

- [ ] Add persistent FAISS storage.
- [ ] Support multiple document sources.
- [ ] Enable document uploads via UI.
- [ ] Improve fallback responses with LLM summaries.

---

## Contributions Welcome!
Feel free to fork the repo, open issues, or submit PRs.

---

## 📄 License

MIT License
