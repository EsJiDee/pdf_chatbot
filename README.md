# 📄 Chat with PDF using Google Gemini

A Streamlit web application that allows users to upload multiple PDF documents, process them, and ask natural language questions about their content. The app uses Google's Gemini 2.0 Flash Lite model for fast, efficient answering and LangChain for document processing.

## 🌟 Features
* **Multiple PDF Uploads:** Process several PDF documents simultaneously.
* **Smart Text Chunking:** Uses recursive character splitting to handle large documents efficiently.
* **Vector Search:** Utilizes FAISS (Facebook AI Similarity Search) and Gemini Embeddings for rapid and accurate context retrieval.
* **Conversational UI:** Built with Streamlit for a clean, interactive web interface.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **PDF Processing:** PyPDF2
* **Orchestration:** LangChain
* **Embeddings:** Google Generative AI (`models/gemini-embedding-001`)
* **LLM:** Google Generative AI (`gemini-2.0-flash-lite`)
* **Vector Store:** FAISS

## 📁 Project Structure & Setup Instructions

*Note: This `README.md` file should be placed in the root directory of your project, alongside your `app.py`, `.env`, and `.gitignore` files.*

### 1. Clone the repository
```bash
git clone [https://github.com/EsJiDee/pdf_chatbot.git](https://github.com/EsJiDee/pdf_chatbot.git)
cd pdf_chatbot
```

### 2. Create and activate a virtual environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install streamlit PyPDF2 langchain langchain-google-genai langchain-community faiss-cpu python-dotenv
```

### 4. Set up your Google API Key
Create a `.env` file in the root directory of the project and add your Google API key:
```env
GOOGLE_API_KEY="your_api_key_here"
```
*(You can get a free API key from [Google AI Studio](https://aistudio.google.com/))*

## 💻 Usage

1. **Start the Streamlit server:**
```bash
streamlit run app.py
```

2. **Interact with the app:**
   * Open the local URL provided in your terminal (usually `http://localhost:8501`).
   * Upload one or more PDF files using the sidebar menu.
   * Click **"Process PDFs"** and wait for the success message.
   * Type your question in the main chat input to query your documents!

## ⚠️ Important Notes
* **Security:** Never upload your `.env` file to GitHub. Ensure it is listed in your `.gitignore` file.
* **Index Updates:** If you change the embedding model in the future, you must delete the local `faiss_index` folder and re-process your PDFs to avoid dimension mismatch errors.