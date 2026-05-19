import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    if not text.strip():
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Updated to the current supported embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question based ONLY on the following context. 
    If the answer isn't in the context, say "I couldn't find relevant information in the documents.":
    
    Context: \n{context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    # Change the model here to a lower tier flash model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Error: Please process PDF documents first!")
        return

    try:
        # Updated to match the index creation model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain({"input_documents": docs, "question": user_question})
        st.write("Reply:", response["output_text"])
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.title("📄 Chat with PDF")
    
    # Initialize session state
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("📁 Upload PDF Documents")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("📚 Process PDFs"):
            if pdf_docs:
                with st.spinner("Reading and indexing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("Failed to extract text from PDFs. The files may be scanned images.")
                        return
                        
                    text_chunks = get_text_chunks(raw_text)
                    
                    if not text_chunks:
                        st.error("No text chunks generated. Please check your PDF files.")
                        return
                        
                    get_vector_store(text_chunks)
                    st.session_state.processed = True
                    st.success("✅ PDFs processed and ready for Q&A!")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

    st.subheader("💬 Ask a question from your uploaded PDFs")
    user_question = st.text_input("Type your question here:")
    
    if st.button("🎯 Get Answer"):
        if not st.session_state.processed:
            st.error("Please process PDF documents before asking questions!")
            return
            
        if not user_question.strip():
            st.warning("⚠️ Please enter a valid question.")
            return
            
        with st.spinner("Analyzing documents..."):
            user_input(user_question)

if __name__ == "__main__":
    main()