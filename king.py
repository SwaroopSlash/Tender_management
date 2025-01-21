import streamlit as st
import pandas as pd
import os
import shutil
from typing import List, Dict
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import tempfile
import json

# Constants
UPLOADED_PDFs_DIR = "./uploaded_pdfs_1"
FAISS_INDEX_DIR = "./faiss_indexes"  # Directory to store individual FAISS indexes
PDF_METADATA_FILE = "./pdf_metadata.json"  # File to store PDF metadata
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class PDFQuestionAnswering:
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings and LLM
        self.embed_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.3
        )
        
        # Create necessary directories
        os.makedirs(UPLOADED_PDFs_DIR, exist_ok=True)
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        
        # Load or initialize PDF metadata
        self.pdf_metadata = self.load_pdf_metadata()
        
        # Initialize vector stores for each PDF
        self.vector_stores = {}
        self.load_all_vector_stores()
    
    def load_pdf_metadata(self):
        """Load PDF metadata from file or create new if doesn't exist"""
        if os.path.exists(PDF_METADATA_FILE):
            with open(PDF_METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_pdf_metadata(self):
        """Save PDF metadata to file"""
        with open(PDF_METADATA_FILE, 'w') as f:
            json.dump(self.pdf_metadata, f)
    
    def get_index_path(self, pdf_name: str):
        """Get the path for a PDF's FAISS index"""
        return os.path.join(FAISS_INDEX_DIR, pdf_name.replace('.pdf', ''))
    
    def load_all_vector_stores(self):
        """Load all existing vector stores"""
        for pdf_name in self.pdf_metadata:
            index_path = self.get_index_path(pdf_name)
            if os.path.exists(index_path):
                try:
                    self.vector_stores[pdf_name] = FAISS.load_local(
                        index_path,
                        self.embed_model,
                        allow_dangerous_deserialization = True
                    )
                except Exception as e:
                    st.error(f"Error loading index for {pdf_name}: {str(e)}")

    def process_pdf(self, pdf_path: str, pdf_name: str):
        """Process a PDF file and store its embeddings"""
        try:
            # Check if PDF has already been processed
            if pdf_name in self.pdf_metadata:
                st.info(f"{pdf_name} is already processed and available")
                return True

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                doc.metadata["page"] = i + 1
                doc.metadata["pdf_name"] = pdf_name
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            
            # Create and save vector store for this PDF
            vector_store = FAISS.from_documents(splits, self.embed_model)
            index_path = self.get_index_path(pdf_name)
            vector_store.save_local(index_path)
            
            # Update metadata and save
            self.vector_stores[pdf_name] = vector_store
            self.pdf_metadata[pdf_name] = {
                "pages": len(documents),
                "date_added": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_pdf_metadata()
            
            return True
            
        except Exception as e:
            st.error(f"Error in processing PDF: {str(e)}")
            return False

    def get_answer(self, question: str, pdf_name: str = None) -> Dict:
        """Get answer for a question with source context"""
        try:
            if pdf_name and pdf_name not in self.vector_stores:
                raise ValueError(f"PDF {pdf_name} not found in processed documents")
            
            vector_store = self.vector_stores[pdf_name]
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "page": doc.metadata["page"],
                    "pdf_name": doc.metadata["pdf_name"],
                    "content": doc.page_content
                })
            
            return {
                "question": question,
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            st.error(f"Error in getting answer: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": []
            }
    
    def delete_pdf(self, pdf_name: str):
        """Delete a PDF and its associated data"""
        try:
            # Delete FAISS index
            index_path = self.get_index_path(pdf_name)
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            
            # Delete from vector stores dict
            if pdf_name in self.vector_stores:
                del self.vector_stores[pdf_name]
            
            # Delete from metadata
            if pdf_name in self.pdf_metadata:
                del self.pdf_metadata[pdf_name]
                self.save_pdf_metadata()
            
            # Delete PDF file
            pdf_path = os.path.join(UPLOADED_PDFs_DIR, pdf_name)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                
            return True
        except Exception as e:
            st.error(f"Error in deleting PDF: {str(e)}")
            return False

    def get_processed_pdfs(self):
        """Get list of processed PDFs with metadata"""
        return self.pdf_metadata

def main():

    st.set_page_config(page_title="PDF Q&A", layout="wide")
    st.title("PDF Contract Analysis Q&A")
    
    # Initialize QA system
    if 'qa_system' not in st.session_state:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.session_state.qa_system = PDFQuestionAnswering(api_key)
    
    with st.sidebar:
        st.header("PDF Management")
        
        # PDF upload
        uploaded_file = st.file_uploader("Upload New PDF", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            if st.sidebar.button("Upload"):
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    pdf_name = uploaded_file.name
                    pdf_save_path = os.path.join(UPLOADED_PDFs_DIR, pdf_name)
                    
                    try:
                        if st.session_state.qa_system.process_pdf(tmp_file_path, pdf_name):
                            shutil.copy2(tmp_file_path, pdf_save_path)
                            st.success(f"Successfully processed {pdf_name}")
                    finally:
                        os.unlink(tmp_file_path)

        # Excel upload
        st.header("Upload Questions")
        questions_file = st.file_uploader("Upload Excel file with questions", type="xlsx", key="excel_uploader")
        
        if questions_file:
            try:
                df = pd.read_excel(questions_file)
                if 'question' not in df.columns:
                    st.error("Excel file must have a 'question' column")
                else:
                    st.session_state.questions = df['question'].tolist()
                    st.session_state.df = df
                    st.session_state.df['answer'] = ''
                    st.session_state.df['reference'] = ''

                    st.success(f"Loaded {len(st.session_state.questions)} questions")
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
        
        # PDF management with metadata display
        st.subheader("Available PDFs")
        processed_pdfs = st.session_state.qa_system.get_processed_pdfs()
        if not any(processed_pdfs): 
            st.info("No PDFs available as of now.")
    
        if processed_pdfs:
            pdf_names = list(processed_pdfs.keys())
            selected_pdf = st.selectbox("Select PDF for Q&A", pdf_names)
            
            if selected_pdf:
                metadata = processed_pdfs[selected_pdf]
                st.write(f"Pages: {metadata['pages']}")
                st.write(f"Added: {metadata['date_added']}")
                
                if st.button("Delete Selected PDF"):
                    if st.session_state.qa_system.delete_pdf(selected_pdf):
                        st.success(f"Successfully deleted {selected_pdf}")
                        
                        st.rerun()

                    
    
    # Main content area
    st.header("Generate Answers")
    if hasattr(st.session_state, 'questions') and processed_pdfs:
        if st.button("Generate Answers"):
            progress_bar = st.progress(0)
            
            results_container = st.container()
            
            with results_container:
                for i, question in enumerate(st.session_state.questions):
                    result = st.session_state.qa_system.get_answer(
                        question, 
                        selected_pdf if selected_pdf else None
                    )

                    st.session_state.df.loc[i + 1, 1] = result["answer"]
                    st.session_state.df.loc[i + 1, 2] = result["answer"]

                    st.markdown(f"### Q{i+1}: {question}")
                    st.markdown("**Answer:**")
                    st.write(result["answer"])
                    with st.expander(f"📄 View Source References"):
                        for idx, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {idx} - Page {source['page']}**")
                            st.markdown(source['content'])
            
                    progress_bar.progress((i + 1) / len(st.session_state.questions))

            st.session_state.df.to_excel('generated_response.xlsx')
            st.success("All questions processed!")
    else:
        st.info("Please upload both a PDF and questions file to generate answers")

if __name__ == "__main__":
    main()