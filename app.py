import streamlit as st
import pandas as pd
import os
from typing import List, Dict
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import shutil
from pinecone import Pinecone, ServerlessSpec

# Constants
PINECONE_INDEX_NAME = "chatpdf"
UPLOADED_PDFs_DIR = "./uploaded_pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class PDFQuestionAnswering:
    def __init__(self, google_api_key: str, pinecone_api_key: str):
        self.google_api_key = google_api_key
        genai.configure(api_key=self.google_api_key)
        
        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.google_api_key,
            temperature=0.3
        )
        
        # Create directory for uploaded PDFs
        os.makedirs(UPLOADED_PDFs_DIR, exist_ok=True)
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists, if not create it
        if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  # Dimension for all-MiniLM-L6-v2 model
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2"
                )
            )

        # Initialize vector store
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=self.embed_model,
            text_key="text"
        )
        
    def process_pdf(self, pdf_path: str, pdf_name: str):
        """Process a PDF file and store its embeddings"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add page numbers to metadata
        for i, doc in enumerate(documents):
            doc.metadata["page"] = i + 1
            doc.metadata["pdf_name"] = pdf_name
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.from_documents(
            documents=splits,
            embedding=self.embed_model,
            index_name=PINECONE_INDEX_NAME,
        )
        
    def get_answer(self, question: str, pdf_name: str = None) -> Dict:
        """Get answer for a question with source context"""
        # Create search filter if pdf_name is provided
        search_filter = {"pdf_name": pdf_name} if pdf_name else None
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3, "filter": search_filter}
            ),
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": question})
        
        # Format response
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
    
    def delete_pdf(self, pdf_name: str):
        """Delete a PDF and its associated embeddings"""
        # Delete from vector store
        self.vector_store.delete(filter={"pdf_name": pdf_name})
        
        # Delete PDF file
        pdf_path = os.path.join(UPLOADED_PDFs_DIR, pdf_name)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

def main():
    st.set_page_config(page_title="PDF Q&A", layout="wide")
    st.title("PDF Contract Analysis Q&A")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        st.session_state.qa_system = PDFQuestionAnswering(api_key, pinecone_api_key)
    
    # Sidebar for PDF management
    with st.sidebar:
        st.header("PDF Management")
        
        # PDF upload
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            pdf_name = uploaded_file.name
            pdf_save_path = os.path.join(UPLOADED_PDFs_DIR, pdf_name)
            
            try:
                st.session_state.qa_system.process_pdf(tmp_file_path, pdf_name)
                shutil.copy2(tmp_file_path, pdf_save_path)
                st.success(f"Successfully processed {pdf_name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                os.unlink(tmp_file_path)

        # Excel upload
        st.header("Upload Questions")
        questions_file = st.file_uploader("Upload Excel file with questions", type="xlsx")
        
        if questions_file:
            try:
                df = pd.read_excel(questions_file)
                if 'question' not in df.columns:
                    st.error("Excel file must have a 'question' column")
                else:
                    st.session_state.questions = df['question'].tolist()
                    st.success(f"Loaded {len(st.session_state.questions)} questions")
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
        
        # PDF selection and deletion
        st.subheader("Manage PDFs")
        pdfs = os.listdir(UPLOADED_PDFs_DIR)
        if pdfs:
            selected_pdf = st.selectbox("Select PDF for Q&A", pdfs)
            if st.button("Delete Selected PDF"):
                st.session_state.qa_system.delete_pdf(selected_pdf)
                st.experimental_rerun()
        else:
            st.info("No PDFs uploaded yet")
    
    st.header("Generate Answers")
    if st.session_state.get('questions') and pdfs:
        if st.button("Generate Answers"):
            progress_bar = st.progress(0)
            
            # Create a container for results
            results_container = st.container()
            
            with results_container:
                for i, question in enumerate(st.session_state.questions):
                    result = st.session_state.qa_system.get_answer(
                        question, 
                        selected_pdf if selected_pdf else None
                    )

                    # Create a card-like container for each Q&A
                    st.markdown("""
                    <div style='padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin-bottom: 1rem;'>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"### Q{i+1}: {question}")
                    st.markdown("**Answer:**")
                    st.write(result["answer"])
                    
                    # Display sources in a collapsible section
                    with st.expander(f"📄 View Source References"):
                        for idx, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {idx} - Page {source['page']}**")
                            st.markdown("```")
                            st.markdown(source['content'])
                            st.markdown("```")
                            if idx < len(result["sources"]):
                                st.markdown("---")  # Separator between sources
                    
                    # Add spacing between questions
                    st.markdown("---")
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(st.session_state.questions))
            
            st.success("All questions processed!")
    else:
        st.info("Please upload both a PDF and questions file to generate answers")

if __name__ == "__main__":
    main()