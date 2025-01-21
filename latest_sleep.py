import streamlit as st
import pandas as pd
import os
import shutil
import time
from typing import List, Dict
import google.generativeai as genai
from google.api_core import exceptions
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import json
from io import BytesIO

# Constants
UPLOADED_PDFs_DIR = "./uploaded_pdfs_1"
FAISS_INDEX_DIR = "./faiss_indexes"
PDF_METADATA_FILE = "./pdf_metadata.json"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 500

# Rate limit handling constants
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 32  # seconds

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
    
    def handle_rate_limit(self, retry_count: int) -> float:
        """Calculate delay time using exponential backoff"""
        delay = min(INITIAL_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_DELAY)
        return delay
    
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
        """Load all existing vector stores with rate limit handling"""
        retry_count = 0
        for pdf_name in self.pdf_metadata:
            while retry_count <= MAX_RETRIES:
                try:
                    index_path = self.get_index_path(pdf_name)
                    if os.path.exists(index_path):
                        self.vector_stores[pdf_name] = FAISS.load_local(
                            index_path,
                            self.embed_model,
                            allow_dangerous_deserialization=True
                        )
                    break
                except exceptions.ResourceExhausted as e:
                    retry_count += 1
                    if retry_count <= MAX_RETRIES:
                        delay = self.handle_rate_limit(retry_count)
                        st.warning(f"Rate limit reached loading {pdf_name}. Waiting {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        st.error(f"Failed to load index for {pdf_name} after maximum retries")
                except Exception as e:
                    st.error(f"Error loading index for {pdf_name}: {str(e)}")
                    break

    def process_pdf(self, pdf_path: str, pdf_name: str):
        """Process a PDF file with rate limit handling"""
        retry_count = 0
        
        while retry_count <= MAX_RETRIES:
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
                
            except exceptions.ResourceExhausted as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    delay = self.handle_rate_limit(retry_count)
                    st.warning(f"Rate limit reached during PDF processing. Waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    st.error("Maximum retries reached during PDF processing. Please try again later.")
                    return False
                    
            except Exception as e:
                st.error(f"Error in processing PDF: {str(e)}")
                return False

    def get_answer(self, input_question: str, pdf_name: str = None) -> Dict:
        """Get answer for a question with source context and rate limit handling"""
        retry_count = 0
        
        while retry_count <= MAX_RETRIES:
            try:
                if pdf_name and pdf_name not in self.vector_stores:
                    raise ValueError(f"PDF {pdf_name} not found in processed documents")
                
                vector_store = self.vector_stores[pdf_name]
                
                template = """You are a professional Tender Document Analyzer specializing in civil works and construction projects. Your task is to analyze the provided tender document context and determine if specific information is present.

                Context from the tender document:
                {context}

                Question: {question}

                Please follow these guidelines in your analysis:
                1. If the response to the question is found in the context:
                - Quote the relevant text directly from the document
                - Provide any additional relevant details from the context
                3. If not available:
                - Explicitly state that the information was not found in the provided sections
                - If related/partial information exists, mention it but clearly indicate it's not an exact match
                4. If the available information is ambiguos:
                - State this explicitly
                - Explain why it's ambiguous
                - Quote the relevant sections that create the ambiguity

                Remember:
                - Stay strictly factual and based only on the provided context
                - Do not make assumptions about information not present in the document
                - If numbers or specifications are involved, be exact in your quotations
                - Flag any discrepancies or contradictions in the document

                Answer: """

                # Implementation in your code:
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )
                result = qa_chain({"query": input_question})
                
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "page": doc.metadata["page"],
                        "pdf_name": doc.metadata["pdf_name"],
                        "content": doc.page_content
                    })
                
                return {
                    "question": input_question,
                    "answer": result["result"],
                    "sources": sources
                }
                
            except exceptions.ResourceExhausted as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    delay = self.handle_rate_limit(retry_count)
                    st.warning(f"Rate limit reached. Waiting {delay} seconds before retry {retry_count}/{MAX_RETRIES}...")
                    time.sleep(delay)
                    continue
                else:
                    st.error("Maximum retries reached. Please try again later.")
                    return {
                        "question": input_question,
                        "answer": "Error: Rate limit exceeded after maximum retries.",
                        "sources": []
                    }
                    
            except Exception as e:
                st.error(f"Error in getting answer: {str(e)}")
                return {
                    "question": input_question,
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

def create_excel_download(answers):
    """Create and return Excel file bytes from current answers state with separated source columns"""
    export_data = []
    
    for answer in answers:
        # Initialize a row with question and answer
        row = {
            'Question': answer['question'],
            'Answer': answer.get('edited_answer', answer['answer']),
            'Source 1 (Page)': '',
            'Source 1 (Content)': '',
            'Source 2 (Page)': '',
            'Source 2 (Content)': '',
            'Source 3 (Page)': '',
            'Source 3 (Content)': ''
        }
        
        # Add sources to their respective columns
        for idx, source in enumerate(answer['sources'], 1):
            if idx <= 3:  # Handle up to 3 sources
                row[f'Source {idx} (Page)'] = f"Page {source['page']}"
                row[f'Source {idx} (Content)'] = source['content']
        
        export_data.append(row)
    
    # Create DataFrame and set column order
    columns = [
        'Question', 'Answer',
        'Source 1 (Page)', 'Source 1 (Content)',
        'Source 2 (Page)', 'Source 2 (Content)',
        'Source 3 (Page)', 'Source 3 (Content)'
    ]
    df_export = pd.DataFrame(export_data)
    df_export = df_export[columns]  # Reorder columns
    
    # Export to Excel
    excel_buffer = BytesIO()
    df_export.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    return excel_buffer

def main():
    st.set_page_config(page_title="PDF Q&A", layout="wide")
    st.title("PDF Contract Analysis Q&A")
    
    # Initialize session state
    if 'qa_system' not in st.session_state:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.session_state.qa_system = PDFQuestionAnswering(api_key)
    
    if 'answers_generated' not in st.session_state:
        st.session_state.answers_generated = False
    
    if 'generated_answers' not in st.session_state:
        st.session_state.generated_answers = []
    
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
                    st.success(f"Loaded {len(st.session_state.questions)} questions")
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")

        # PDF management
        st.subheader("Available PDFs")
        processed_pdfs = st.session_state.qa_system.get_processed_pdfs()
        if not processed_pdfs:
            st.info("No PDFs available as of now.")
        else:
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

    # Add a reset button if answers were previously generated
    if st.session_state.answers_generated:
        if st.button("Generate New Answers"):
            st.session_state.answers_generated = False
            st.session_state.generated_answers = []
            st.rerun()

    # Generate answers button
    if hasattr(st.session_state, 'questions') and processed_pdfs:
        if not st.session_state.answers_generated and st.button("Generate Answers"):
            progress_bar = st.progress(0)
            st.session_state.generated_answers = []
            
            for i, question in enumerate(st.session_state.questions):
                result = st.session_state.qa_system.get_answer(
                    question, 
                    selected_pdf if 'selected_pdf' in locals() else None
                )
                st.session_state.generated_answers.append(result)
                progress_bar.progress((i + 1) / len(st.session_state.questions))
            
            st.session_state.answers_generated = True
            st.success("All questions processed! You can now edit answers below.")
            st.rerun()

    # Display Q&A boxes and download button if answers are generated
    if st.session_state.answers_generated:
        # Download button at the top
        excel_data = create_excel_download(st.session_state.generated_answers)
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_data,
            file_name="qa_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Display all Q&A boxes
        for i, result in enumerate(st.session_state.generated_answers):
            st.markdown(f"### Q{i+1}: {result['question']}")
            st.markdown("**Answer:**")
            
            # Edit box for each answer
            edited_answer = st.text_area(
                "Edit answer if needed:",
                value=result.get('edited_answer', result['answer']),
                key=f"edit_answer_{i}",
                height=150
            )
            
            # Update the edited answer in session state
            result['edited_answer'] = edited_answer
            
            with st.expander("📄 View Source References"):
                for idx, source in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {idx} - Page {source['page']}**")
                    st.markdown(source['content'])
            
            st.markdown("---")

    else:
        st.info("Please upload both a PDF and questions file to generate answers")

if __name__ == "__main__":
    main()