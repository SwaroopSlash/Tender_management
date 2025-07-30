# PDF Contract Analysis Q&A System

A Streamlit-based application that allows users to upload PDF documents and ask questions about their content using Google's Gemini AI. The system is specifically designed for tender document analysis and construction project contracts.

## Features

- **PDF Upload & Processing**: Upload multiple PDF files and process them for question-answering
- **Question Import**: Upload Excel files containing questions to be answered automatically
- **Intelligent Q&A**: Uses Google Gemini AI with RAG (Retrieval Augmented Generation) for accurate answers
- **Source References**: Each answer includes references to specific pages and content from the PDF
- **Answer Editing**: Edit generated answers before exporting
- **Excel Export**: Download results with questions, answers, and source references in Excel format
- **Rate Limit Handling**: Robust handling of API rate limits with exponential backoff
- **API Key Configuration**: Support for user-provided Gemini API keys
- **Professional Prompting**: Specialized prompts for tender document analysis

## Prerequisites

### Required Python Packages

```bash
pip install streamlit
pip install pandas
pip install google-generativeai
pip install langchain-google-genai
pip install langchain-community
pip install langchain
pip install faiss-cpu
pip install PyPDF2
pip install openpyxl
```

### API Requirements

- Google Gemini API key (either configured in Streamlit secrets or provided through the UI)

## Installation & Setup

1. **Clone or download the application file** (`latest_sleep_3.py`)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key** (Optional):
   - Create a `.streamlit/secrets.toml` file in your project directory
   - Add your Google API key:
     ```toml
     GOOGLE_API_KEY = "your-google-api-key-here"
     ```
   - Alternatively, you can enter your API key directly in the application interface

4. **Run the application**:
   ```bash
   streamlit run latest_sleep_3.py
   ```

## Usage Guide

### 1. API Key Configuration
- If you haven't configured the API key in secrets, enter it in the "API Key Configuration" section in the sidebar
- The key will be saved for the current session

### 2. Upload PDF Documents
- Use the "Upload New PDF" section in the sidebar
- Select a PDF file and click "Upload"
- The system will process the PDF and create embeddings for search
- Processed PDFs are stored and can be reused

### 3. Prepare Questions
- Create an Excel file with a column named "question"
- Each row should contain one question you want to ask about the PDF
- Upload the Excel file using "Upload Excel file with questions"

### 4. Generate Answers
- Select a processed PDF from the "Available PDFs" dropdown
- Click "Generate Answers" to process all questions
- The system will show a progress bar and generate answers for each question

### 5. Review and Edit Answers
- Each generated answer is displayed with an editable text area
- You can modify answers as needed
- Source references are available in expandable sections showing the relevant PDF content

### 6. Export Results
- Click "📥 Download Results as Excel" to export all Q&A pairs
- The Excel file includes:
  - Questions and answers (including any edits)
  - Source references with page numbers and content
  - Separate columns for up to 3 source references per question

## File Structure

The application creates the following directories:
- `./uploaded_pdfs_1/` - Stores uploaded PDF files
- `./faiss_indexes/` - Stores FAISS vector indexes for each PDF
- `pdf_metadata.json` - Metadata about processed PDFs

## Configuration Options

### Document Processing Settings
- **Chunk Size**: 3000 characters (larger chunks for better context)
- **Chunk Overlap**: 500 characters (overlap between chunks)
- **Retrieval**: Top 3 most relevant chunks per question

### Rate Limiting
- **Max Retries**: 3 attempts for rate-limited requests
- **Exponential Backoff**: 2, 4, 8... up to 32 seconds delay
- **Automatic Recovery**: System handles API limits gracefully

## Specialized Features for Tender Analysis

The system includes specialized prompting for tender document analysis:

- **Factual Analysis**: Quotes relevant text directly from documents
- **Ambiguity Detection**: Identifies and explains unclear information
- **Missing Information**: Clearly states when information is not found
- **Exact Quotations**: Preserves numbers and specifications exactly as written
- **Contradiction Detection**: Flags discrepancies in documents

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**: 
   - The system automatically handles rate limits with delays
   - If persistent, wait a few minutes before retrying

2. **API Key Issues**:
   - Ensure your Gemini API key is valid and has sufficient quota
   - Check that the key has access to the required models

3. **PDF Processing Errors**:
   - Ensure PDFs are text-based (not scanned images)
   - Check that PDFs are not password-protected

4. **Excel Upload Issues**:
   - Verify the Excel file has a column named "question"
   - Ensure the file is in .xlsx format

### Memory Management

- The system stores vector indexes on disk to preserve memory
- Each PDF gets its own FAISS index for efficient retrieval
- Processed PDFs persist between sessions

## Technical Architecture

- **Frontend**: Streamlit web interface
- **LLM**: Google Gemini 1.5 Flash for question answering
- **Embeddings**: Google Embedding-001 model
- **Vector Store**: FAISS for similarity search
- **Document Processing**: LangChain for PDF loading and text splitting
- **Rate Limiting**: Exponential backoff strategy

## Version Information

This README corresponds to `latest_sleep_3.py` - the most feature-complete version with:
- User API key input support
- Enhanced Excel export with separate source columns
- Professional tender document analysis prompts
- Robust rate limit handling
- Session persistence for processed PDFs
