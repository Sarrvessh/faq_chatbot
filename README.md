# PDF-Based RAG Chatbot

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that extracts information from PDF documents and answers user queries using **FAISS-based vector retrieval** and a **transformer-based QA model**.

## Features
- **PDF Upload & Text Extraction:** Extracts text from uploaded PDFs.
- **Chunking for Efficient Retrieval:** Splits text into meaningful chunks for better searchability.
- **Vector Database (FAISS) for Fast Search:** Converts text chunks into embeddings using **SentenceTransformers** and stores them in **FAISS**.
- **Transformer-Based Question Answering:** Uses **RoBERTa-Large-SQuAD2** for context-aware answering.
- **Interactive Chatbot Interface:** Users can input queries and receive answers dynamically.

## Tech Stack
- **Python**
- **PyPDF2** (PDF text extraction)
- **FAISS** (Vector-based retrieval)
- **SentenceTransformers** (Embeddings)
- **Transformers (Hugging Face)** (QA model)
- **Streamlit** (Frontend UI for chatbot)

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed, then install dependencies:

```bash
pip install streamlit PyPDF2 faiss-cpu sentence-transformers transformers numpy
```

### Running the Application
```bash
streamlit run app.py
```

## How to Use
1. **Upload a PDF Document:** Click the "Upload PDF" button and select a file.
2. **Ask Questions:** Type a query related to the uploaded document.
3. **Receive Answers:** The chatbot retrieves relevant chunks and provides answers.
4. **Reset:** Clear the uploaded document and input for a new session.

## File Structure
```
├── app.py                      # Streamlit chatbot UI
├── pdf_processor.py             # PDF text extraction and chunking
├── vector_store.py              # FAISS index creation and retrieval
├── qa_model.py                  # Transformer-based QA model
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
```

## Example Interaction
```
User: "What is the main topic of the document?"
Chatbot: "The document discusses advancements in AI for healthcare."
```

## Authors
- **Sarvesh PV**
