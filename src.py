# Import required libraries
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Step 1: Validate the PDF file
def validate_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) > 0:  # Check if the PDF has pages
                return True
            else:
                return False
    except Exception as e:
        logging.error(f"Error validating PDF: {e}")
        return False

# Step 2: Load and extract text from the PDF
def load_pdf(file_path):
    try:
        if not validate_pdf(file_path):
            return None, "The PDF file is corrupted or invalid."

        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure text extraction is successful
                    text += page_text
            if not text.strip():  # Check if text is empty
                return None, "The PDF file contains no extractable text."
            return text, None
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        return None, f"Error loading PDF: {e}"

# Step 3: Split text into chunks
def split_text(text, chunk_size=500, chunk_overlap=100):  # Increased overlap for better context
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        return None

# Step 4: Generate embeddings and create a vector database
def create_vector_db(chunks):
    try:
        # Load a more advanced embedding model
        model = SentenceTransformer('all-mpnet-base-v2')  # Higher quality embeddings
        embeddings = model.encode(chunks)

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        return index, model, chunks
    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        return None, None, None

# Step 5: Retrieve relevant chunks based on the query
def retrieve_relevant_chunks(query, index, model, chunks, top_k=5):  # Increased top_k for more context
    try:
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)
        relevant_chunks = [chunks[i] for i in indices[0]]
        return relevant_chunks
    except Exception as e:
        logging.error(f"Error retrieving relevant chunks: {e}")
        return None

# Step 6: Generate an answer using a language model
def generate_answer(query, relevant_chunks):
    try:
        # Combine chunks into context
        context = " ".join(relevant_chunks)

        # Load a more powerful QA model
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")  # Larger model

        # Generate answer
        result = qa_pipeline(question=query, context=context)
        return result['answer']
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer."

# Step 7: Main RAG pipeline
def rag_pipeline(file_path, query):
    # Step 1: Load PDF
    text, error = load_pdf(file_path)
    if error:
        return f"Error: {error}"

    # Step 2: Split text into chunks
    chunks = split_text(text)
    if not chunks:
        return "Error: Could not split text."

    # Step 3: Create vector database
    index, model, chunks = create_vector_db(chunks)
    if not index:
        return "Error: Could not create vector database."

    # Step 4: Retrieve relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, index, model, chunks)
    if not relevant_chunks:
        return "Error: Could not retrieve relevant chunks."

    # Step 5: Generate answer
    answer = generate_answer(query, relevant_chunks)
    return answer

# Chatbot function
def chatbot(file_path):
    print("Welcome to the PDF-based Chatbot!")
    print("You can ask questions about the document. Type 'exit' or 'quit' to end the chat.")

    while True:
        # Get user input
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        # Get answer from the RAG pipeline
        answer = rag_pipeline(file_path, query)
        print(f"Chatbot: {answer}")

# Example usage
if __name__ == "__main__":
    # Replace with the path to your PDF file
    file_path = r"C:\Users\Sarvesh PV\Downloads\Project Set\chatbot\resume.pdf"

    # Start the chatbot
    chatbot(file_path)