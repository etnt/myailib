import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from runnable_parsers import DocumentMessageToString 

class PDFVectorDatabase:
    def __init__(self, pdf_directory, persist_directory=None):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.all_chunks = []
        self.vector_db = None

        # Initialize the text splitter and embeddings model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)                                          
        self.embeddings_model = HuggingFaceEmbeddings()

        # Load PDFs and create chunks
        self._load_and_chunk_pdfs()

    def _load_and_chunk_pdfs(self):
        # Check if a persistent database already exists
        if self.persist_directory:
            if os.path.exists(self.persist_directory):
                chroma_file_path = os.path.join(self.persist_directory, 'chroma.sqlite3')
                if os.path.isfile(chroma_file_path):
                    print(f"Loading existing Chroma database at {chroma_file_path}")
                    # Load the existing Chroma database
                    self.vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings_model)
                    return

        # Check if the PDF directory exists  
        if not os.path.exists(self.pdf_directory):
            raise FileNotFoundError(f"The directory {self.pdf_directory} does not exist.")
         
        # Loop over each PDF file in the directory and split the text into chunks
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                print(f"Processing: {filename}")
                file_path = os.path.join(self.pdf_directory, filename)
                
                with open(file_path, 'rb') as file:
                    reader = PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text = page.extract_text()
                        
                        chunks = self.text_splitter.split_text(text)
                        self.all_chunks.extend(chunks)
        
        # Create the Chroma vector database
        if self.persist_directory:
            self.vector_db = Chroma.from_texts(
                self.all_chunks, 
                self.embeddings_model, 
                persist_directory=self.persist_directory
            )
            print(f"Vector database persisted to {self.persist_directory}")
        else:
            self.vector_db = Chroma.from_texts(self.all_chunks, self.embeddings_model)
    
    def query_database(self, query):
        if not self.vector_db:
            raise ValueError("Database is empty. Please ensure PDFs are loaded and chunks created.")
        
        return self.vector_db.similarity_search(query)

# Example usage
if __name__ == "__main__":
    pdf_directory = './PDFs'
    persist_directory = './vector_db'
    db = PDFVectorDatabase(pdf_directory, persist_directory)

    out_parser = DocumentMessageToString()

    chain = RunnableLambda(db.query_database) | out_parser

    query = "What is foobar?"

    print(chain.invoke(query))
    
    
    
