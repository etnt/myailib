import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from runnable_parsers import DocumentMessageToString
from sentence_transformers import CrossEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class PDFVectorDatabase:
    def __init__(self, pdf_directory, persist_directory=None):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.all_chunks = []
        self.vector_db = None

        # Initialize the text splitter and embeddings model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)                                          
        self.embeddings_model = HuggingFaceEmbeddings()

        # Initialize query expansion model
        self.query_expansion_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.query_expansion_tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

        # Initialize cross-encoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Initialize BM25 for reranking
        self.bm25 = None

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
    
    def expand_query(self, query):
        input_text = f"expand query: {query}"
        input_ids = self.query_expansion_tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.query_expansion_model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=3,
            num_beams=3,
            early_stopping=True
        )
        expanded_queries = [self.query_expansion_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return [query] + expanded_queries

    def rerank_with_bm25(self, query, documents, top_k=10):
        if self.bm25 is None:
            tokenized_corpus = [doc.page_content.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [documents[i] for i in top_indices]

    def query_database(self, query):
        if not self.vector_db:
            raise ValueError("Database is empty. Please ensure PDFs are loaded and chunks created.")
        
        # Query expansion
        expanded_queries = self.expand_query(query)
        
        # First-stage retrieval
        all_results = []
        for expanded_query in expanded_queries:
            results = self.vector_db.similarity_search(expanded_query, k=20)
            all_results.extend(results)
        
        # Remove duplicates
        unique_results = list({doc.page_content: doc for doc in all_results}.values())
        
        # Rerank with Cross-Encoder
        pairs = [(query, doc.page_content) for doc in unique_results]
        cross_scores = self.cross_encoder.predict(pairs)
        reranked_results = [doc for _, doc in sorted(zip(cross_scores, unique_results), reverse=True)]
        
        # Final reranking with BM25 to address "lost in the middle"
        final_results = self.rerank_with_bm25(query, reranked_results)
        
        return final_results

# Example usage
if __name__ == "__main__":
    pdf_directory = './PDFs'
    persist_directory = './vector_db'
    db = PDFVectorDatabase(pdf_directory, persist_directory)

    out_parser = DocumentMessageToString()

    chain = RunnableLambda(db.query_database) | out_parser

    query = "How can I implement an NSO action in my Python code?"

    results = chain.invoke(query)
    
    console = Console()
    
    console.print(Panel.fit(
        Text("Query: ", style="bold green") + Text(query, style="italic"),
        title="Input Query",
        border_style="green"
    ))
    
    console.print("\n[bold blue]Top Results:[/bold blue]")
    
    for i, result in enumerate(results[:3], 1):
        console.print(Panel(
            Text(result[:300] + "..." if len(result) > 300 else result, style="dim"),
            title=f"Result {i}",
            border_style="blue",
            expand=False
        ))
        console.print()  # Add a blank line between results
    
    
    
