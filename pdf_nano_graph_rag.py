import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nano_graphrag import GraphRAG, QueryParam

# See: https://github.com/gusye1234/nano-graphrag

class PDFNanoGraphRag:
    def __init__(self, pdf_directory, persist_directory):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.all_chunks = []
        self.vector_db = None

        self.graph_func = GraphRAG(working_dir=persist_directory)

        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 

        # Load PDFs and create chunks
        #self._load_and_chunk_pdfs()

    def _load_and_chunk_pdfs(self):
        chunks = []
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
                        chunks.extend(chunks)

        self.graph_func.insert(chunks)

    def query_database(self, query):
        return self.graph_func.query(query)


if __name__ == "__main__":

    ngrag = PDFNanoGraphRag(pdf_directory="./PDFs", persist_directory="./output")

    print(ngrag.query_database(input("Ask a question: ").strip()))
