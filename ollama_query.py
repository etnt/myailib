import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from pdf_vdb import PDFVectorDatabase
from runnable_parsers import DocumentMessageToString
import argparse
import textwrap
from rich.console import Console
from rich.markdown import Markdown


class OllamaQuerySystem:
    def __init__(self, pdf_directory, model_name="qwen2.5-coder"):
        self.pdf_db = PDFVectorDatabase(pdf_directory)
        self.llm = Ollama(model=model_name)
        self.output_parser = DocumentMessageToString()

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Use the following context to answer the question at the end.
            If you don't know the answer, just say you don't know. Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:
            """
        )

        query_chain = RunnableLambda(self.pdf_db.query_database) | self.output_parser
        # self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        self.llm_chain = {"context": query_chain, "question": RunnablePassthrough()} | self.prompt_template | self.llm

        #self.qa_chain = (
        #    {"context": self.pdf_db.query_database, "question": RunnablePassthrough()}
        #    | self.llm_chain
        #)

    def query(self, question):
        #return self.qa_chain.invoke(question)
        return self.llm_chain.invoke(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF files and query a vector database.")
    parser.add_argument('--pdf-dir', type=str, required=True, help='Directory containing PDF files')
    args = parser.parse_args()
    pdf_directory = args.pdf_dir

    qa_system = OllamaQuerySystem(pdf_directory)
    console = Console()

    print("\n")
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer = qa_system.query(user_query)

        # Join the list of answer strings into a single string
        answer_text = ''.join(answer)

        # Use textwrap to format the text into a paragraph
        #formatted_text = textwrap.fill(answer_text, width=80)

        # Print the formatted text using rich for Markdown rendering
        print("Answer::\n")
        console.print(Markdown(answer_text))
        print("\n")
        

print("Thank you, and goodbye for now!")
