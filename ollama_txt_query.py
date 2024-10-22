import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
import argparse

class OllamaQuerySystem:
    def __init__(self, text, model_name="qwen2.5-coder"):
        self.llm = Ollama(model=model_name)

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. You may use the following context to extend your knowledge
            in order to aid you to answer the question at the end. Use a Chain of Thought process to
            generate a response.
            If you don't know the answer, just say you don't know. Don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:
            """
        )

        self.llm_chain = (
            {
                "context": lambda _: text,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
        )

    def query(self, question):
        try:
            return self.llm_chain.invoke(question)
        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query the Ollama AI system with a text file as context.")
    parser.add_argument("file", type=str, help="Path to the text file containing the context.")
    args = parser.parse_args()

    # Read text from the file specified on the command line
    try:
        with open(args.file, "r") as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: The file '{args.file}' was not found.")
        exit(1)
    except IOError:
        print(f"Error: Could not read the file '{args.file}'.")
        exit(1)

    qa_system = OllamaQuerySystem(text)
    console = Console()

    print("\n")
    while True:
        user_query = input("Enter your question (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer = qa_system.query(user_query)

        # Pretty print the input query
        print("\n")
        console.print(Panel.fit(
            Text("Query: ", style="bold green") + Text(user_query, style="green"), 
            title="Input Query",
            border_style="green"
        ))

        # Print the formatted text using rich for Markdown rendering
        print("Answer:\n")
        console.print(Markdown(answer))
        print("\n")

    print("Thank you, and goodbye for now!")
