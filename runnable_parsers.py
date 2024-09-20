from langchain.schema.runnable import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# Experiment with a Runnable that can convert a list of Message objects to a list of strings

class HumanMessageToString(Runnable):
    def invoke(self, input_messages, config):
        #print(f"config: {config}")
        # Check if input_messages is a list
        if isinstance(input_messages, list):
            # Initialize an empty list to store content strings
            content_strings = []
            # Loop over each message in the list
            for message in input_messages:
                # Check if the message is a HumanMessage instance
                if isinstance(message, HumanMessage):
                    # Append the content string to the list
                    content_strings.append(message.content)
                else:
                    raise TypeError("Expected each item in the list to be a HumanMessage object.")
            # Return the list of content strings
            return content_strings
        else:
            raise TypeError("Expected input_messages to be a list of HumanMessage objects.")
        
class DocumentMessageToString(Runnable):
    def invoke(self, input_documents, config):
        # Check if input_documents is a list
        if isinstance(input_documents, list):
            # Initialize an empty list to store content strings
            content_strings = []
            # Loop over each document in the list
            for document in input_documents:
                # Check if the document is a Document instance
                if isinstance(document, Document):
                    # Append the content string to the list
                    content_strings.append(document.page_content)
                else:
                    raise TypeError("Expected each item in the list to be a Document object.")
            # Return the list of content strings
            return content_strings
        else:
            raise TypeError("Expected input_documents to be a list of Document objects.")
