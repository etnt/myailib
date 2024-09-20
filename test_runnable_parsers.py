import unittest
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from runnable_parsers import HumanMessageToString, DocumentMessageToString

class TestHumanMessageToString(unittest.TestCase):
    def setUp(self):
        self.parser = HumanMessageToString()

    def test_single_message(self):
        input_messages = [HumanMessage(content="Hello, world!")]
        result = self.parser.invoke(input_messages, {})
        self.assertEqual(result, ["Hello, world!"])

    def test_multiple_messages(self):
        input_messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
            HumanMessage(content="Third message")
        ]
        result = self.parser.invoke(input_messages, {})
        self.assertEqual(result, ["First message", "Second message", "Third message"])

    def test_empty_list(self):
        input_messages = []
        result = self.parser.invoke(input_messages, {})
        self.assertEqual(result, [])

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.parser.invoke("Not a list", {})

    def test_invalid_message_type(self):
        with self.assertRaises(TypeError):
            self.parser.invoke([HumanMessage(content="Valid"), "Invalid"], {})

class TestDocumentMessageToString(unittest.TestCase):
    def setUp(self):
        self.parser = DocumentMessageToString()

    def test_single_document(self):
        input_documents = [Document(page_content="Document content")]
        result = self.parser.invoke(input_documents, {})
        self.assertEqual(result, ["Document content"])

    def test_multiple_documents(self):
        input_documents = [
            Document(page_content="First document"),
            Document(page_content="Second document"),
            Document(page_content="Third document")
        ]
        result = self.parser.invoke(input_documents, {})
        self.assertEqual(result, ["First document", "Second document", "Third document"])

    def test_empty_list(self):
        input_documents = []
        result = self.parser.invoke(input_documents, {})
        self.assertEqual(result, [])

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.parser.invoke("Not a list", {})

    def test_invalid_document_type(self):
        with self.assertRaises(TypeError):
            self.parser.invoke([Document(page_content="Valid"), "Invalid"], {})

if __name__ == '__main__':
    unittest.main()
