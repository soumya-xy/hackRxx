import requests
from pypdf import PdfReader
import docx2txt
import io
from urllib.parse import urlparse

class DocumentParser:
    @staticmethod
    def parse_pdf(url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        pdf_reader = PdfReader(io.BytesIO(response.content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def parse_docx(file_path: str) -> str:
        # This part of the logic needs to be updated to handle URLs as well.
        # For now, it's fine as a placeholder.
        return docx2txt.process(file_path)

    def parse_document(self, doc_url: str) -> str:
        # Use urlparse to get the path of the URL, ignoring the query string.
        path = urlparse(doc_url).path
        
        if path.lower().endswith(".pdf"):
            return self.parse_pdf(doc_url)
        # Add logic for DOCX, email, etc.
        # For simplicity, we'll only handle PDF for this example.
        raise ValueError("Unsupported document type")