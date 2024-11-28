import pdfplumber
import tiktoken


class Parser():
    def __init__(self):
        pass

    def parse_pdf(self, pdf_path, chunk_size):
        """
         Converts a PDF file to text chunks for vector database preparation.

         Parameters:
         - pdf_path (str): Path to the input PDF file.
         - chunk_size (int): Maximum size of each text chunk in characters.

         Returns:
         - chunks (list): List of text chunks.
         """
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Split the text into chunks of the specified size
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i + chunk_size]
                        chunks.append(chunk)
        return chunks