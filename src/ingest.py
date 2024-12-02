import pdfplumber
import tiktoken

import faiss
from bs4 import BeautifulSoup
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


tokenizer = tiktoken.get_encoding("cl100k_base")


class Ingest:
    def __init__(self):
        pass

    def pdf_to_chunks(self, pdf_path, max_tokens):
        """
         Converts a PDF file to text chunks for vector database preparation.

         Parameters:
         - pdf_path (str): Path to the input PDF file.
         - chunk_size (int): Maximum size of each text chunk in characters.

         Returns:
         - chunks (list): List of text chunks.
         """
        full_text = ""

        # Extract all text from the PDF
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"  # Add a newline between pages for readability

        return self._chunking(full_text, tokenizer, max_tokens=max_tokens)

    def txt_to_chunks(self, txt_path, max_tokens):
        """
         Converts a PDF file to text chunks for vector database preparation.

         Parameters:
         - pdf_path (str): Path to the input PDF file.
         - chunk_size (int): Maximum size of each text chunk in characters.

         Returns:
         - chunks (list): List of text chunks.
         """
        # Read the full text from the TXT file
        with open(txt_path, "r", encoding="utf-8") as txt:
            full_text = txt.read()

        return self._chunking(full_text, tokenizer, max_tokens=max_tokens)

    def html_to_chunks(self, html_path, max_tokens):
        """
         Converts a HTML file to text chunks for vector database preparation.

         Parameters:
         - html_path (str): Path to the input PDF file.
         - chunk_size (int): Maximum size of each text chunk in characters.

         Returns:
         - chunks (list): List of text chunks.
         """

        with open(html_path, "r", encoding="utf-8") as h:
            html = h.read()

        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.find_all(['h3', 'b', 'blockquote'])
        full_text = "\n".join([text.get_text() for text in texts if text.b is None])

        return self._chunking(full_text, tokenizer, max_tokens=max_tokens)

    def _chunking(self, full_text, tokenizer, max_tokens=100):
        sentences = full_text.split(
            '\n')  # Split by newline to preserve sentence boundaries

        # Create chunks without breaking sentences
        current_chunk = ""
        current_tokens = 0

        chunks = []

        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence)
            sentence_length = len(sentence_tokens)

            # If adding this sentence exceeds the token limit, finalize the current chunk
            if current_tokens + sentence_length > max_tokens:
                chunks.append(
                    current_chunk.strip() + '\n')  # Ensure the chunk ends with a newline
                current_chunk = ""
                current_tokens = 0

            # Add the sentence to the current chunk
            current_chunk += sentence + '\n'
            current_tokens += sentence_length

        # Add the last chunk if any text remains
        if current_chunk.strip():
            chunks.append(current_chunk.strip() + '\n')

        return chunks

    def ingest_chunks_to_faiss_with_metadata(self, chunks, index_name,
                                             folder, mode="append"):
        """
        Ingests a list of text chunks into a FAISS vector database and saves metadata separately.

        Parameters:
        - chunks (list): List of text chunks to embed and index.
        - index_name (str): Base name for the FAISS index and metadata files.
        - mode (str): "append" to add chunks to an existing index, "overwrite" to replace the index.

        Saves:
        - FAISS index to a file named <index_name>.index.
        - Metadata to a file named <index_name>_metadata.npy.
        """
        # Initialize OpenAI embedding model
        embedding_model = OpenAIEmbedding(model="text-embedding-3-large")
        os.makedirs(folder, exist_ok=True)
        # Embed chunks into vectors
        embeddings = [embedding_model.get_text_embedding(chunk) for chunk in chunks]
        embedding_array = np.array(embeddings).astype("float32")

        faiss_index_file = f"{index_name}.index"
        metadata_file = f"{index_name}_metadata.npy"

        if os.path.exists(faiss_index_file):
            if mode == "append":
                logging.info(f"Appending to existing FAISS index: '{faiss_index_file}'")
                # Load existing FAISS index
                faiss_index = faiss.read_index(faiss_index_file)

                # Load existing metadata
                existing_metadata = np.load(metadata_file, allow_pickle=True).item()

                # Check for dimensionality mismatch
                if faiss_index.d != embedding_array.shape[1]:
                    raise ValueError(
                        f"Dimensionality mismatch: index has {faiss_index.d} dimensions, "
                        f"but embeddings have {embedding_array.shape[1]} dimensions."
                    )

                # Append new metadata
                new_metadata = {len(existing_metadata) + i: chunk for i, chunk in
                                enumerate(chunks)}
                metadata = {**existing_metadata, **new_metadata}

            elif mode == "overwrite":
                logging.info(f"Overwriting existing FAISS index: '{faiss_index_file}'")
                dimension = embedding_array.shape[1]
                faiss_index = faiss.IndexFlatIP(dimension)
                metadata = {i: chunk for i, chunk in enumerate(chunks)}
            else:
                raise ValueError("Invalid mode. Choose 'append' or 'overwrite'.")
        else:
            logging.info(f"Creating new FAISS index: '{faiss_index_file}'")
            dimension = embedding_array.shape[1]
            faiss_index = faiss.IndexFlatIP(dimension)
            metadata = {i: chunk for i, chunk in enumerate(chunks)}

        # Add embeddings to the FAISS index
        faiss_index.add(embedding_array)

        # Save FAISS index and metadata
        faiss.write_index(faiss_index, f"{folder}/{faiss_index_file}")
        np.save(f"{folder}/{metadata_file}", metadata)

        logging.info(
            f"FAISS index saved as '{folder}/{faiss_index_file}'. Metadata saved as '{folder}/{faiss_index_file}'.")