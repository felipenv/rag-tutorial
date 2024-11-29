import faiss
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RAG:
    def __init__(self):
        pass

    def query_faiss_index_with_metadata(self, query_text, folder, index_name, top_k=3):
        """
        Queries a FAISS index for the closest matches and retrieves metadata.

        Parameters:
        - query_text (str): The text query to search for.
        - index_name (str): Base name for the FAISS index and metadata files.
        - top_k (int): Number of top matches to retrieve.

        Returns:
        - results (list): A list of dictionaries containing match index, distance, and chunk content.
        """
        # Load FAISS index and metadata
        faiss_index_file = f"{folder}/{index_name}.index"
        metadata_file = f"{folder}/{index_name}_metadata.npy"
        faiss_index = faiss.read_index(faiss_index_file)
        metadata = np.load(metadata_file, allow_pickle=True).item()

        logging.info(f"Querying FAISS index: '{faiss_index_file}'")

        # Initialize OpenAI embedding model
        embedding_model = OpenAIEmbedding(model="text-embedding-3-large")

        # Generate embedding for the query
        query_vector = embedding_model.get_text_embedding(query_text)

        # Search the FAISS index for the top_k closest vectors
        distances, indices = faiss_index.search(np.array([query_vector]), k=top_k)

        # Retrieve the results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1:  # FAISS returns -1 for empty results
                results.append({
                    "index": idx,
                    "distance": distance,
                    "content": metadata[idx]
                })

        logging.info(f"Query completed. Retrieved {len(results)} results.")
        return results