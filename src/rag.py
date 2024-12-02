import faiss
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RAG:
    def __init__(self):
        pass

    def query_index(self, query_text, index_folder, index_name, top_k=3):
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
        faiss_index_file = f"{index_folder}/{index_name}.index"
        metadata_file = f"{index_folder}/{index_name}_metadata.npy"
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

    def generate_rag_response(self, query, settings, character, top_k=3):
        """
        Generates a response using RAG by retrieving context from FAISS and using
        GPT-4o to answer.

        Parameters:
        - query (str): User's query.
        - settings (dict): index_folder: Base folder for the FAISS index and
        metadata files
            prompt_folder: Base folder for the prompts
        - character (str): which character to use for response
        - top_k (int): Number of top chunks to retrieve as context.

        Returns:
        - response (str): Generated response from GPT-4.
        """
        # Retrieve context from FAISS
        results = self.query_index(query, settings['index_folder'], character,
                                   top_k=top_k)
        context = "\n".join([result["content"] for result in results])

        # Build the GPT-4 prompt
        with open(f"{settings['prompt_folder']}/{character}", "r") as file:
            prompt = file.read()

        prompt = prompt.format(CONTEXT=context, QUERY=query)

        # Query GPT-4o for an answer
        logging.info("Sending query to GPT-4o...")

        openai_api_key = os.environ['OPENAI_API_KEY']
        openai_base_url = os.environ['OPENAI_API_BASE']
        client = openai.OpenAI(api_key=openai_api_key, base_url=openai_base_url)

        chat_completion = client.chat.completions.create(messages=[{
            "role": "user",
            "content": prompt,
        }],
            model="gpt-4o"
        )

        # Extract and return the answer
        answer = chat_completion.choices[0].message.content
        return answer