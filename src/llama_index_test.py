from llama_index.core import SimpleDirectoryReader, GPTVectorStoreIndex
import os

openai_api_key = os.environ['OPENAI_API_KEY']
openai_api_base = os.environ['OPENAI_API_BASE']


def test_llamaindex(query):
    try:
        # Create a sample document for testing
        with open("sample1.txt", "w") as f:
            f.write("Deadpool is a witty, sarcastic anti-hero.")
        with open("sample2.txt", "w") as f:
            f.write("William Shakespeare: Shakespeare is a poetic and eloquent "
                    "wordsmith, whose deep understanding of human nature shines "
                    "through his dialogue.")

        # Load data and build the index
        documents = SimpleDirectoryReader(input_dir=".").load_data()
        index = GPTVectorStoreIndex.from_documents(documents)

        # Create a query engine
        query_engine = index.as_query_engine()

        response = query_engine.query(query)
        print("Response:", response.response)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    test_llamaindex()