

import os
import json
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Set up logging
setup_logger("lightrag", level="INFO")

# --- Configuration ---
WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# --- RAG Initialization ---
async def initialize_rag():
    """Initializes the LightRAG instance with PostgreSQL storage."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="PGGraphStorage",
        doc_status_storage="PGDocStatusStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# --- Main Execution ---
async def main():
    """Main function to load data, index it, and run a query."""
    rag = None
    try:
        # 1. Initialize LightRAG
        rag = await initialize_rag()

        # 2. Load and process the input JSON file
        with open('cars.json', 'r') as f:
            cars_data = json.load(f)
        
        # Convert each car object to a descriptive string for insertion
        documents_to_insert = [
            f"{car['make']} {car['model']} ({car['year']}): {car['description']}"
            for car in cars_data
        ]

        # 3. Insert the documents into LightRAG
        print("\n--- Inserting documents into LightRAG ---")
        await rag.ainsert(documents_to_insert)
        print("\n--- Documents inserted successfully ---")

        # 4. Query the indexed data
        print("\n--- Querying the data ---")
        query = "Which car is known for being a muscle car?"
        response = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        
        print(f"\nQuery: {query}")
        print(f"Response: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    # Ensure you have set your OPENAI_API_KEY environment variable
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
    else:
        asyncio.run(main())

