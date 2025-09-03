import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()

WORKING_DIR = "./custom_kg"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

client = genai.Client(vertexai=True)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""

    for msg in history_messages:
        # Each msg is expected to be a dict: {"role": "...", "content": "..."}
        combined_prompt += f"{msg['role']}: {msg['content']}\n"

    # Finally, add the new user prompt
    combined_prompt += f"user: {prompt}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[combined_prompt],
        config=types.GenerateContentConfig(system_instruction=system_prompt, max_output_tokens=None, temperature=0.1),
    )

    # Try to get text from response.text first
    if response and hasattr(response, 'text') and response.text:
        return response.text
    
    # If response.text is None, try to extract from candidates
    if response and response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            if text_parts:
                return ''.join(text_parts)
    
    # If no valid response
    return ""


async def embedding_func(texts: list[str]) -> np.ndarray:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",
                                        output_dimensionality=3072))

    # Extract the actual embedding values from ContentEmbedding objects
    return np.array([np.array(emb.values) for emb in result.embeddings])


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=2048,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

custom_kg = {
    "entities": [
        {
            "entity_name": "CompanyA",
            "entity_type": "Organization",
            "description": "A major technology company",
            "source_id": "Source1",
        },
        {
            "entity_name": "ProductX",
            "entity_type": "Product",
            "description": "A popular product developed by CompanyA",
            "source_id": "Source1",
        },
        {
            "entity_name": "PersonA",
            "entity_type": "Person",
            "description": "A renowned researcher in AI",
            "source_id": "Source2",
        },
        {
            "entity_name": "UniversityB",
            "entity_type": "Organization",
            "description": "A leading university specializing in technology and sciences",
            "source_id": "Source2",
        },
        {
            "entity_name": "CityC",
            "entity_type": "Location",
            "description": "A large metropolitan city known for its culture and economy",
            "source_id": "Source3",
        },
        {
            "entity_name": "EventY",
            "entity_type": "Event",
            "description": "An annual technology conference held in CityC",
            "source_id": "Source3",
        },
    ],
    "relationships": [
        {
            "src_id": "CompanyA",
            "tgt_id": "ProductX",
            "description": "CompanyA develops ProductX",
            "keywords": "develop, produce",
            "weight": 1.0,
            "source_id": "Source1",
        },
        {
            "src_id": "PersonA",
            "tgt_id": "UniversityB",
            "description": "PersonA works at UniversityB",
            "keywords": "employment, affiliation",
            "weight": 0.9,
            "source_id": "Source2",
        },
        {
            "src_id": "CityC",
            "tgt_id": "EventY",
            "description": "EventY is hosted in CityC",
            "keywords": "host, location",
            "weight": 0.8,
            "source_id": "Source3",
        },
    ],
    "chunks": [
        {
            "content": "ProductX, developed by CompanyA, has revolutionized the market with its cutting-edge features.",
            "source_id": "Source1",
            "source_chunk_index": 0,
        },
        {
            "content": "One outstanding feature of ProductX is its advanced AI capabilities.",
            "source_id": "Source1",
            "chunk_order_index": 1,
        },
        {
            "content": "PersonA is a prominent researcher at UniversityB, focusing on artificial intelligence and machine learning.",
            "source_id": "Source2",
            "source_chunk_index": 0,
        },
        {
            "content": "EventY, held in CityC, attracts technology enthusiasts and companies from around the globe.",
            "source_id": "Source3",
            "source_chunk_index": 0,
        },
        {
            "content": "None",
            "source_id": "UNKNOWN",
            "source_chunk_index": 0,
        },
    ],
}


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    
    # Insert custom knowledge graph
    rag.insert_custom_kg(custom_kg)
    
    print("Custom knowledge graph inserted successfully!")


if __name__ == "__main__":
    main()
