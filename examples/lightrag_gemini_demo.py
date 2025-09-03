# pip install -q -U google-genai to use gemini as a client

import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)

client = genai.Client(vertexai=True)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    # Combine prompts: system prompt, history, and user prompt
    if history_messages is None:
        history_messages = []

    combined_prompt = ""
    # if system_prompt:
    #     combined_prompt += f"{system_prompt}\n"

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

    # model = SentenceTransformer("all-MiniLM-L6-v2")
    # embeddings = model.encode(texts, convert_to_numpy=True)
    # return embeddings

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


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    file_path = "./book.txt"
    with open(file_path, "r") as file:
        text = file.read()

    rag.insert(text)

    print("=" * 80)
    print("RAG-based Response:")
    print("=" * 80)
    response = rag.query(
        query="What is the main theme of the story?",
        param=QueryParam(mode="hybrid", top_k=5, response_type="single line"),
    )
    print(response)

    print("\n" + "=" * 80)
    print("Direct Gemini Flash Response (without RAG):")
    print("=" * 80)
    
    # Direct query to Gemini Flash with the full text
    direct_prompt = f"""Here is a story:

{text}

Question: What is the main theme of the story?"""

    direct_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[direct_prompt],
        config=types.GenerateContentConfig(
            system_instruction="Provide a concise, single-line answer about the main theme.",
            max_output_tokens=None,
            temperature=0.1
        ),
    )
    
    if direct_response and hasattr(direct_response, 'text') and direct_response.text:
        print(direct_response.text)
    elif direct_response and direct_response.candidates and len(direct_response.candidates) > 0:
        candidate = direct_response.candidates[0]
        if candidate.content and candidate.content.parts:
            text_parts = []
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)
            if text_parts:
                print(''.join(text_parts))
    else:
        print("No response received from Gemini Flash")


if __name__ == "__main__":
    main()
