import asyncio
import os
import inspect
import logging
import openai
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# Set OpenAI API key
openai.api_key = "sk-ZDruL5Z23mx1MMUtG9ygfvXX6zli3fxLIlgSsX61iGZRFWWI"

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Define the embedding function using OpenAI's embeddings
def openai_embed(texts, embed_model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        model=embed_model,
        input=texts
    )
    return [embedding['embedding'] for embedding in response['data']]

# Define the completion function using OpenAI's Chat API
def openai_model_complete(prompt, model="gpt-3.5-turbo", max_tokens=1500, **kwargs):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        **kwargs
    )
    return response.choices[0].message['content']

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_model_complete,  # Use OpenAI's model completion function
    llm_model_name="gpt-3.5-turbo",  # Or another model like "gpt-4"
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"temperature": 0.7},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=openai_embed,  # Use OpenAI's embedding function
    ),
)

# Insert data into the RAG
with open("./kg.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="hybrid"))
)

# Stream response
resp = rag.query(
    "What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题",
    param=QueryParam(mode="hybrid", stream=True),
)

async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
