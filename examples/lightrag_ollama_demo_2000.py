# 用的kg前2000行，测试连续对话
# 修改 WORKING_DIR = "./dickens_2000"
# with open("../kg_2000.txt", "r", encoding="utf-8") as f:

import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./dickens_2000"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="deepseek-r1:32b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

with open("./kg_2000.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
# print(
#     rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
# )


# print(
#     rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="naive"))
# )

# # ITGB3 PARTICIPATES_GpPW IL4-mediated_signaling_events

# # Perform local search
# print(
#     rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="local"))
# )

# # Perform global search
# print(
#     rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="global"))
# )

# # # Perform hybrid search
# print(
#     rag.query("What is the relationship between ABAT and neuron_projection? 以三元组A-B-C的形式，其中A、C是实体，B是A和C的关系；用英文回答我的问题", param=QueryParam(mode="hybrid"))
# )

# stream response
# resp = rag.query(
#     "What are the top themes in this story?",
#     param=QueryParam(mode="hybrid", stream=True),
# )

resp = rag.query(
    "What is the relationship between AASS and metabolic_process? 以三元组A-B-C的形式,，其中A和C都是实体，B描述它们之间的关系用英文回答我的问题",
    param=QueryParam(mode="hybrid", stream=True),
)



async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
