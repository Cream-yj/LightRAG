import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# 1. 设置工作目录
WORKING_DIR = "./dickens_100"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 2. 初始化 LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="deepseek-r1:7b",
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

# 3. 插入知识库
with open("/root/LightRAG/examples/kg_100.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# 4. 定义一个异步函数，用来逐块打印流式输出
async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()  # 输出完毕后换行

# 5. 交互循环
def main():
    print("=== Interactive mode ===")
    print("输入你的问题, 或者输入 'quit' / 'exit' 退出程序")
    
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Bye!")
            break

        # 6. 对用户问题执行查询
        print("[DEBUG] Before querying the model...")
        resp = rag.query(user_input, param=QueryParam(mode="hybrid", stream=True))
        print("[DEBUG] After querying the model, got:", resp)


        # 7. 如果是流式输出，就异步打印；否则直接打印
        if inspect.isasyncgen(resp):
            asyncio.run(print_stream(resp))
        else:
            print(resp)

# 8. 运行主函数
if __name__ == "__main__":
    main()
