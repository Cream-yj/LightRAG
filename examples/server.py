import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

app = Flask(__name__, static_folder="static", static_url_path="/static")

WORKING_DIR = "./dickens_2000"
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

# 插入知识库
with open("/root/LightRAG/examples/kg_2000.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/query", methods=["POST"])
def query_model():
    """
    前端会发送:
    {
      "conversation": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好! 我是XX..."},
        {"role": "user", "content": "再给我讲讲..."}
      ]
    }
    """
    data = request.json
    conversation = data.get("conversation", [])

    if not conversation:
        return jsonify({"error": "No conversation provided."}), 400

    # 仅获取用户最后一条的问题
    user_input = conversation[-1]["content"]

    # 直接调用 rag.query，而不再拼接任何上下文
    resp = rag.query(user_input, param=QueryParam(mode="hybrid"))

    # 返回回答
    return jsonify({
        "answer": resp
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
