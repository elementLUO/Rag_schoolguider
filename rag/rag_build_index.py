"""
用于：读取 docs/ 下的介绍文本 -> 切成 chunks -> 用通义模型生成 embedding -> 保存到本地文件。
"""

import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# === 通义客户端 ===
# 注意：需要在环境变量里设置 DASHSCOPE_API_KEY，或者自己改成明文（不推荐）
#os.environ["DASHSCOPE_API_KEY"] = ""
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DOCS_DIR = "./docs"
EMBEDDING_MODEL = "text-embedding-v4"   # 通义 embedding 模型
SAVE_PATH = "../rag/corpus_embeddings.npz"   # 保存 chunks + embeddings


def get_embeddings(texts, model=EMBEDDING_MODEL):
    """
    调用通义千问的 embedding 接口，将一批文本转为向量。
    texts: List[str]
    return: List[List[float]]
    """
    completion = client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [data.embedding for data in completion.data]
    return embeddings


def load_docs(doc_dir=DOCS_DIR):
    """
    简单版：把 docs/ 里的每个文件看成一个 chunk。
    也可以让按段落切分。
    """
    chunks = []
    for fname in os.listdir(doc_dir):
        if fname.endswith(".txt") or fname.endswith(".md"):
            path = os.path.join(doc_dir, fname)
            with open(path, "r", encoding="utf8") as f:
                text = f.read().strip()
            if text:
                # 在 chunk 里保留文件名，方便调试
                chunk = f"[文件: {fname}]\n{text}"
                chunks.append(chunk)
    return chunks


def build_index():
    chunks = load_docs()
    if not chunks:
        print("⚠️ docs 目录为空，无法构建 RAG 语料。")
        return

    print(f"加载到 {len(chunks)} 个文本块，开始生成嵌入向量...")
    embeddings = get_embeddings(chunks)
    embeddings = np.array(embeddings, dtype="float32")  # [N, D]

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    # 保存 chunks + embeddings 到一个 npz 文件
    np.savez(SAVE_PATH, chunks=np.array(chunks), embeddings=embeddings)
    print(f"✅ 向量语料已保存到 {SAVE_PATH}")


if __name__ == "__main__":
    build_index()
