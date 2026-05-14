"""
根据用户问题：
1）加载之前保存的 chunks + embeddings
2）用通义 embedding 计算问题向量
3）用 cosine_similarity 找到 top_k 相关文本
4）把这些文本 + 问题喂给 Qwen 生成答案
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SAVE_PATH = "../rag/corpus_embeddings.npz"
EMBEDDING_MODEL = "text-embedding-v4"
CHAT_MODEL = "qwen-plus"   # 或 "qwen-turbo" 等
#os.environ["DASHSCOPE_API_KEY"] = ""
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def get_embeddings(texts, model=EMBEDDING_MODEL):
    completion = client.embeddings.create(
        model=model,
        input=texts
    )
    embeddings = [data.embedding for data in completion.data]
    return embeddings


def load_corpus():
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"找不到 {SAVE_PATH}，请先运行 rag_build_index_tongyi.py 来构建语料向量。"
        )
    data = np.load(SAVE_PATH, allow_pickle=True)
    chunks = data["chunks"].tolist()         # List[str]
    embeddings = data["embeddings"]          # np.ndarray [N, D]
    return chunks, embeddings


def get_top_k_similar_documents(question_embedding, chunks_embeddings, chunks, top_k=3):
    similarities = cosine_similarity([question_embedding], chunks_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    top_k_chunks = [chunks[i] for i in top_k_indices]
    return top_k_chunks


def get_response(messages, model=CHAT_MODEL, temperature=0.7):
    """调用通义聊天模型，返回文本。"""
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def rag_answer(question: str, top_k: int = 3) -> str:
    """
    对外暴露的主函数：给一个 question，返回基于文档的回答。
    """
    chunks, chunks_embeddings = load_corpus()

    # 1. 问题向量
    q_emb = get_embeddings([question])[0]

    # 2. 找 top_k 文本块
    top_k_chunks = get_top_k_similar_documents(q_emb, chunks_embeddings, chunks, top_k)
    separator = "\n" + "=" * 30 + "\n"
    contextual_text = separator.join(top_k_chunks)

    # 3. 构建 Prompt，让模型“只根据这些文章回答”
    question_answering_prompt = f"""
你是一位专业的中文问答助手，擅长根据提供的文章内容回答用户的问题。
请注意：
- 优先使用文章中的信息
- 如果文章里没有提到，不要乱编，可以说“资料中没有明确说明”

文章内容：
'''
{contextual_text}
'''

用户问题：
'''
{question}
'''

请用简洁、自然的中文回答用户。
"""

    messages = [
        {"role": "system", "content": "你是阳光学院校园助手。"},
        {"role": "user", "content": question_answering_prompt},
    ]
    answer_result = get_response(messages)
    return answer_result


if __name__ == "__main__":
    q = "阳光学院图书馆主要有哪些功能？"
    print(rag_answer(q))
