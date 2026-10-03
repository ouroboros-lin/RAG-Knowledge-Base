# SPDX-License-Identifier: MIT

"""单独测试向量检索。"""

import getpass

import chromadb
from zhipuai import ZhipuAI

client = ZhipuAI(api_key=getpass.getpass("请输入智谱 API Key: ").strip())

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge")


def get_embedding(text):
    resp = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return resp.data[0].embedding


def vector_search(query, top_k=3):
    q_vec = get_embedding(query)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k
    )
    return results["documents"][0]


# 测试
for q in ["什么是RAG", "怎么让AI不乱编", "今天天气怎么样"]:
    print("问题：", q)
    docs = vector_search(q)
    for d in docs:
        print("-", d)
    print()
