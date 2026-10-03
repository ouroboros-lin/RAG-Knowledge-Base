# SPDX-License-Identifier: MIT

"""RAG 链：检索 + 生成。

全部走智谱：Embedding 用 embedding-2，生成回答用 glm。
Key 由调用方（网页上用户填的那把）传进来.
"""

import hashlib
import os

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from embeddings import ZhipuEmbeddings

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "knowledge"
LLM_MODEL = "glm-4-flash"

# 只存 Key 的 SHA256 前 8 位，用来判断"是不是换了 Key"。
# 存的是哈希，不是 Key 本身，也没法反推出 Key。
FINGERPRINT_FILE = "data/key_fingerprint.txt"

prompt = ChatPromptTemplate.from_template("""
请根据下面提供的资料回答问题。
要求：
1. 只使用资料里的信息，不要编造；
2. 如果资料里没有答案，就说“资料中没有提到”；
3. 回答要通俗，2-3句话。

资料：
{context}

问题：{question}
""")


# ---------- Key 指纹 ----------

def key_fingerprint(api_key: str) -> str:
    """Key 的 SHA256 前 8 位。只用来比对，不能还原出 Key。"""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:8]


def read_fingerprint() -> str | None:
    if not os.path.exists(FINGERPRINT_FILE):
        return None
    with open(FINGERPRINT_FILE, "r", encoding="utf-8") as f:
        saved = f.read().strip()
    return saved or None


def write_fingerprint(api_key: str) -> None:
    os.makedirs(os.path.dirname(FINGERPRINT_FILE), exist_ok=True)
    with open(FINGERPRINT_FILE, "w", encoding="utf-8") as f:
        f.write(key_fingerprint(api_key))


def clear_fingerprint() -> None:
    """知识库清空后，Key 和向量库的绑定也一起断掉。"""
    if os.path.exists(FINGERPRINT_FILE):
        os.remove(FINGERPRINT_FILE)


def key_changed(api_key: str) -> bool:
    """和当初入库用的 Key 不是同一把。

    没存过指纹（比如知识库是空的、或者指纹被删了）就当作没换过。
    """
    saved = read_fingerprint()
    return saved is not None and saved != key_fingerprint(api_key)


# ---------- 向量库 ----------

def get_vectorstore(api_key: str) -> Chroma:
    """按这个 Key 现建一个 Chroma 实例。

    每次请求现建，Key 只活在这一个请求里：
    不会在模块级别留一份，也不会被下一个请求复用。
    """
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=ZhipuEmbeddings(api_key=api_key),
    )


# ---------- 问答 ----------

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def build_llm(api_key: str) -> ChatZhipuAI:
    return ChatZhipuAI(model=LLM_MODEL, api_key=api_key, temperature=0.3)


def ask(query: str, api_key: str):
    """检索 + 生成。调用方负责先确认 Key 没换（key_changed）。"""
    vectorstore = get_vectorstore(api_key)
    docs = vectorstore.similarity_search(query, k=3)

    chain = prompt | build_llm(api_key) | StrOutputParser()
    answer = chain.invoke({
        "context": format_docs(docs),
        "question": query
    })

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }


if __name__ == "__main__":
    import getpass

    result = ask("什么是RAG", getpass.getpass("请输入智谱 API Key: ").strip())
    print("回答：", result["answer"])
    print("来源：")
    for s in result["sources"]:
        print("-", s)
