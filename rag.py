# SPDX-License-Identifier: MIT

"""RAG 链：检索 + 生成。

全部走智谱：Embedding 用 embedding-2，生成回答用 glm。
Key 由调用方（网页上用户填的那把）传进来.

本文件顶部的常量就是检索/生成相关的全部可调参数，改这里就行。
"""

import hashlib
import os
import threading
import time
from collections import OrderedDict

import chromadb
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from embeddings import ZhipuEmbeddings

# ---------- 可调参数 ----------

CHROMA_DIR = "data/chroma_db"        # 向量库落盘目录
COLLECTION_NAME = "knowledge"        # 集合名
LLM_MODEL = "glm-4-flash"            # 生成回答用的模型
LLM_TEMPERATURE = 0.3                # 生成温度，越低越稳

TOP_K = 3                            # 最终交给 LLM 的资料条数
CANDIDATE_K = 8                      # 先多检索几条候选，去重 / 合并相邻块后再截到 TOP_K

MAX_HISTORY_TURNS = 5                # 多轮对话最多带几轮（一问一答算一轮）
MAX_HISTORY_CHARS = 500              # 每轮对话最多保留多少字

CACHE_MAX_SIZE = 128                 # 检索结果缓存最多存多少条
CACHE_TTL_SECONDS = 300              # 缓存的兜底过期时间（秒）

# 只存 Key 的 SHA256 前 8 位，用来判断"是不是换了 Key"。
# 存的是哈希，不是 Key 本身，也没法反推出 Key。
FINGERPRINT_FILE = "data/key_fingerprint.txt"

# ---------- Prompt ----------

_RAG_INSTRUCTIONS = """请根据下面提供的资料回答问题。
要求：
1. 只使用资料里的信息，不要编造；
2. 如果资料里没有答案，就说“资料中没有提到”；
3. 回答要通俗，2-3句话。"""

# 只有传了 history 才会插进 Prompt，不传时下面的 prompt 和以前一模一样。
_RAG_HISTORY = """
以下是之前几轮对话，只用它来理解这次问的是什么，回答依然以资料为准：
{history}
"""

_RAG_TAIL = """
资料：
{context}

问题：{question}
"""

prompt = ChatPromptTemplate.from_template(
    "\n" + _RAG_INSTRUCTIONS + "\n" + _RAG_TAIL
)

prompt_with_history = ChatPromptTemplate.from_template(
    "\n" + _RAG_INSTRUCTIONS + "\n" + _RAG_HISTORY + _RAG_TAIL
)


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


def count_chunks() -> int:
    """向量库里现在有多少块。只数数，不用算 Embedding，所以不需要 Key。"""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection(COLLECTION_NAME).count()
    except Exception:
        # 还没建过库、或者库文件有问题，都当 0，别把健康检查搞挂
        return 0


# ---------- 检索结果缓存 ----------
#
# 缓存的是"检索结果"而不是最终回答：回答还跟 history 有关，缓存它容易张冠李戴，
# 而检索只是查向量库，键就是 (Key 指纹, 问题)，安全得多。
#
# Key 指纹进缓存键，所以换了 Key 天然命中不到旧缓存；
# 至于上传 / 删除 / 清空，由 ingest.py 里的写操作直接调 clear_cache()。
# CACHE_TTL_SECONDS 只是兜底：多进程（比如 --reload、多 worker）时缓存不共享，
# 万一有进程没清干净，超时后也会自己失效。

_cache: "OrderedDict[tuple, tuple[float, list]]" = OrderedDict()
_cache_lock = threading.Lock()


def clear_cache() -> None:
    """知识库一变（上传 / 删除 / 清空），缓存就整体作废。"""
    with _cache_lock:
        _cache.clear()


def cache_size() -> int:
    """当前缓存条数，健康检查里用。"""
    with _cache_lock:
        return len(_cache)


def _cache_get(key: tuple):
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        expire_at, docs = item
        if expire_at < time.time():
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return docs


def _cache_put(key: tuple, docs: list) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + CACHE_TTL_SECONDS, docs)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_SIZE:
            _cache.popitem(last=False)


# ---------- 检索 ----------

def _chunk_index(doc: Document) -> int:
    """块在原文件里的序号。没有序号的排到最后。"""
    index = doc.metadata.get("chunk_index")
    return index if isinstance(index, int) else 10 ** 9


def _join_overlap(head: str, tail: str) -> str:
    """把相邻两块拼起来，顺便去掉切分时留下的那段重叠文字。"""
    limit = min(len(head), len(tail))
    for size in range(limit, 0, -1):
        if head.endswith(tail[:size]):
            return head + tail[size:]
    return head + "\n" + tail


def merge_adjacent(docs: list) -> list:
    """去重 + 把同一来源里连续的块拼回去，最多留 TOP_K 条。

    1. 内容完全一样的只留最早出现的那条（相关度最高）；
    2. 同一个文件里 chunk_index 是连续的几块，说明本来就是一段话被切开的，
       拼回一整段，拼的时候把重叠部分去掉；
    3. 合并后按各自最靠前的相关度重新排，再截到 TOP_K。
    """
    # 去重，顺便复制一份，避免改到缓存 / 调用方的 Document
    unique = []
    seen = set()
    for doc in docs:
        mark = (doc.metadata.get("source", ""), doc.page_content)
        if mark in seen:
            continue
        seen.add(mark)
        unique.append(Document(page_content=doc.page_content,
                               metadata=dict(doc.metadata)))

    # 按来源分组，组内按块序号排好，再合并连续的块
    groups: dict[str, list] = {}
    for rank, doc in enumerate(unique):
        groups.setdefault(doc.metadata.get("source", ""), []).append((rank, doc))

    merged = []
    for items in groups.values():
        items.sort(key=lambda item: _chunk_index(item[1]))
        best_rank, current = items[0]
        current_index = _chunk_index(current)
        for rank, doc in items[1:]:
            index = _chunk_index(doc)
            if index == current_index + 1:
                current.page_content = _join_overlap(current.page_content,
                                                     doc.page_content)
                current.metadata["chunk_index"] = index
                current_index = index
                best_rank = min(best_rank, rank)
            else:
                merged.append((best_rank, current))
                best_rank, current, current_index = rank, doc, index
        merged.append((best_rank, current))

    merged.sort(key=lambda item: item[0])
    return [doc for _, doc in merged[:TOP_K]]


def retrieve(query: str, api_key: str) -> list:
    """检索相关资料。先看缓存，没命中才真的去查向量库。

    返回的 Document 是共享的，调用方别去改它。
    """
    key = (key_fingerprint(api_key), query.strip())
    cached = _cache_get(key)
    if cached is not None:
        return cached

    docs = get_vectorstore(api_key).similarity_search(query, k=CANDIDATE_K)
    docs = merge_adjacent(docs)
    _cache_put(key, docs)
    return docs


# ---------- 生成 ----------

def format_docs(docs) -> str:
    return "\n".join(doc.page_content for doc in docs)


def build_llm(api_key: str) -> ChatZhipuAI:
    return ChatZhipuAI(model=LLM_MODEL, api_key=api_key,
                       temperature=LLM_TEMPERATURE)


def format_history(history) -> str:
    """把多轮对话压成一段纯文本。

    history 是 [{"role": "user"/"assistant", "content": "..."}, ...]。
    只取最近 MAX_HISTORY_TURNS 轮，每轮最多 MAX_HISTORY_CHARS 字，
    免得 Prompt 越滚越长。
    """
    if not history:
        return ""

    lines = []
    for item in list(history)[-MAX_HISTORY_TURNS * 2:]:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS] + "……"
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def stream_answer(query: str, docs: list, api_key: str, history=None):
    """流式生成回答：一小段一小段地往外吐。"""
    turns = format_history(history)
    template = prompt_with_history if turns else prompt
    chain = template | build_llm(api_key) | StrOutputParser()

    payload = {"context": format_docs(docs), "question": query}
    if turns:
        payload["history"] = turns

    yield from chain.stream(payload)


def ask(query: str, api_key: str, history=None):
    """检索 + 生成。调用方负责先确认 Key 没换（key_changed）。

    history 可以不传，不传时和以前的行为完全一样。
    """
    docs = retrieve(query, api_key)
    answer = "".join(stream_answer(query, docs, api_key, history))

    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }


def ask_stream(query: str, api_key: str, history=None):
    """流式问答。先吐出检索到的来源，再一段段吐回答。

    产出的是普通 dict，转成什么协议由调用方决定：
      {"type": "sources", "sources": [...]}
      {"type": "delta", "text": "..."}
    """
    docs = retrieve(query, api_key)
    yield {"type": "sources", "sources": [doc.page_content for doc in docs]}
    for piece in stream_answer(query, docs, api_key, history):
        yield {"type": "delta", "text": piece}


if __name__ == "__main__":
    import getpass

    result = ask("什么是RAG", getpass.getpass("请输入智谱 API Key: ").strip())
    print("回答：", result["answer"])
    print("来源：")
    for s in result["sources"]:
        print("-", s)