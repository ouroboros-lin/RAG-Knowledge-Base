# SPDX-License-Identifier: MIT

"""RAG 链：检索 + 生成。

全部走智谱：Embedding 用 embedding-2，生成回答用 glm。
Key 由调用方（网页上用户填的那把）传进来.

本文件顶部的常量就是检索/生成相关的全部可调参数，改这里就行。

关于相似度（重要约定）：
    向量库统一按 **cosine 距离** 建，见 get_vectorstore 里的
    collection_metadata={"hnsw:space": "cosine"}。

    Chroma 返回的 score 是**距离**不是相似度，越小越相关；cosine 空间下
        距离 = 1 - 余弦相似度
    所以本文件里一律用换算后的「相似度」：
        相似度 = 1 - 距离，裁剪到 0~1，越大越相关。
    缓存里存的、merge_adjacent 合并时比的、对外返回的 sources 里的 score，
    全程都是这个相似度，不再出现原始距离。

    注意：这个约定是后加的。**以前建的向量库用的是 Chroma 默认的 l2 空间**，
    换成 cosine 之后旧向量不会自动重建，`1 - score` 算出来的分数没有意义
    （l2 下距离可以远大于 1，clamp 之后会一片 0）。所以升级后必须
    「清空知识库」再用同一把 Key 重新上传。
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

    collection_metadata 指定用 cosine 距离——Chroma 默认是 l2，
    那样 similarity_search_with_score 给出的分数没法换算成 0~1 的相似度。

    注意：metadata 只在**建集合**时生效。集合已经存在的话，
    get_or_create_collection 会直接把旧的拿回来，这里的设置会被忽略——
    所以老库必须清空重建（ingest.clear_all 走 reset_collection，
    删掉再建，会带上这份 metadata）。
    """
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=ZhipuEmbeddings(api_key=api_key),
        collection_metadata={"hnsw:space": "cosine"},
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
# 缓存的值是 [(Document, 相似度), ...]——相似度一起存，
# 不然命中缓存的那次就拿不到分数了（而来源里要显示它）。
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
        expire_at, pairs = item
        if expire_at < time.time():
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return pairs


def _cache_put(key: tuple, pairs: list) -> None:
    with _cache_lock:
        _cache[key] = (time.time() + CACHE_TTL_SECONDS, pairs)
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


def merge_adjacent(pairs: list) -> list:
    """去重 + 把同一来源里连续的块拼回去，最多留 TOP_K 条。

    进出的都是 [(Document, 相似度), ...]，相似度 0~1，越大越相关。

    1. 来源和内容都一样的只留相似度最高的那条；
    2. 同一个文件里 chunk_index 是连续的几块，说明本来就是一段话被切开的，
       拼回一整段（拼的时候去掉重叠部分），合并后这条的相似度**取组内最高的**；
    3. 按相似度从高到低排，再截到 TOP_K。
    """
    # 去重，顺便复制一份 Document，避免改到缓存里的对象
    best: dict[tuple, tuple[float, Document]] = {}
    for doc, score in pairs:
        mark = (doc.metadata.get("source", ""), doc.page_content)
        if mark not in best or score > best[mark][0]:
            best[mark] = (score, Document(page_content=doc.page_content,
                                          metadata=dict(doc.metadata)))

    # 按来源分组，组内按块序号排好，再合并连续的块
    groups: dict[str, list] = {}
    for score, doc in best.values():
        groups.setdefault(doc.metadata.get("source", ""), []).append((score, doc))

    merged = []
    for items in groups.values():
        items.sort(key=lambda item: _chunk_index(item[1]))
        top_score, current = items[0]
        current_index = _chunk_index(current)
        for score, doc in items[1:]:
            index = _chunk_index(doc)
            if index == current_index + 1:
                current.page_content = _join_overlap(current.page_content,
                                                     doc.page_content)
                # 合并后的相似度取组内最高的那个；chunk_index 保持第一块的，
                # 这样来源里显示的「第 N 块」是这段话开头所在的块
                top_score = max(top_score, score)
                current_index = index
            else:
                merged.append((current, top_score))
                top_score, current, current_index = score, doc, index
        merged.append((current, top_score))

    merged.sort(key=lambda item: item[1], reverse=True)
    return merged[:TOP_K]


def search_with_score(query: str, api_key: str) -> list:
    """查向量库，返回 [(Document, 相似度), ...]。

    用 similarity_search_with_score 拿到的 score 是**距离**（越小越相关），
    cosine 空间下 距离 = 1 - 余弦相似度，所以相似度 = 1 - 距离。
    再 clamp 到 0~1：浮点误差可能让 1.0 变成 1.0000000000000002，
    而余弦为负的向量算出来会是负数。
    """
    hits = get_vectorstore(api_key).similarity_search_with_score(query, k=CANDIDATE_K)
    return [(doc, _to_similarity(distance)) for doc, distance in hits]


def _to_similarity(distance: float) -> float:
    """cosine 距离 → 相似度，裁剪到 0~1。"""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve(query: str, api_key: str) -> list:
    """检索相关资料。先看缓存，没命中才真的去查向量库。

    返回 [(Document, 相似度), ...]，Document 是共享的，调用方别去改它。
    """
    key = (key_fingerprint(api_key), query.strip())
    cached = _cache_get(key)
    if cached is not None:
        return cached

    pairs = merge_adjacent(search_with_score(query, api_key))
    _cache_put(key, pairs)
    return pairs


# ---------- 生成 ----------

def format_docs(pairs) -> str:
    """把 [(Document, 相似度), ...] 拼成给 LLM 看的资料文本。"""
    return "\n".join(doc.page_content for doc, _ in pairs)


def source_items(pairs) -> list[dict]:
    """[(Document, 相似度), ...] → 对外的来源列表。

    每项 {"text": 片段正文, "source": 文件名, "chunk_index": 块序号, "score": 相似度}。
    score 保留 4 位小数，够用又不至于带出一串浮点尾巴（前端显示两位）。
    """
    items = []
    for doc, score in pairs:
        items.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", ""),
            "chunk_index": doc.metadata.get("chunk_index"),
            "score": round(float(score), 4),
        })
    return items


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


def stream_answer(query: str, pairs: list, api_key: str, history=None):
    """流式生成回答：一小段一小段地往外吐。"""
    turns = format_history(history)
    template = prompt_with_history if turns else prompt
    chain = template | build_llm(api_key) | StrOutputParser()

    payload = {"context": format_docs(pairs), "question": query}
    if turns:
        payload["history"] = turns

    yield from chain.stream(payload)


def ask(query: str, api_key: str, history=None):
    """检索 + 生成。调用方负责先确认 Key 没换（key_changed）。

    history 可以不传，不传时和以前的行为完全一样。
    sources 是 [{"text", "source", "chunk_index", "score"}, ...]。
    """
    pairs = retrieve(query, api_key)
    answer = "".join(stream_answer(query, pairs, api_key, history))

    return {
        "answer": answer,
        "sources": source_items(pairs)
    }


def ask_stream(query: str, api_key: str, history=None):
    """流式问答。先吐出检索到的来源，再一段段吐回答。

    产出的是普通 dict，转成什么协议由调用方决定：
      {"type": "sources", "sources": [{"text", "source", "chunk_index", "score"}, ...]}
      {"type": "delta", "text": "..."}
    """
    pairs = retrieve(query, api_key)
    yield {"type": "sources", "sources": source_items(pairs)}
    for piece in stream_answer(query, pairs, api_key, history):
        yield {"type": "delta", "text": piece}


if __name__ == "__main__":
    import getpass

    result = ask("什么是RAG", getpass.getpass("请输入智谱 API Key: ").strip())
    print("回答：", result["answer"])
    print("来源：")
    for s in result["sources"]:
        print(f"- {s['source']} 第 {s['chunk_index']} 块 相似度 {s['score']:.2f}")
        print(f"  {s['text']}")