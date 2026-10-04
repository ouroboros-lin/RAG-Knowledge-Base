# SPDX-License-Identifier: MIT

"""上传文件的入库处理：读取 → 切分 → 打 metadata → 分批存入 Chroma。

main.py（上传接口、粘贴文本接口）和 vector.py（首次初始化种子数据）都调用这里的公共函数。

关于 Key：
向量库是按 Key 现建的（见 rag.get_vectorstore），这里每进来一个操作就带上
调用方的 api_key，用完即弃，不在模块级别缓存。

关于向量库句柄：
同一份数据只用一个 collection，clear_all() 用 reset_collection() 删集合再重建，
不会留下失效的句柄。

关于缓存：
这里几个写操作（add_text / delete_source / clear_all）会顺手把 rag 的检索缓存清掉，
所以只要知识库有变动，缓存就不会留旧数据。
"""

import os
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import clear_cache, get_vectorstore

# ---------- 可调参数 ----------

UPLOAD_DIR = "data/uploads"              # 上传的原文件存这里

# 第一步只支持纯文本
ALLOWED_EXTENSIONS = {".txt", ".md"}

CHUNK_SIZE = 100                         # 每块多少字
CHUNK_OVERLAP = 20                       # 相邻块重叠多少字
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]

# 智谱 embedding-2 单次请求最多 64 条 input，超过报 1214 错误。
# 入库时按这个批大小分批调用，和"文件能传多大"无关。
EMBED_BATCH_SIZE = 64

MAX_PASTE_CHARS = 200_000                # 网页上直接粘贴的文本最多多少字
PASTE_NAME_MAX = 60                      # 粘贴文本自动生成的文件名最多多少字

# Windows 文件名里不能出现的字符
_UNSAFE_NAME_CHARS = '\\/:*?"<>|'

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=CHUNK_SEPARATORS,
)


def is_allowed(filename: str) -> bool:
    """只放行 .txt 和 .md。"""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def ensure_upload_dir() -> None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_text_file(path: str) -> str:
    """读文本。UTF-8 读不了就退到 GBK，再不行就忽略坏字节，尽量不抛异常。"""
    for encoding in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_documents(text: str, source: str, upload_time: str) -> list[Document]:
    """切分并给每个 chunk 打上 source / chunk_index / upload_time。"""
    chunks = [c for c in _splitter.split_text(text) if c.strip()]
    return [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_index": i,
                "upload_time": upload_time,
            },
        )
        for i, chunk in enumerate(chunks)
    ]


def add_text(text: str, source: str, api_key: str,
             upload_time: str | None = None) -> int:
    """切分 + 入库，返回 chunk 数。

    分批提交：智谱 embedding-2 单次请求最多 64 条 input，
    一次性把所有 chunk 丢过去，文件稍微大一点就会报
    "input数组最大不得超过64条"。按 EMBED_BATCH_SIZE 一批批入库，
    文件多大都不受这个限制。

    不指定 chunk id，交给 LangChain/Chroma 自动生成 UUID，
    这样重复上传同名文件也不会因为 id 冲突写不进去。
    """
    if upload_time is None:
        upload_time = now_str()
    docs = split_documents(text, source, upload_time)

    vectorstore = get_vectorstore(api_key)
    for i in range(0, len(docs), EMBED_BATCH_SIZE):
        vectorstore.add_documents(docs[i:i + EMBED_BATCH_SIZE])
    clear_cache()
    return len(docs)


def delete_source(source: str, api_key: str) -> None:
    """只删这个文件产生的 chunk，别的文件不受影响。"""
    get_vectorstore(api_key)._collection.delete(where={"source": source})
    clear_cache()


def clear_all(api_key: str) -> None:
    """清空整个向量库。

    不用 collection.delete(where={})（有些 Chroma 版本不支持空 where），
    改成删掉集合再重建，最干净。
    """
    get_vectorstore(api_key).reset_collection()
    clear_cache()


def count(api_key: str) -> int:
    """当前向量库里有多少 chunk。"""
    return get_vectorstore(api_key)._collection.count()


def ingest_file(path: str, api_key: str, source: str | None = None,
                upload_time: str | None = None) -> int:
    """把一个文件读进来入库。同名 source 的旧向量会先被删掉，保证不重复。"""
    if source is None:
        source = os.path.basename(path)
    if upload_time is None:
        upload_time = now_str()
    delete_source(source, api_key)
    return add_text(read_text_file(path), source, api_key, upload_time)


def paste_filename(title: str) -> str:
    """给粘贴的文本起个文件名：清掉路径和非法字符，兜底用时间戳。

    这样粘贴进来的内容也会老老实实躺在 data/uploads/ 里，
    网页文件列表里看得到、也删得掉，和上传的文件一个待遇。
    """
    name = os.path.basename((title or "").strip())
    name = os.path.splitext(name)[0]      # 用户自己带了 .md 之类的后缀就去掉
    name = "".join(ch for ch in name
                   if ch not in _UNSAFE_NAME_CHARS and ch.isprintable())
    name = name.strip(" .")
    if not name:
        name = "粘贴文本-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    return name[:PASTE_NAME_MAX] + ".txt"


def add_pasted_text(title: str, text: str, api_key: str) -> tuple[str, int]:
    """把网页上粘贴的文本当成一个 .txt 文件存下来再入库。

    返回 (文件名, 块数)。同名文件会覆盖旧的，和上传一致。
    """
    ensure_upload_dir()
    filename = paste_filename(title)
    with open(os.path.join(UPLOAD_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text)
    return filename, ingest_file(os.path.join(UPLOAD_DIR, filename), api_key)