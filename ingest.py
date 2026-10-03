# SPDX-License-Identifier: MIT

"""上传文件的入库处理：读取 → 切分 → 打 metadata → 分批存入 Chroma。

main.py（上传接口）和 vector.py（首次初始化种子数据）都调用这里的公共函数。

关于 Key：
向量库是按 Key 现建的（见 rag.get_vectorstore），这里每进来一个操作就带上
调用方的 api_key，用完即弃，不在模块级别缓存。

关于向量库句柄：
同一份数据只用一个 collection，clear_all() 用 reset_collection() 删集合再重建，
不会留下失效的句柄。
"""

import os
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import get_vectorstore

UPLOAD_DIR = "data/uploads"

# 第一步只支持纯文本
ALLOWED_EXTENSIONS = {".txt", ".md"}

# 智谱 embedding-2 单次请求最多 64 条 input，超过报 1214 错误。
# 入库时按这个批大小分批调用，和"文件能传多大"无关。
EMBED_BATCH_SIZE = 64

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""],
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
    return len(docs)


def delete_source(source: str, api_key: str) -> None:
    """只删这个文件产生的 chunk，别的文件不受影响。"""
    get_vectorstore(api_key)._collection.delete(where={"source": source})


def clear_all(api_key: str) -> None:
    """清空整个向量库。

    不用 collection.delete(where={})（有些 Chroma 版本不支持空 where），
    改成删掉集合再重建，最干净。
    """
    get_vectorstore(api_key).reset_collection()


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
