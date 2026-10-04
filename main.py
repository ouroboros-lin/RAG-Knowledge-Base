# SPDX-License-Identifier: MIT

import json
import logging
import os
import time
from datetime import datetime

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import ingest
import rag
from rag import ask as rag_ask

# ---------- 可调参数 ----------

APP_VERSION = "1.3.0"

# 允许跨域访问的来源。本机自用就放 "*"；想收紧就改成
# ["http://127.0.0.1:8000", "http://localhost:8000"] 这样。
CORS_ALLOW_ORIGINS = ["*"]

MAX_QUESTION_CHARS = 2000            # 单个问题最多多少字
UPLOAD_CHUNK_SIZE = 1024 * 1024      # 上传时每次读/写 1MB，不把整个文件读进内存
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

logger = logging.getLogger("ragkb")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

app = FastAPI(
    title="AI知识库接口",
    description="FastAPI + LangChain RAG（智谱 Embedding + 智谱 glm 生成）",
    version=APP_VERSION
)

# 前端和接口同源，本来不需要 CORS；加上是为了方便你把页面放到别的地方调这个服务。
# 不用 allow_credentials：Key 走 X-Zhipu-Key 请求头，不靠 Cookie。
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 日志和耗时 ----------

# 日志里只可能出现：请求方法、路径、状态码、耗时、文件名、块数、字数。
# 不会出现：完整 API Key、文件全文、完整 Prompt、URL 里的查询串（问题原文在里面）。
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        logger.exception("%s %s -> 500 (%.0f ms)",
                         request.method, request.url.path, elapsed)
        raise
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.0f ms)",
                request.method, request.url.path, response.status_code, elapsed)
    return response


# ---------- 依赖：从前端请求头里拿智谱 Key ----------

def get_zhipu_key(x_zhipu_key: str | None = Header(default=None, alias="X-Zhipu-Key")) -> str:
    """从 X-Zhipu-Key 请求头取 Key。

    取到的只是这一瞬间的一个字符串，不写全局变量、不落盘、不进日志。
    """
    if not x_zhipu_key or not x_zhipu_key.strip():
        raise HTTPException(status_code=401, detail="请先输入智谱 API Key")
    return x_zhipu_key.strip()


# ---------- 智谱报错 → HTTP 状态码 ----------

# 智谱的业务错误码
AUTH_CODES = {1000, 1001, 1002, 1003, 1004}          # 认证失败
QUOTA_CODES = {1110, 1111, 1112, 1113, 1114}         # 账户异常 / 欠费 / 额度不足


def zhipu_business_code(e: Exception) -> int:
    """从智谱 SDK 的异常里掏出它的业务错误码，掏不到就返回 0。"""
    response = getattr(e, "response", None)
    if response is None:
        return 0
    try:
        return int(response.json().get("error", {}).get("code") or 0)
    except Exception:
        return 0


def zhipu_http_error(e: Exception, api_key: str) -> HTTPException:
    """把智谱的报错翻成合适的 HTTP 状态码。

    顺手把 Key 从报错文本里抹掉，避免 Key 顺着错误信息漏到前端或日志里。
    """
    status = getattr(e, "status_code", None) or 0
    code = zhipu_business_code(e)
    message = str(e).replace(api_key, "***") if api_key else str(e)

    if status == 401 or code in AUTH_CODES:
        return HTTPException(status_code=401, detail="API Key 无效")
    if status in (402, 403) or code in QUOTA_CODES:
        return HTTPException(status_code=403, detail="智谱账户额度不足，请检查余额")
    if status == 429 or code == 1301:
        return HTTPException(status_code=429, detail="请求太频繁，请稍后再试")
    return HTTPException(status_code=500, detail=f"调用智谱失败：{message}")


def ensure_key_usable(api_key: str) -> None:
    """知识库里已经有向量、但 Key 换了，就拒绝检索。

    向量是用 Key 算出来的，换一把 Key 就是两套坐标系，检索结果没有意义。
    """
    if ingest.count(api_key) > 0 and rag.key_changed(api_key):
        raise HTTPException(status_code=409, detail="Key 已更换，请重新上传知识库")


def check_question(question: str) -> str:
    """问题去空白 + 限长。"""
    text = (question or "").strip()
    if len(text) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"问题太长了，最多 {MAX_QUESTION_CHARS} 字")
    return text


def to_history(items) -> list[dict] | None:
    """把请求里的 history 转成 rag 认识的 [{"role", "content"}, ...]。"""
    if not items:
        return None
    return [{"role": item.role, "content": item.content} for item in items]


def sse(payload: dict) -> str:
    """一条 Server-Sent Events 消息。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


# ---------- 数据模型 ----------

class HistoryItem(BaseModel):
    """一轮对话。role 取 user 或 assistant，其它值一律当 assistant 处理。"""
    role: str = "user"
    content: str = ""

class AskRequest(BaseModel):
    question: str
    # 多轮对话上下文，可选。不传就和以前完全一样（单轮问答）。
    history: list[HistoryItem] | None = None

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str

class TextRequest(BaseModel):
    text: str
    # 给这段文本起个名字，空了就用「粘贴文本-时间戳」
    title: str = ""

class TextResponse(BaseModel):
    filename: str
    chunks: int
    message: str

class FileInfo(BaseModel):
    filename: str
    size: int
    upload_time: str

class FileListResponse(BaseModel):
    files: list[FileInfo]

class MessageResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    time: str
    chunks: int
    key_bound: bool
    cache_size: int


# ---------- 页面 ----------

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/hello")
def hello():
    return {
        "message": "你好，这是我的第一个FastAPI接口"
    }

@app.get("/hello/{name}")
def hello_name(name: str):
    return {
        "message": f"你好，{name}"
    }


# ---------- 健康检查 ----------

@app.get("/health", response_model=HealthResponse)
def health():
    """健康检查：不需要 Key，只回答"服务活着没、库里有多少块"。

    key_bound 只是个布尔值，不会把指纹本身暴露出去。
    """
    return {
        "status": "ok",
        "version": APP_VERSION,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": rag.count_chunks(),
        "key_bound": rag.read_fingerprint() is not None,
        "cache_size": rag.cache_size(),
    }


# ---------- 问答 ----------

# 3. 保留GET版，方便直接用 curl 测（要带上 X-Zhipu-Key 请求头）
@app.get("/ask")
def ask_get(q: str = "", api_key: str = Depends(get_zhipu_key)):
    if not q:
        return {
            "question": "",
            "answer": "请提供问题，例如 /ask?q=什么是RAG",
            "sources": []
        }
    question = check_question(q)
    ensure_key_usable(api_key)
    try:
        result = rag_ask(question, api_key)
    except HTTPException:
        raise
    except Exception as e:
        raise zhipu_http_error(e, api_key)
    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    }

# 4. 新增POST版
@app.post("/ask", response_model=AskResponse)
def ask_post(req: AskRequest, api_key: str = Depends(get_zhipu_key)):
    question = check_question(req.question)
    ensure_key_usable(api_key)
    history = to_history(req.history)
    try:
        result = rag_ask(question, api_key, history)
    except HTTPException:
        raise
    except Exception as e:
        raise zhipu_http_error(e, api_key)
    return {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"]
    }

# 5. 流式版：新端点，老的 /ask 一点没动
@app.post("/ask/stream")
def ask_stream(req: AskRequest, api_key: str = Depends(get_zhipu_key)):
    """流式问答，Server-Sent Events。

    每条消息都是一个 JSON：先一条 {"type": "sources", "sources": [...]}，
    接着一串 {"type": "delta", "text": "..."}，最后一条 {"type": "done"}。
    中途出错就发一条 {"type": "error", "detail": "..."}，之后照样跟一条 done。

    参数校验、Key 校验、Key 已更换这些错误都在推流开始之前，
    所以它们还是正常的 HTTP 状态码（400 / 401 / 409），不会塞进 SSE 里。

    日志里只记问题字数，不记问题原文。
    """
    question = check_question(req.question)
    ensure_key_usable(api_key)
    history = to_history(req.history)

    def event_stream():
        start = time.perf_counter()

        def elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000

        try:
            for item in rag.ask_stream(question, api_key, history):
                yield sse(item)
        except GeneratorExit:
            # 客户端提前断开，别再往一条已经没人读的连接里写东西
            logger.info("POST /ask/stream 客户端断开 问题=%d字 耗时=%.0f ms",
                        len(question), elapsed_ms())
            raise
        except Exception as e:
            detail = zhipu_http_error(e, api_key).detail
            logger.warning("POST /ask/stream 出错：%s", detail)
            yield sse({"type": "error", "detail": detail})

        yield sse({"type": "done"})
        logger.info("POST /ask/stream 完成 问题=%d字 耗时=%.0f ms",
                    len(question), elapsed_ms())

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 知识库管理 ----------

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), api_key: str = Depends(get_zhipu_key)):
    """上传一个 .txt / .md 文件，保存原文件并入库。同名文件会覆盖旧的。"""
    filename = os.path.basename(file.filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    if not ingest.is_allowed(filename):
        raise HTTPException(status_code=400, detail="只支持 .txt 和 .md 文件")

    # 库里已经有别人家的向量了，就别再往里掺新 Key 的向量
    if ingest.count(api_key) > 0 and rag.key_changed(api_key):
        raise HTTPException(status_code=409,
                            detail="Key 已更换，请先点「清空知识库」，再重新上传")

    ingest.ensure_upload_dir()
    dest = os.path.join(ingest.UPLOAD_DIR, filename)

    # 先落盘保存原文件，再入库；同名时直接覆盖。
    # 按 1MB 分块读、分块写，不把整个文件读进内存，多大的文件都不会把内存撑爆。
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)

    start = time.perf_counter()
    try:
        chunks = ingest.ingest_file(dest, api_key)
    except Exception as e:
        raise zhipu_http_error(e, api_key)

    # 入库成功，记下这是哪把 Key 算出来的向量
    rag.write_fingerprint(api_key)
    logger.info("入库完成 文件=%s 块数=%d 耗时=%.0f ms",
                filename, chunks, (time.perf_counter() - start) * 1000)

    return {
        "filename": filename,
        "chunks": chunks,
        "message": "上传成功"
    }


@app.post("/ingest_text", response_model=TextResponse)
def ingest_text(req: TextRequest, api_key: str = Depends(get_zhipu_key)):
    """把一段文本直接入库，不用先存成文件。

    文本会以 .txt 的形式落到 data/uploads/ 里（文件名取自 title），
    所以它和上传的文件一样，能在文件列表里看到、也能单独删掉。
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="文本内容不能为空")
    if len(text) > ingest.MAX_PASTE_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"文本太长了，最多 {ingest.MAX_PASTE_CHARS} 字")

    if ingest.count(api_key) > 0 and rag.key_changed(api_key):
        raise HTTPException(status_code=409,
                            detail="Key 已更换，请先点「清空知识库」，再重新入库")

    start = time.perf_counter()
    try:
        filename, chunks = ingest.add_pasted_text(req.title, text, api_key)
    except Exception as e:
        raise zhipu_http_error(e, api_key)

    rag.write_fingerprint(api_key)
    logger.info("粘贴入库完成 文件=%s 字数=%d 块数=%d 耗时=%.0f ms",
                filename, len(text), chunks, (time.perf_counter() - start) * 1000)

    return {
        "filename": filename,
        "chunks": chunks,
        "message": "已入库"
    }


@app.get("/files", response_model=FileListResponse)
def list_files():
    """列出 data/uploads/ 里已上传的文件。只看本地文件，不用 Key。"""
    ingest.ensure_upload_dir()
    files = []
    for name in sorted(os.listdir(ingest.UPLOAD_DIR)):
        path = os.path.join(ingest.UPLOAD_DIR, name)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        files.append({
            "filename": name,
            "size": stat.st_size,
            "upload_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        })
    return {"files": files}


@app.delete("/files/{filename}", response_model=MessageResponse)
def delete_file(filename: str, api_key: str = Depends(get_zhipu_key)):
    """删掉一个文件：原文件和它在向量库里的 chunk 一起删。

    这里不校验指纹——删东西不需要 Embedding，换过 Key 也应该能删。
    """
    filename = os.path.basename(filename)
    path = os.path.join(ingest.UPLOAD_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"文件不存在：{filename}")

    os.remove(path)
    ingest.delete_source(filename, api_key)
    logger.info("已删除文件 %s 及其向量", filename)
    return {"message": f"已删除 {filename}"}


@app.delete("/files", response_model=MessageResponse)
def clear_files(api_key: str = Depends(get_zhipu_key)):
    """清空知识库：删除所有原文件，清空向量库，并解绑 Key。

    同样不校验指纹——这正是换 Key 之后该走的补救路径。
    """
    ingest.ensure_upload_dir()
    for name in os.listdir(ingest.UPLOAD_DIR):
        path = os.path.join(ingest.UPLOAD_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

    ingest.clear_all(api_key)
    rag.clear_fingerprint()
    logger.info("知识库已清空，检索缓存一并作废")
    return {"message": "知识库已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)