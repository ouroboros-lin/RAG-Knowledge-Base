import os
from datetime import datetime

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import ingest
import rag
from rag import ask as rag_ask

app = FastAPI(
    title="AI知识库接口",
    description="FastAPI + LangChain RAG（智谱 Embedding + 智谱 glm 生成）",
    version="1.2.0"
)

# 上传时每次读/写 1MB，避免把整个文件读进内存
UPLOAD_CHUNK_SIZE = 1024 * 1024


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


# ---------- 数据模型 ----------

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

class UploadResponse(BaseModel):
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
    ensure_key_usable(api_key)
    try:
        result = rag_ask(q, api_key)
    except HTTPException:
        raise
    except Exception as e:
        raise zhipu_http_error(e, api_key)
    return {
        "question": q,
        "answer": result["answer"],
        "sources": result["sources"]
    }

# 4. 新增POST版
@app.post("/ask", response_model=AskResponse)
def ask_post(req: AskRequest, api_key: str = Depends(get_zhipu_key)):
    ensure_key_usable(api_key)
    try:
        result = rag_ask(req.question, api_key)
    except HTTPException:
        raise
    except Exception as e:
        raise zhipu_http_error(e, api_key)
    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


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

    try:
        chunks = ingest.ingest_file(dest, api_key)
    except Exception as e:
        raise zhipu_http_error(e, api_key)

    # 入库成功，记下这是哪把 Key 算出来的向量
    rag.write_fingerprint(api_key)

    return {
        "filename": filename,
        "chunks": chunks,
        "message": "上传成功"
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
    return {"message": "知识库已清空"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
