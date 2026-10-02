from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rag import ask as rag_ask

app = FastAPI(
    title="AI知识库接口",
    description="FastAPI + LangChain RAG",
    version="1.0.0"
)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

@app.get("/")
def index():
    return FileResponse("static/index.html")



@app.get("/")
def root():
    return {
        "message": "服务已启动",
        "docs": "/docs"
    }

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

# 3. 保留GET版，方便浏览器直接测
@app.get("/ask")
def ask_get(q: str = ""):
    if not q:
        return {
            "question": "",
            "answer": "请提供问题，例如 /ask?q=什么是RAG",
            "sources": []
        }
    result = rag_ask(q)
    return {
        "question": q,
        "answer": result["answer"],
        "sources": result["sources"]
    }

# 4. 新增POST版
@app.post("/ask", response_model=AskResponse)
def ask_post(req: AskRequest):
    result = rag_ask(req.question)
    return {
        "question": req.question,
        "answer": result["answer"],
        "sources": result["sources"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)