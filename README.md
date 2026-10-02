# 基于RAG的AI知识库问答系统

一个用 FastAPI + LangChain + Chroma + DeepSeek 搭建的极简 RAG 问答项目。  
把合法采集的文本，变成一个可以提问、能溯源、不瞎编的知识库接口，并带一个极简前端页面。
 
---

## 一、这是什么

RAG = 检索增强生成。  
通俗说：先查资料，再让 AI 回答，减少瞎编。

这个项目的流程：

用户提问
→ 前端页面
→ FastAPI 接收
→ LangChain 检索器去向量库找相关段落
→ 拼成资料
→ 发给 DeepSeek 生成回答
→ 返回答案和来源
→ 前端显示

---

## 二、项目结构

ai-crawler-lab/
  main.py              # FastAPI 入口，暴露 /ask 接口，托管前端
  rag.py               # RAG 链，检索 + 生成
  embeddings.py        # 智谱 Embedding 封装，供 LangChain 使用
  vector.py            # 建向量库：读 notes.txt → 切分 → 存 Chroma
  search.py            # 单独测试向量检索
  test_post.py         # 测试 POST /ask 接口
  static/
    index.html         # 极简前端页面（单文件，内联CSS和JS）
  data/
    knowledge/
      notes.txt        # 知识库原文
    chroma_db/         # Chroma 向量库（本地持久化）
  notes/               # 学习笔记
  .env                 # API Key（不上传、不分享）
  README.md            # 项目说明

---

## 三、依赖

- Python 3.10+
- FastAPI：提供接口
- LangChain：串联检索和生成
- Chroma：本地向量库
- DeepSeek：生成回答
- 智谱 Embedding：把文字变成向量

---

## 四、环境准备

### 1. 安装依赖

```bash
pip install fastapi "uvicorn[standard]" python-dotenv openai zhipuai chromadb langchain langchain-community langchain-chroma langchain-openai langchain-text-splitters
```

### 2. 配置 API Key

在项目根目录新建 `.env`：

```text
DEEPSEEK_API_KEY=你的DeepSeek Key
ZHIPU_API_KEY=你的智谱 Key
```

说明：
- DeepSeek 用于生成回答：https://platform.deepseek.com
- 智谱用于 Embedding：https://open.bigmodel.cn

注意：  
`.env` 里是钥匙，不能上传、不能截图、不能发给别人。

### 3. 准备知识库文本

编辑 `data/knowledge/notes.txt`，一行一条，写你要让 AI 回答的合法内容。  
只用公开、合规、你有权使用的文本。

---

## 五、使用步骤

### 第一步：建向量库

把 `notes.txt` 切分、变成向量、存进 Chroma：

```bash
cd /d D:\PythonProject\ai-crawler-lab
python vector.py
```

看到 `已存入向量库，共 N 条` 就成功了。

重建向量库：
1. 删掉 `data/chroma_db/` 文件夹。
2. 再跑一次 `python vector.py`。

### 第二步：测试检索（可选）

```bash
python search.py
```

会打印几个问题的检索结果，看相关性能不能接受。

### 第三步：启动 FastAPI

```bash
python main.py
```

或者：

```bash
python -m uvicorn main:app --reload
```

看到：

```text
Uvicorn running on http://127.0.0.1:8000
```

就成功了。

### 第四步：打开前端页面

浏览器打开：

```text
http://127.0.0.1:8000/
```

你会看到一个极简问答页面：
- 输入框和“提问”按钮
- 几个示例问题，点一下就能问
- 回答和来源显示在下方
- 支持回车提交
- 加载中会禁用按钮

原来的 `/docs` 依然可以访问，用来调试接口。

---

## 六、接口说明

### GET /ask

方便浏览器直接测：

```text
http://127.0.0.1:8000/ask?q=什么是RAG
```

返回：

```json
{
  "question": "什么是RAG",
  "answer": "RAG是检索增强生成，先查资料再让AI回答。",
  "sources": ["RAG 是检索增强生成，先查资料再让 AI 回答。"]
}
```

### POST /ask

前端页面实际调用的就是这个接口，参数放请求体里：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"什么是RAG\"}"
```

或用 `test_post.py`：

```bash
python test_post.py
```

### 资料里没有的问题

```text
http://127.0.0.1:8000/ask?q=今天天气怎么样
```

会返回：

```json
{
  "question": "今天天气怎么样",
  "answer": "资料中没有提到。",
  "sources": []
}
```

这叫**约束幻觉**：让 AI 别编。

---

## 七、每个文件做什么

| 文件 | 作用 |
|---|---|
| `main.py` | FastAPI 入口，提供 `/ask` 接口，托管前端页面 |
| `rag.py` | RAG 链，检索 + 生成 |
| `embeddings.py` | 智谱 Embedding 封装 |
| `vector.py` | 建向量库 |
| `search.py` | 单独测试向量检索 |
| `test_post.py` | 测试 POST 接口 |
| `static/index.html` | 极简前端页面，单文件，内联 CSS 和 JS |
| `data/knowledge/notes.txt` | 知识库原文 |
| `data/chroma_db/` | 向量库文件 |
| `notes/` | 学习笔记 |

---

## 八、常见问题

**Q：`ModuleNotFoundError: No module named 'xxx'`**  
A：没装依赖，跑一次安装命令。

**Q：`ZHIPU_API_KEY` 读不到**  
A：检查 `.env` 是否在项目根目录，等号两边不要有空格，不要加引号。

**Q：检索结果不相关**  
A：调 `vector.py` 里的 `chunk_size`（每块多少字）和 `chunk_overlap`（相邻块重叠多少字），然后删 `data/chroma_db/` 重建。

**Q：改了 `notes.txt` 但回答没变**  
A：向量库没重建。删 `data/chroma_db/`，重跑 `python vector.py`。

**Q：`/ask` 返回 500**  
A：先单独跑 `python rag.py`，看具体报错。

**Q：端口被占用**  
A：换端口：`uvicorn main:app --reload --port 8001`。

**Q：打开 `/` 显示 404**  
A：检查 `static/index.html` 是否存在于项目根目录下的 `static/` 文件夹，并且 `main.py` 里加了 `@app.get("/")` 返回该文件。

**Q：前端页面能打开但提问没反应**  
A：打开浏览器开发者工具（F12），看 Console 和 Network 里 `/ask` 请求是否报错。

