# 基于RAG的AI知识库问答系统

一个用 FastAPI + LangChain + Chroma + 智谱 搭建的极简 RAG 问答项目。  
把合法采集的文本，变成一个可以提问、能溯源、不瞎编的知识库接口，并带一个极简前端页面。

Embedding 和生成回答都用智谱（`embedding-2` + `glm-4-flash`），一把 Key 搞定。  
**API Key 不放在后端**，由你在网页上输入，存在浏览器里。

flowchart LR
  U[浏览器<br/>localStorage 存 Key] -->|X-Zhipu-Key| F[FastAPI]
  F --> R[rag.py<br/>检索+生成]
  R --> C[(Chroma)]
  R --> Z[智谱<br/>embedding-2 + glm-4-flash]
  I[ingest.py] --> C
  F --> I



![主页页面](assets/网页主页页面展示.png)

 
---

## 一、这是什么

RAG = 检索增强生成。  
通俗说：先查资料，再让 AI 回答，减少瞎编。

这个项目的流程：

用户提问
→ 前端页面（带上 X-Zhipu-Key 请求头）
→ FastAPI 接收
→ LangChain 检索器去向量库找相关段落
→ 拼成资料
→ 发给智谱 glm 生成回答
→ 返回答案和来源
→ 前端显示

---

## 二、项目结构

RAG-Knowledge-Base/
  main.py              # FastAPI 入口：问答/上传/列表/删除/清空/健康检查，托管前端
  rag.py               # RAG 链，检索 + 生成 + 缓存
  embeddings.py        # 智谱 Embedding 封装，供 LangChain 使用
  ingest.py            # 入库处理：读取 → 切分 → 打 metadata → 分批入库
  vector.py            # 建向量库：读 notes.txt → 切分 → 存 Chroma
  search.py            # 单独测试向量检索
  test_post.py         # 测试 POST /ask 接口
  static/
    index.html         # 极简前端页面（单文件，内联CSS和JS）
  data/
    uploads/           # 网页上传的原始文件（粘贴入库的文本也存这里）
    knowledge/
      notes.txt        # 知识库原文（种子数据）
    chroma_db/         # Chroma 向量库（本地持久化）
    key_fingerprint.txt  # 只存 Key 的 SHA256 前 8 位，用来判断有没有换 Key
  notes/               # 学习笔记
  README.md            # 项目说明

> 想调参数（分块大小、top_k、缓存时间……）不用翻代码，
> 每个文件顶部的「可调参数」一段就是全部，见下面「七、可调参数在哪」。

---

## 三、依赖

- Python 3.10+
- FastAPI：提供接口
- python-multipart：接收上传文件，缺了它 `/upload` 会直接报错
- LangChain：串联检索和生成
- Chroma：本地向量库
- 智谱：Embedding（`embedding-2`）+ 生成回答（`glm-4-flash`），一把 Key 全包

---

## 四、环境准备

### 1. 安装依赖

```bash
pip install fastapi "uvicorn[standard]" python-multipart zhipuai chromadb langchain langchain-community langchain-chroma langchain-text-splitters
```

### 2. 准备智谱 API Key

**不用配 `.env`，也不用改代码。** 启动服务后打开网页，在页面最上方的
“智谱 API Key”卡片里把 Key 粘进去，点“保存”就行。

- Key 存在你自己浏览器的 `localStorage` 里，后端不落盘、不写日志。
- 页面上只显示前 4 位，不会把完整 Key 显示出来。
- 智谱 Key 在这里申请：https://open.bigmodel.cn

Embedding 和生成回答都用智谱，一把 Key 就够了。

### 3. 准备知识库文本

两种方式，任选：

- **网页上传**（推荐，日常就用这个）：启动后在页面上传 `.txt` / `.md` 文件。
- **改种子文件**：编辑 `data/knowledge/notes.txt`，一行一条，再跑一次 `python vector.py`。

不管哪种，都只用公开、合规、你有权使用的文本。

---

## 五、使用步骤

### 先启动服务

```bash
cd /d D:\...\RAG-Knowledge-Base
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

### 第一步：输入智谱 Key

浏览器打开：

```text
http://127.0.0.1:8000/
```

页面最上方是“智谱 API Key”卡片，把 Key 粘进密码框，点“保存”。

- 保存后显示“已保存 Key（前4位：xxxx...）”，**不会显示完整 Key**。
- Key 存在浏览器 `localStorage`，下次打开页面自动填回来，不用重输。
- **没填 Key 之前，上传和提问都是禁用的**，页面顶部会提示“请先输入智谱 API Key”。
- Key 填错了会提示“API Key 无效，请检查”。

### 第二步：上传文档

在上传区选一个 `.txt` 或 `.md` 文件，或者直接拖进去。上传时会显示“上传中……”，
完成后文件列表自动刷新。

关于大小：

**上传没有文件大小限制**，只限制文件类型（`.txt` / `.md`）。
一个几 MB 的纯文本可以正常传上去。

但要注意，文件越大会越慢：入库时要把文本切成块，一块块送给智谱做 Embedding
（单次请求最多 64 块，程序会自动分批），所以大文件的上传时间大致和文件大小成正比。
实测 100KB 约 5 秒、1MB 约 50 秒、5MB 大概几分钟。等的时候页面会一直显示“上传中……”，
别关页面就行。

**不想存文件？** 下面还有一张“粘贴文本入库”卡片：把文字粘进大输入框，
想的话再起个名字，点“入库”就完事。后端会把它按这个名字存成
`data/uploads/` 里的一个 `.txt`，所以之后在文件列表里看得到、也删得掉，
和上传的文件一个待遇。

### 第三步：提问

在下面输入框里提问，或者点示例问题。回答和来源显示在下方，支持回车提交，
加载中会禁用按钮。

- 默认是**流式**：回答一个字一个字往外冒，不用干等；想回到“等完整结果再显示”，
  把输入框下面的“流式输出”勾掉就行。
- 页面会**自动记住上下文**：你问过的每一轮都会带上，所以可以直接问“它和微调有什么区别”。
  点右上角的“新对话（清空上下文）”就把之前的记录清掉。
- 回答完状态栏会显示这次花了多少秒。
- 回答下面的“来源”会逐条列出命中的片段：`[序号] 文件名 · 第 N 块 · 相似度 0.82`，
  下面跟片段正文。相似度越大越相关，详见[「来源里的字段」](#来源里的字段)。

页面从上到下是：上传文档、粘贴文本入库、已上传文件、提问。

原来的 `/docs` 依然可以访问，用来调试接口。

### 第四步：删除 / 清空

- **删除单个文件** —— 文件列表里每行右边有“删除”，点一下会二次确认。删掉的同时，
  它在向量库里的 chunk 也一起删，别的文件不受影响。
- **清空知识库** —— 右下角（文件列表右上角）的“清空知识库”，二次确认后删掉所有文件
  并清空整个向量库。

### （可选）用 notes.txt 做首次初始化

`data/knowledge/notes.txt` 作为种子保留着。想让它的内容一开始就在知识库里：

```bash
python vector.py
```

会提示你输入智谱 Key（命令行里输，不回显）。看到
`已把 data/knowledge/notes.txt 初始化进向量库，共 N 块` 就成功了。  
文件不存在也不会报错，只会提示跳过。

> 注意：这样入库的 chunk，`source` 是 `notes.txt`，不在 `data/uploads/` 里，
> 所以**不会出现在网页的文件列表里**，只能靠“清空知识库”连带清掉。

### （可选）测试检索

```bash
python search.py
```

同样会先让你输入 Key。检索用的 Key 必须和当初入库用的是同一把，否则结果没有意义。

---

## 六、接口说明

### 先说请求头

**凡是会调用智谱的接口，都要带 `X-Zhipu-Key` 请求头**，值是你的智谱 Key：

```text
X-Zhipu-Key: 你的智谱 Key
```

需要带头的接口：`GET /ask`、`POST /ask`、`POST /ask/stream`、`POST /upload`、
`POST /ingest_text`、`DELETE /files/{filename}`、`DELETE /files`。

不用带的：`GET /health`（只看服务状态和块数，不调智谱）、`GET /files`（只读本地目录）、`/`、`/docs`。

服务也开了 CORS（`main.py` 顶部的 `CORS_ALLOW_ORIGINS`，默认 `["*"]`），
方便你把前端页面放到别的地方调这个接口。Key 走请求头、不靠 Cookie，
所以没开 `allow_credentials`。

没带或带了空的 → 401：

```json
{"detail": "请先输入智谱 API Key"}
```

Key 不对 → 401：

```json
{"detail": "API Key 无效"}
```

额度不足 → 403：

```json
{"detail": "智谱账户额度不足，请检查余额"}
```

### GET /ask

方便命令行直接测（要带请求头）：

```bash
curl -H "X-Zhipu-Key: 你的智谱Key" "http://127.0.0.1:8000/ask?q=什么是RAG"
```

返回：

```json
{
  "question": "什么是RAG",
  "answer": "RAG是检索增强生成，先查资料再让AI回答。",
  "sources": [
    {
      "text": "RAG 是检索增强生成，先查资料再让 AI 回答。",
      "source": "notes.txt",
      "chunk_index": 3,
      "score": 0.8235
    }
  ]
}
```

`sources` 里每项的含义见下面的[「来源里的字段」](#来源里的字段)。

### POST /ask

前端页面实际调用的就是这个接口，参数放请求体里：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Zhipu-Key: 你的智谱Key" \
  -d "{\"question\": \"什么是RAG\"}"
```

或用 `test_post.py`（会先让你输入 Key）：

```bash
python test_post.py
```

### 多轮对话（可选字段 `history`）

`POST /ask` 和 `POST /ask/stream` 都接受一个**可选**的 `history` 字段，
用来把前几轮对话带进上下文。**不传就是单轮问答，和以前完全一样。**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Zhipu-Key: 你的智谱Key" \
  -d "{\"question\": \"它和微调有什么区别\", \"history\": [{\"role\": \"user\", \"content\": \"什么是RAG\"}, {\"role\": \"assistant\", \"content\": \"RAG是检索增强生成。\"}]}"
```

- `history` 是一个数组，每项 `{"role": "user" | "assistant", "content": "..."}`。
- 顺序按时间从早到晚，最后一条应该是上一轮的 `assistant` 回答。
- 后端只取最近 `MAX_HISTORY_TURNS` 轮（默认 5 轮），每轮最多 `MAX_HISTORY_CHARS` 字
  （默认 500），多出来的直接丢掉，不会让 Prompt 无限膨胀。
- 历史只是帮 AI 理解"你在问什么"，回答依然只依据检索到的资料。

前端页面已经在自动攒 `history` 了，每问一次就把「问题 + 回答」记下来，
点「新对话（清空上下文）」可以随时清零。

### POST /ask/stream

流式问答，**新端点，老的 `POST /ask` 一点没动**。返回 `text/event-stream`（SSE），
每条消息都是一行 `data:` 加一个 JSON：

```text
data: {"type": "sources", "sources": [{"text": "RAG 是检索增强生成……", "source": "notes.txt", "chunk_index": 3, "score": 0.8235}]}

data: {"type": "delta", "text": "RAG"}

data: {"type": "delta", "text": "是检索增强生成。"}

data: {"type": "done"}
```

- `sources` 只有一条，最先发，是检索到的资料（已经去过重、合并过相邻块），
  格式和 `POST /ask` 的 `sources` 完全一样。
- `delta` 一条接一条，`text` 拼起来就是完整回答。
- `done` 是结束标记。中途出错会给 `{"type": "error", "detail": "..."}`，之后同样跟一条 `done`。

请求体和 `POST /ask` 完全一样（`question` + 可选的 `history`），
参数校验、Key 校验、`Key 已更换` 的 409 也都一样——
**这些错误在流开始之前就以正常 HTTP 状态码返回**，不会塞进 SSE 里。

用 curl 看效果（`-N` 关掉缓冲）：

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "X-Zhipu-Key: 你的智谱Key" \
  -d "{\"question\": \"什么是RAG\"}"
```

前端页面右上角有个模式勾选框，**默认就是流式**，回答会一个字一个字往外冒；
关掉它就走原来的 `POST /ask`。

### 资料里没有的问题

```bash
curl -H "X-Zhipu-Key: 你的智谱Key" "http://127.0.0.1:8000/ask?q=今天天气怎么样"
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

### POST /upload

上传一个文件，`multipart/form-data`，字段名 `file`。只收 `.txt` 和 `.md`。

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -H "X-Zhipu-Key: 你的智谱Key" \
  -F "file=@A.txt"
```

成功返回：

```json
{
  "filename": "A.txt",
  "chunks": 12,
  "message": "上传成功"
}
```

同名文件再传一次会先删掉旧向量、再覆盖原文件重新入库，不会重复也不会残留。  
**没有文件大小限制**，只校验扩展名，格式不对返回 400：

```json
{"detail": "只支持 .txt 和 .md 文件"}
```

如果知识库里已经有向量、而这次用的 Key 和当初入库的不是同一把，会返回 409：

```json
{"detail": "Key 已更换，请先点「清空知识库」，再重新上传"}
```

### POST /ingest_text

不想存文件、只想把一段文字丢进知识库时用它。请求体两个字段：

- `text`：必填，要入库的正文。
- `title`：可选，给这段文本起个名字，留空就用 `粘贴文本-年月日-时分秒`。

```bash
curl -X POST http://127.0.0.1:8000/ingest_text \
  -H "Content-Type: application/json" \
  -H "X-Zhipu-Key: 你的智谱Key" \
  -d "{\"title\": \"随手记\", \"text\": \"RAG 是检索增强生成……\"}"
```

```json
{
  "filename": "随手记.txt",
  "chunks": 3,
  "message": "已入库"
}
```

它和上传走的是**同一套入库逻辑**，区别只有一个：文本会先按 `title` 落成
`data/uploads/` 里的一个 `.txt`，然后再入库。所以——

- 它会出现在 `GET /files` 的文件列表里，也能用 `DELETE /files/{filename}` 单独删掉。
- 名字里不能出现的字符（`\ / : * ? " < > |`）会被清掉，首尾的点号和空格也会去掉。
- 同名的话和上传一样：旧的先删掉再重新入库。
- 文本最长 `ingest.MAX_PASTE_CHARS`（默认 20 万字），超了返回 400。

空文本返回 400：

```json
{"detail": "文本内容不能为空"}
```

### GET /files

列出 `data/uploads/` 里已上传的文件：

```json
{
  "files": [
    {"filename": "A.txt", "size": 1234, "upload_time": "2026-10-02 12:00:00"}
  ]
}
```

`size` 是字节数，`upload_time` 取文件修改时间。

### DELETE /files/{filename}

删掉一个文件，同时删掉它在向量库里的所有 chunk（只删这个文件的，别的文件不受影响）：

```bash
curl -X DELETE http://127.0.0.1:8000/files/A.txt \
  -H "X-Zhipu-Key: 你的智谱Key"
```

```json
{"message": "已删除 A.txt"}
```

文件不存在返回 404。

删除不校验 Key 指纹——删东西不需要算 Embedding，换过 Key 也应该删得掉。

### DELETE /files

清空知识库：删掉 `data/uploads/` 里所有文件，清空整个向量库集合，
并把 Key 指纹一起删掉（也就是解绑）。

```bash
curl -X DELETE http://127.0.0.1:8000/files \
  -H "X-Zhipu-Key: 你的智谱Key"
```

```json
{"message": "知识库已清空"}
```

同样不校验指纹——**这正是你换了 Key 之后该走的补救路径**：先清空，再用新 Key 重新上传。

### GET /health

健康检查，**不用带 Key**，随时可以打：

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "version": "1.3.0",
  "time": "2026-10-04 21:00:00",
  "chunks": 42,
  "key_bound": true,
  "cache_size": 3
}
```

- `chunks`：向量库里现在有多少块。数数不需要算 Embedding，所以不用 Key。
- `key_bound`：有没有绑着 Key 指纹（也就是知识库里有没有东西）。
  这里只给布尔值，**不会把指纹本身暴露出去**。
- `cache_size`：当前检索缓存里有多少条。
- 服务活着就返回 200，向量库文件有问题也只是 `chunks` 记 0，不会让健康检查挂掉。

### 来源里的字段

`GET /ask`、`POST /ask`、`POST /ask/stream` 返回的 `sources` 都是一个数组，
每项长这样：

| 字段 | 含义 |
|---|---|
| `text` | 命中的片段正文（去重、合并相邻块之后的） |
| `source` | 这段出自哪个文件 |
| `chunk_index` | 是文件里的第几块（**从 0 开始数**，就是入库时打的 `chunk_index`）。如果是几块拼起来的，这里是**第一块**的序号 |
| `score` | 相似度，**0~1，越大越相关**，后端保留 4 位小数 |

页面上显示成：

```text
来源：
[1] notes.txt · 第 3 块 · 相似度 0.82
    RAG 是检索增强生成，先查资料再让 AI 回答。
```

（页面把 `score` 显示成两位小数。）

**这个 `score` 是 cosine 相似度**，不是 Chroma 原始的距离：

- 向量库是用 **cosine 距离** 建的——`rag.get_vectorstore` 里写死了
  `collection_metadata={"hnsw:space": "cosine"}`。Chroma 默认是 `l2`。
- Chroma 的 `similarity_search_with_score` 给的是**距离**，越小越相关；
  cosine 空间下 `距离 = 1 - 余弦相似度`。
- 所以 `rag.py` 里统一换算成相似度：`相似度 = 1 - 距离`，再 clamp 到 `0~1`。
  缓存里存的、合并相邻块时比的、返回给你的，**全程都是这个相似度**，
  不会再出现原始距离。

> ⚠️ **升级提醒：必须重建知识库。**
> 这个约定是后加的。**你之前建的向量库是用 Chroma 默认的 l2 空间建的**，
> 而 collection 的 metadata 只在**建集合**那一刻生效——
> 已有的集合不会被自动改成 cosine，旧向量也不会重算。
> 拿 l2 的距离去套 `1 - score`，算出来的分数是错的
> （l2 距离经常远大于 1，clamp 之后会一片 0.00）。
>
> 所以升级后请务必：**点「清空知识库」→ 用同一把 Key 重新上传所有文件**。
> 清空走的是 `reset_collection()`（先删集合再建），新集合会带上 cosine 设置。
> 不重建的话，回答本身还是对的（用的是向量排序，跟距离空间无关），
> 但**来源里显示的相似度不可信**。

### 检索缓存

`rag.py` 里带了一层很小的检索缓存，省掉重复的 Embedding 调用：

- 缓存的键是 **(Key 指纹, 问题)**，值是检索出来、去重合并之后的资料。
- 最多 `CACHE_MAX_SIZE` 条（默认 128），超了就淘汰最久没用的那条。
- 兜底过期时间 `CACHE_TTL_SECONDS`（默认 300 秒）。
- **上传、粘贴入库、删除文件、清空知识库，都会立刻把缓存整体清空。**
- 换 Key 不用单独处理：Key 指纹在缓存键里，换了 Key 天然命中不到旧缓存；
  而且真换了 Key 的话，检索本来就会先被 409 挡下来。

> **换了 Key 之后要重新上传知识库。** 向量是用 Key 算出来的，换 Key 等于换坐标系，
> 旧向量跟新问题对不上。前面说的 409 就是这个意思。
> 正确顺序是：**清空知识库 → 换新 Key → 重新上传所有文件**。
> 清空的同时缓存也会一起清掉，不会留下任何旧结果。

### Key 和向量库的关系（重要）

Embedding 是拿 Key 算出来的，**换一把 Key 就是换了一套坐标系**，旧向量和新问题对不上。

所以后端在 `data/key_fingerprint.txt` 里存了一份 Key 的 **SHA256 前 8 位**
（只有哈希，没有明文，也反推不出 Key）。上传和提问时比对：

- 没换 Key → 正常检索。
- 换了 Key → 拒绝检索，返回 409：

```json
{"detail": "Key 已更换，请重新上传知识库"}
```

处理办法：点“清空知识库”，再用新 Key 重新上传所有文件。

---

## 七、可调参数在哪

**所有能调的参数都是代码顶部的常量，没有 `.env`，也没有 `config.yaml`。**
每个文件最上面的「可调参数」一段就是全部，改完重启服务生效。

### `main.py` 顶部

| 常量 | 默认值 | 作用 |
|---|---|---|
| `APP_VERSION` | `"1.3.0"` | 版本号，会在 `/health` 和 `/docs` 里显示 |
| `CORS_ALLOW_ORIGINS` | `["*"]` | 允许跨域的来源，想收紧就改成具体的域名列表 |
| `MAX_QUESTION_CHARS` | `2000` | 单个问题最多多少字，超了返回 400 |
| `UPLOAD_CHUNK_SIZE` | `1MB` | 上传时每次读/写多少字节 |
| `LOG_LEVEL` | `logging.INFO` | 日志级别，改成 `logging.WARNING` 就安静了 |

### `rag.py` 顶部

| 常量 | 默认值 | 作用 |
|---|---|---|
| `CHROMA_DIR` | `"data/chroma_db"` | 向量库放哪 |
| `COLLECTION_NAME` | `"knowledge"` | 集合名 |
| `LLM_MODEL` | `"glm-4-flash"` | 生成回答用哪个模型 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度，调低更稳、调高更活 |
| `TOP_K` | `3` | **最终**给 LLM 几条资料 |
| `CANDIDATE_K` | `8` | 先检索几条候选，去重合并后再截到 `TOP_K` |
| `MAX_HISTORY_TURNS` | `5` | 多轮对话最多带几轮 |
| `MAX_HISTORY_CHARS` | `500` | 每轮对话最多保留多少字 |
| `CACHE_MAX_SIZE` | `128` | 检索缓存最多几条 |
| `CACHE_TTL_SECONDS` | `300` | 检索缓存的兜底过期时间（秒） |

> **来源里显示的相似度是按 cosine 距离算的**：`rag.get_vectorstore` 建集合时写死了
> `collection_metadata={"hnsw:space": "cosine"}`，检索拿到的距离再换算成
> `相似度 = 1 - 距离`（clamp 到 0~1）。这一项**故意没做成常量**——
> 它一改，整个库里所有分数的含义就变了。
> 注意这个 metadata 只在**建集合**时生效，**老库（l2 建的）必须「清空知识库」重新上传**，
> 否则分数不准。详见[「来源里的字段」](#来源里的字段)。

### `ingest.py` 顶部

| 常量 | 默认值 | 作用 |
|---|---|---|
| `UPLOAD_DIR` | `"data/uploads"` | 原文件（和粘贴的文本）存哪 |
| `ALLOWED_EXTENSIONS` | `{".txt", ".md"}` | 允许上传的扩展名 |
| `CHUNK_SIZE` | `100` | **每块多少字** |
| `CHUNK_OVERLAP` | `20` | **相邻块重叠多少字** |
| `CHUNK_SEPARATORS` | `["\n\n", "\n", "。", "，", " ", ""]` | 切分时按什么切，越靠前优先级越高 |
| `EMBED_BATCH_SIZE` | `64` | **Embedding 批大小**，智谱单次上限就是 64 |
| `MAX_PASTE_CHARS` | `200000` | 粘贴入库的文本最多多少字 |
| `PASTE_NAME_MAX` | `60` | 粘贴文本自动生成的文件名最长多少字 |

### `embeddings.py` 顶部

Embedding 模型名写在 `ZhipuEmbeddings.__init__` 的 `model` 参数默认值里，默认 `embedding-2`。

> 改了 `CHUNK_SIZE` / `CHUNK_OVERLAP` / `EMBED_BATCH_SIZE` 之后，
> 要让新参数生效得**重新入库**：点「清空知识库」，再把文件传一遍。
> 光改常量不会动到已经切好存进去的旧向量。

## 八、每个文件做什么

| 文件 | 作用 |
|---|---|
| `main.py` | FastAPI 入口，问答/流式/上传/粘贴/列表/删除/清空/健康检查，取 `X-Zhipu-Key`，托管前端页面；顶部是 CORS、日志、限长等常量 |
| `rag.py` | RAG 链，检索 + 生成；按 Key 现建向量库；读写 Key 指纹；去重合并相邻块；检索缓存；顶部是 top_k、模型、缓存等常量 |
| `embeddings.py` | 智谱 Embedding 封装，接收 `api_key` 参数 |
| `ingest.py` | 入库处理：读取、切分、打 metadata、分批入库；顶部是分块大小、重叠、批大小等常量 |
| `vector.py` | 建向量库（命令行会问你要 Key） |
| `search.py` | 单独测试向量检索（同样会问 Key） |
| `test_post.py` | 测试 POST 接口（同样会问 Key） |
| `static/index.html` | 极简前端页面，单文件，内联 CSS 和 JS；Key 输入框、粘贴入库、流式开关都在这里 |
| `data/uploads/` | 网页上传的原始文件（粘贴入库的文本也存这里） |
| `data/knowledge/notes.txt` | 知识库原文（种子数据） |
| `data/chroma_db/` | 向量库文件 |
| `data/key_fingerprint.txt` | Key 的 SHA256 前 8 位，判断有没有换 Key |
| `notes/` | 学习笔记 |

---

## 九、常见问题

**Q：`ModuleNotFoundError: No module named 'xxx'`**  
A：没装依赖，跑一次安装命令。

**Q：没填 Key 会怎样？**  
A：上传和提问按钮是禁用的，页面顶部提示“请先输入智谱 API Key”。
真绕过界面直接打接口，后端也返回 401 `{"detail": "请先输入智谱 API Key"}`。

**Q：提示“API Key 无效，请检查”**  
A：Key 不对或者过期了。到 https://open.bigmodel.cn 重新复制一把，
粘回页面顶部的输入框点“保存”。注意别把前后空格带进去。

**Q：换了 Key 之后，提问提示“Key 已更换，请重新上传知识库”**  
A：这是故意的。向量是用旧 Key 算出来的，换 Key 后两套坐标系对不上，检索结果没意义。
处理办法：点“清空知识库”，再用新 Key 重新上传所有文件。
（清空会顺便删掉 `data/key_fingerprint.txt`，也就是解绑。）

**Q：Key 存在哪？会不会被后端存下来？**  
A：只存在你自己浏览器的 `localStorage` 里（键名 `zhipu_api_key`）。
每次请求放在 `X-Zhipu-Key` 请求头里发给后端，后端拿它调智谱，用完即弃——
不写文件、不写数据库、不写日志、不放进全局变量。
唯一落盘的是 `data/key_fingerprint.txt`，里面是 Key 的 SHA256 前 8 位，**不是 Key 本身**。

**Q：页面上能看到完整的 Key 吗？**  
A：看不到。页面只在保存后显示“已保存 Key（前4位：xxxx...）”。
输入框是 `type="password"`，只是回填给你自己用的。

**Q：上传有文件大小限制吗？**  
A：没有。只限制文件类型，`.txt` 和 `.md` 之外的都会被拒（前端先拦一道，后端再校验一次）。
`main.py` 接收上传时是按 1MB 分块写盘的，不把整个文件读进内存，所以多大的文件都不会撑爆内存。

**Q：上传大文件为什么要等很久？**  
A：入库要把文本切成块，一块块送给智谱做 Embedding，这一步是主要耗时，而且大致和文件大小成正比。
智谱的 Embedding 接口单次请求最多 64 条 input，程序会自动分批提交（`ingest.py` 里的 `EMBED_BATCH_SIZE`），
所以不会因为文件大就报错，只是要等。实测 100KB 约 5 秒、1MB 约 50 秒、5MB 大概几分钟。
等的时候页面会一直显示“上传中……”，别关页面。

**Q：上传报 `input数组最大不得超过64条`**  
A：这是智谱 Embedding 接口的单次条数上限，不是文件大小限制。正常不该出现了——
入库时已经按 64 条一批分批提交。如果你自己改了 `ingest.py` 里的 `add_text`
（比如绕过 `EMBED_BATCH_SIZE` 直接一次性 `add_documents`），就会重新撞到这个错。

**Q：上传成功但提问答不出来**  
A：先看上传返回的 `chunks` 是不是 0。是 0 说明文件是空的或者太短没切出块。再确认问的内容确实在那个文件里。

**Q：上传 `.pdf` / `.docx` 被拒**  
A：现在只支持 `.txt` 和 `.md`。

**Q：删了文件，提问却还能答出来**  
A：看看回答下面的“来源”。可能是这个内容别的文件里也有，检索命中了另一个文件。
如果是用 `python vector.py` 从 `notes.txt` 入库的，它的来源是 `notes.txt`，不在
`data/uploads/` 里，网页上删不到它——用“清空知识库”，或者再上传一次同名文件覆盖。

**Q：上传的文件存哪了？能直接改吗**  
A：存在 `data/uploads/`，就是原文件。直接改它没用，向量库不会跟着变，要在网页上重新上传一次。

**Q：检索结果不相关**  
A：调 `ingest.py` 顶部的 `CHUNK_SIZE`（每块多少字）和 `CHUNK_OVERLAP`（相邻块重叠多少字），
也可以调 `rag.py` 顶部的 `TOP_K`（给 LLM 几条资料）和 `CANDIDATE_K`（先捞几条候选）。
改完点“清空知识库”重新上传（或者删掉 `data/chroma_db/` 再重来）。

**Q：改了 `notes.txt` 但回答没变**  
A：向量库没重建。删 `data/chroma_db/`，重跑 `python vector.py`。

**Q：改了 `CHUNK_SIZE` 但没效果**  
A：这些是入库时的参数，只对**新入库**的文本生效。已经切好存进向量库的旧块不会变，
得“清空知识库”再重新上传一遍。`TOP_K` / `CANDIDATE_K` 是检索时才用的，改完重启就生效。

**Q：来源里的相似度全是 0.00，或者看着明显不对**  
A：多半是向量库还是老的。相似度要求向量库用 **cosine** 空间建，
而早期版本的库用的是 Chroma 默认的 `l2`，collection 的距离空间**只在建集合时确定**，
已有集合不会被改掉。解决：点“清空知识库”，再用同一把 Key 重新上传所有文件。
（清空走 `reset_collection()`，先删集合再建，新集合就带上 cosine 了。）
详见[「来源里的字段」](#来源里的字段)。

**Q：相似度为什么是 `1 - 距离`，不是直接给相似度**  
A：Chroma 的 `similarity_search_with_score` 返回的是**距离**，越小越相关，
而且含义随距离空间变。cosine 空间下 `距离 = 1 - 余弦相似度`，
所以 `rag.py` 里统一换算成 `相似度 = 1 - 距离` 并 clamp 到 `0~1`，越大越相关，
对外的 `score` 用的都是这个值。换算函数是 `rag._to_similarity`。

**Q：改了知识库，回答怎么还是旧的**  
A：正常不会。上传、粘贴入库、删除、清空都会立刻把检索缓存清掉。
如果你开了多个进程（`--reload`、多 worker），缓存各自独立，
最多 `CACHE_TTL_SECONDS`（默认 300 秒）之后也会自己失效。
想确认缓存状态，看一眼 `GET /health` 里的 `cache_size`。

**Q：流式输出能不能关**  
A：能。提问输入框下面有个“流式输出”勾选框，默认勾着（走 `POST /ask/stream`），
勾掉就走原来的 `POST /ask`，一次返回完整回答。

**Q：流式接口报错了，为什么是 200**  
A：`POST /ask/stream` 一旦开始推流就只能是 200 了——HTTP 状态码在第一个字节发出去时就定了。
所以**参数校验、Key 校验、409 这些错误都会在推流之前以正常状态码返回**；
如果是生成到一半出错，会发一条 `{"type": "error", "detail": "..."}` 的 SSE 消息，
前端会把它显示成红字。

**Q：`/ask` 返回 500**  
A：先单独跑 `python rag.py`（会让你输入 Key），看具体报错。

**Q：日志在哪看，会不会把我的 Key 或文档打出来**  
A：日志直接打在启动服务的那个终端里，每行是
`时间 级别 [ragkb] 方法 路径 -> 状态码 (耗时 ms)`。
**只记方法、路径、状态码、耗时、文件名、块数、问题字数**，
不记完整 Key、不记文件全文、不记完整 Prompt，也不记 URL 里的查询串
（不然 `GET /ask?q=...` 的问题原文就漏进日志了）。
嫌吵就把 `main.py` 顶部的 `LOG_LEVEL` 改成 `logging.WARNING`。

**Q：端口被占用**  
A：换端口：`uvicorn main:app --reload --port 8001`。

**Q：打开 `/` 显示 404**  
A：检查 `static/index.html` 是否存在于项目根目录下的 `static/` 文件夹，并且 `main.py` 里加了 `@app.get("/")` 返回该文件。

**Q：前端页面能打开但提问没反应**  
A：打开浏览器开发者工具（F12），看 Console 和 Network 里 `/ask`（或 `/ask/stream`）请求是否报错。
页面上会直接把后端返回的原因显示成红字。

**Q：粘贴的文本去哪了**  
A：`POST /ingest_text` 会把文本按 `title` 存成 `data/uploads/<title>.txt`，然后走和上传一样的入库流程。
所以它在文件列表里能看到、也能单独删掉。`title` 留空就叫 `粘贴文本-年月日-时分秒.txt`。

---

## 十、安全说明

- **Key 只在你自己的浏览器里。** 保存在 `localStorage`（键名 `zhipu_api_key`），
  随每次请求放在 `X-Zhipu-Key` 头里发给本机的后端。
- **后端不保存 Key。** 不写文件、不写数据库、不写日志、不收进全局变量；
  取出来只在这一次请求里用，用完就没了。错误信息里也会先把 Key 抹掉再返回。
- **唯一落盘的是指纹。** `data/key_fingerprint.txt` 里是 Key 的 SHA256 前 8 位，
  只有哈希，没有明文，也没法反推出 Key。它的唯一用途是判断你有没有换过 Key。
- **换 Key 后向量库会失效。** 因为向量是用 Key 算出来的，换 Key 就等于换坐标系。
  后端会拒绝检索并提示你重新上传，避免你拿到一堆对不上的结果还以为是资料问题。
- **`.env` 里已经没有 Key 了。** 后端也不再读 `.env`。
- **来源里的相似度是相对的，不是“对错”的判据。** 它是 cosine 相似度（0~1），
  只说明这段资料和你的问题在向量空间里有多接近，不保证内容真的回答了问题。
  分数高但答非所问，通常是分块或 `TOP_K` 需要调，见常见问题。
- **只用公开、合规、你有权使用的文本。** 别把隐私数据、公司机密、别人的版权内容丢进去。
- **上传内容只存在本地**（原文件在 `data/uploads/`，向量在 `data/chroma_db/`），
  但**提问和上传时，文本片段会被发给智谱**，这是 RAG 的必然代价，传之前想清楚。
- **CORS 默认是放开的。** `main.py` 顶部的 `CORS_ALLOW_ORIGINS` 默认是 `["*"]`，
  意思是任何网页都能调你本机的这个接口，方便你换个地方放前端。
  它拿不到你浏览器里存的 Key（那在 `localStorage` 里，别的站点读不到），
  但**别人可以用自己的 Key 往你的知识库里塞东西**。本机自用没问题，
  要收紧就把这项改成本地地址。
- **不要把这个服务直接暴露到公网。** 它没有登录、没有权限、没有限流；
  别人只要知道你的地址，就能用他们自己的 Key 往你的知识库里塞东西。

## License

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。