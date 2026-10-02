import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

# 1. 加载 + 切分
loader = TextLoader("data/knowledge/notes.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)
chunks = splitter.split_documents(documents)
print("切成了", len(chunks), "块")

# 2. 把每块文字变成向量
def get_embedding(text):
    resp = client.embeddings.create(
        model="embedding-2",
        input=text
    )
    return resp.data[0].embedding

# 3. 存进 Chroma
import chromadb

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge")

for i, chunk in enumerate(chunks):
    vec = get_embedding(chunk.page_content)
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk.page_content],
        embeddings=[vec],
        metadatas=[chunk.metadata]
    )

print("已存入向量库，共", collection.count(), "条")