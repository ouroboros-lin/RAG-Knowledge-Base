"""智谱 Embedding 封装，供 LangChain 使用。

Key 不再从 .env 读，由调用方传进来（网页上用户自己填的那把）。
这里只持有这一个请求用到的 Key，不落盘、不写日志。
"""

from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI


class ZhipuEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-2"):
        if not api_key or not api_key.strip():
            raise ValueError("缺少智谱 API Key")
        self.client = ZhipuAI(api_key=api_key.strip())
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把多段文字变成向量"""
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        """把一个问题变成向量"""
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding
