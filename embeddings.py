import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI

load_dotenv()

class ZhipuEmbeddings(Embeddings):
    def __init__(self, model: str = "embedding-2"):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
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