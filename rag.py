import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from embeddings import ZhipuEmbeddings

load_dotenv()

embeddings = ZhipuEmbeddings()

vectorstore = Chroma(
    persist_directory="data/chroma_db",
    collection_name="knowledge",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
请根据下面提供的资料回答问题。
要求：
1. 只使用资料里的信息，不要编造；
2. 如果资料里没有答案，就说“资料中没有提到”；
3. 回答要通俗，2-3句话。

资料：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def ask(query):
    docs = retriever.invoke(query)
    answer = rag_chain.invoke(query)
    return {
        "answer": answer,
        "sources": [doc.page_content for doc in docs]
    }


if __name__ == "__main__":
    result = ask("什么是RAG")
    print("回答：", result["answer"])
    print("来源：")
    for s in result["sources"]:
        print("-", s)