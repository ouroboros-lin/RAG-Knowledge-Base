import requests

url = "http://127.0.0.1:8000/ask"

payload = {
    "question": "什么是RAG",
    "top_k": 3
}

try:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    print("问题：", data["question"])
    print("回答：", data["answer"])
    print("来源：")
    for s in data["sources"]:
        print("-", s)
except requests.exceptions.RequestException as e:
    print("请求出错：", e)