# SPDX-License-Identifier: MIT

import getpass

import requests

url = "http://127.0.0.1:8000/ask"

# Key 现在从网页/命令行来，不放 .env
api_key = getpass.getpass("请输入智谱 API Key: ").strip()

payload = {
    "question": "什么是RAG",
    "top_k": 3
}

try:
    r = requests.post(url, json=payload,
                      headers={"X-Zhipu-Key": api_key}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("问题：", data["question"])
    print("回答：", data["answer"])
    print("来源：")
    for s in data["sources"]:
        print("-", s)
except requests.exceptions.RequestException as e:
    print("请求出错：", e)
    if getattr(e, "response", None) is not None:
        print("服务端说：", e.response.text[:200])
