# SPDX-License-Identifier: MIT

"""首次初始化：把 data/knowledge/notes.txt 作为种子数据入库。

以前这里是"读死路径的 notes.txt 自己入库"，
现在只保留一层薄封装，真正的切分和入库逻辑都在 ingest.py 里，
和网页上传走的是同一套代码。
"""

import getpass
import os

from ingest import ingest_file
from rag import write_fingerprint

SEED_FILE = "data/knowledge/notes.txt"


if __name__ == "__main__":
    if not os.path.exists(SEED_FILE):
        print(f"没有找到种子文件 {SEED_FILE}，跳过初始化。")
    else:
        api_key = getpass.getpass("请输入智谱 API Key: ").strip()
        if not api_key:
            print("没有输入 Key，退出。")
        else:
            chunks = ingest_file(SEED_FILE, api_key)
            # 记下这批向量是哪把 Key 算出来的
            write_fingerprint(api_key)
            print(f"已把 {SEED_FILE} 初始化进向量库，共 {chunks} 块")
