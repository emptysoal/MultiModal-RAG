"""
    requests 客户端，访问使用 vLLM 部署的 Qwen3-VL-Embedding-2B 模型
服务端部署命令：
    bash start_vllm_qwen3_vl_embedding.sh
"""

import requests
from config import *


def get_embeddings(content, instruction: str = "Represent the user's input."):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content}
    ]

    # 调用 Embedding API
    response = requests.post(
        EMBEDDING_ENDPOINT,
        json={
            "model": EMBEDDING_MODEL,
            "messages": conversation,
            "encoding_format": "float",
        }
    )

    response_json = response.json()

    return response_json


if __name__ == '__main__':
    # image url (network path)
    test_content = [
        {"type": "image_url",
         "image_url": {"url": "http://host.docker.internal:32810/doc-images/ViT模型详解/page_1_img_1.png"}},
        {"type": "text", "text": "ViT模型的网络结构图"}
    ]

    ret = get_embeddings(test_content)
    # print(ret)
    if ret.get("error"):
        print(ret["error"]["message"])
    else:
        print("Embedding output:", ret["data"][0]["embedding"])
