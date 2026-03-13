"""
    配置文件
"""

import os

# 图片服务配置
MINIO_ENDPOINT = "127.0.0.1:32810"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
MINIO_SECURE = False
MINIO_BUCKET = "doc-images"
# 在做多模态嵌入时，是否替换图片链接的 host，比如多模态嵌入为 docker 部署，那么它无法根据 "127.0.0.1" 访问图片，这时需替换 host
REPLACE_HOST = True
REPLACE_SOURCE_HOST = "127.0.0.1"
REPLACE_TARGET_HOST = "host.docker.internal"
# REPLACE_TARGET_HOST = "172.17.0.1"  # 使用 WSL2 主机 IP，Docker 默认网关

# Milvus 配置
MILVUS_URI = "http://127.0.0.1:19530"
COLLECTION_NAME = "MultiModal"
# 搜索参数
DEFAULT_TOP_K = 5

# 大模型相关服务
# DeepSeek-OCR-2 服务配置
DEEPSEEK_OCR2_ENDPOINT = "https://127.0.0.1:32809/v1"
DEEPSEEK_OCR2_API_KEY = os.getenv("DEEPSEEK_OCR2_API_KEY", "EMPTY")

# 多模态嵌入模型
EMBEDDING_ENDPOINT = "http://127.0.0.1:32808/v1/embeddings"
EMBEDDING_MODEL = "/models/Qwen3-VL-Embedding-2B"
EMBEDDING_DIM = 2048

# 多模态交互模型
MLLM_ENDPOINT = "http://127.0.0.1:32807/v1"
MLLM_MODEL_NAME = "/models/Qwen3-VL-8B-Instruct"
MLLM_API_KEY = os.getenv("MLLM_API_KEY", "EMPTY")
MLLM_TEMPERATURE = 0.3
MLLM_MESSAGE_CONVERT = True  # 访问外部多模态模型时，需要转换消息格式，因为图片服务是本地部署的，外部无法访问

# 大模型对话提示词参数
SYSTEM_PROMPT = """
# 角色：
你是一个专业的问题解答助手，并且具备处理多模态消息的能力。
基于用户提供的'参考资料'（含文本和图片）回答问题，优先引用图片中的信息。

## 限制：
1. 如果'参考资料'中包含图片，你可以考虑是否在回答中引用相关图片以增强回复效果，如果引用图片必须是Markdown格式的，格式为：![描述](图片URL)
2. 如果引用图片，一定确保引用的图片URL与'参考资料'中提供的URL一致，非常重要！一定要确保！！
3. 如果用户的提问或查询与'参考资料'相关，则结合'参考资料'，根据用户输入给出回答；
4. 如果用户的提问或查询与'参考资料'不相关，则坚决不使用'参考资料'，仅根据用户输入给出回答；
5. 回答应当准确、完整，可以自己判断是否指出信息来源
"""
