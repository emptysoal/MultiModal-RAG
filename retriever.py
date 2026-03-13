"""
    多模态检索器
"""

from pymilvus import MilvusClient

from embedding_client import get_embeddings
from config import *


class MultiModalRetriever:
    def __init__(
            self,
            milvus_uri=MILVUS_URI,
            milvus_collection_name=COLLECTION_NAME,
            top_k=DEFAULT_TOP_K
    ):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = milvus_collection_name

        self.top_k = top_k

    def semantic_retrieve(self, content: list, top_k: int = 0) -> dict:
        """
            语义搜索（向量检索）
        密集搜索使用向量 Embeddings 来查找具有相似含义的文档，即使它们没有完全相同的关键词。这种方法有助于理解上下文和语义，非常适合自然语言查询。
        :param content: 用于检索的多模态的输入
        :param top_k: 取最相近的 k 条结果
        :return: [{"id": "", "distance": float, "entity": {"title": "", "content": ""}}, ...]
        """
        # Generate embedding for query
        response = get_embeddings(content)
        # print(ret)
        if response.get("error"):
            error_msg = response["error"]["message"]
            return {"error": error_msg}
        query_embedding = response["data"][0]["embedding"]

        # Semantic search using dense vectors
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k if top_k > 0 else self.top_k,
            output_fields=["text", "$meta"]
        )

        if not results:
            return {"data": []}

        # 格式化检索结果
        final_result = []
        for result in results[0]:
            final_result.append(
                {
                    "id_": result["id"],
                    "score": result["distance"],
                    "text": result["entity"]["text"],
                    "file_name": result["entity"]["file_name"],
                }
            )

        return {"data": final_result}


if __name__ == '__main__':
    retriever = MultiModalRetriever()

    # Example query for search
    test_content = [
        {"type": "image_url",
         "image_url": {"url": "http://host.docker.internal:32810/doc-images/ViT模型详解/page_1_img_1.png"}},
        {"type": "text", "text": "ViT模型的网络结构图"}
    ]

    res = retriever.semantic_retrieve(test_content)
    print(res)
