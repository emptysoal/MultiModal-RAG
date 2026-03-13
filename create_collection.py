"""
    在 Milvus 中创建 Collection
"""

from pymilvus import MilvusClient, DataType
from config import *


def create_milvus_collection(uri=MILVUS_URI, collection_name=COLLECTION_NAME, dim=EMBEDDING_DIM):
    client = MilvusClient(uri=uri)

    # Collections Schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=65535
    )
    schema.add_field(
        field_name="doc_id",
        datatype=DataType.VARCHAR,
        max_length=65535
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=65535
    )
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=dim
    )

    # 索引
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    # 检查 Collections 是否已存在，如果已存在，则删除它
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )
    print(f"Collection '{collection_name}' created successfully")

    client.close()


if __name__ == '__main__':
    create_milvus_collection()
