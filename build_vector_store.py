"""
    构建知识库
"""

import uuid
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pymilvus import MilvusClient

from pdf_load.load_pdf import process_pdf_with_inline_images
from pdf_ocr_convert.convert_pdf import process_pdf as process_pdf_by_ocr
from multimodal_convert.md_text_to_multimodal import convert_to_multimodal_format
from embedding_client import get_embeddings
from create_collection import create_milvus_collection
from config import *


def get_text_splitter(chunk_size: int = 512, chunk_overlap: int = 100):
    """获取文本分割器"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ]
    )
    return text_splitter


def get_md_splitter(header_chars=("#", "##")):
    header_dict = {
        "#": "Header 1",
        "##": "Header 2",
        "###": "Header 3",
        "####": "Header 4",
    }
    headers_to_split_on = [(header_char, header_dict[header_char]) for header_char in header_chars]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter


def embed_chunks(chunks: list[str]) -> dict:
    """多模态嵌入"""
    embeddings = []
    for chunk in chunks:
        # 格式化文本块为多模态模型输入格式
        chunk_multimodal_content = convert_to_multimodal_format(chunk)
        # 嵌入
        response = get_embeddings(chunk_multimodal_content)
        if response.get("error"):
            error_msg = response["error"]["message"]
            return {"error": error_msg}
        embedding = response["data"][0]["embedding"]
        embeddings.append(embedding)

    return {"data": embeddings}


def process_doc(doc_path: str, use_ocr=False) -> dict:
    """对一个 PDF 文档加载、分块、嵌入、存储"""

    doc_name = os.path.basename(doc_path)
    doc_id = str(uuid.uuid4())

    # 1. 加载文档
    if not use_ocr:
        doc_content = process_pdf_with_inline_images(doc_path)
    else:
        doc_content = process_pdf_by_ocr(doc_path)

    # 2. 分块
    if not use_ocr:
        text_splitter = get_text_splitter(512, 100)  # 创建文本分割器
        chunks = text_splitter.split_text(doc_content)
    else:
        markdown_splitter = get_md_splitter()
        chunks = markdown_splitter.split_text(doc_content)
        chunks = [chunk.page_content for chunk in chunks]
    print(f"获取到 {len(chunks)} 个分块")

    # 3. 嵌入
    ret = embed_chunks(chunks)
    if ret.get("error"):
        return ret
    embeddings = ret["data"]
    print(f"嵌入了 {len(embeddings)} 个多模态嵌入数据，每个嵌入向量维度是: {len(embeddings[0])}")

    # 4. 存储、索引
    client = MilvusClient(uri=MILVUS_URI)
    entities = []
    for i, chunk in enumerate(tqdm(chunks, desc="Saving to Milvus")):
        entity = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "text": chunk,
            "embedding": embeddings[i],
            "file_name": doc_name
        }
        entities.append(entity)
    client.insert(collection_name=COLLECTION_NAME, data=entities)
    client.close()
    print(f"Inserted {len(entities)} documents")

    return {"state": "success"}


def handle_documents(files: list[str], overwrite=True):
    # 创建 Milvus Collection
    if overwrite:
        create_milvus_collection(uri=MILVUS_URI, collection_name=COLLECTION_NAME, dim=EMBEDDING_DIM)

    for file in files:
        ret = process_doc(file)
        if ret.get("error"):
            print(f"处理文件：{file} 时，报错：{ret.get('error')}")
        print(f"文件：{file} 成功嵌入并存储。")


if __name__ == '__main__':
    # 把文档分块、嵌入、存储
    pdf_path_list = "./documents/doc_pdf/ViT模型详解.pdf"
    handle_documents([pdf_path_list])
