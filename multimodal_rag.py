"""
    多模态 RAG
"""

from pathlib import Path

from langgraph.graph import MessagesState, START, END, StateGraph
from langchain.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

from retriever import MultiModalRetriever
from multimodal_convert.md_text_to_multimodal import convert_to_multimodal_format
from multimodal_convert.multimodal_message_convert import convert_content
from upload_images import ImageManager
from config import *

retriever = MultiModalRetriever()

mllm = ChatOpenAI(
    model=MLLM_MODEL_NAME,
    base_url=MLLM_ENDPOINT,
    api_key=MLLM_API_KEY,
    temperature=MLLM_TEMPERATURE
)


class State(MessagesState):
    files: list[str]
    documents: list[str]


# Nodes
def retrieve_docs(state: State):
    """根据用户的查询，从向量数据库检索相关的文档"""

    # 获取最近的用户的消息
    last_message = state["messages"][-1]
    assert isinstance(last_message, HumanMessage), "In retriever node, last message is not HumanMessage."

    # 检索
    user_content = last_message.content
    retrieve_res = retriever.semantic_retrieve(user_content)
    # print(retrieve_res)
    if retrieve_res.get("error"):
        print("检索失败，模型将根据自身知识储备做出回复。")
        return {"files": [], "documents": []}

    retrieved_data = retrieve_res.get("data")
    if not retrieved_data:
        print("未检索到任何内容，模型将根据自身知识储备做出回复。")
        return {"files": [], "documents": []}

    files = [doc_chunk_dict["file_name"] for doc_chunk_dict in retrieved_data]
    docs = [doc_chunk_dict["text"] for doc_chunk_dict in retrieved_data]

    return {"files": files, "documents": docs}


# 测试检索
# input_message = HumanMessage(content=[
#     {"type": "text", "text": "帮我讲解下ViT模型的网络结构"},
#     {"type": "image_url",
#      "image_url": {"url": "http://host.docker.internal:32810/doc-images/ViT模型详解/page_1_img_1.png"}}
# ])
# ret = retrieve_docs({"messages": [input_message]})
# print(ret)


def mllm_call(state: State):
    # 把检索到的内容拼接成字符串
    context_list = []
    for file_name, doc_content in zip(state["files"], state["documents"]):
        one_context = f"文档名：\n{file_name}\n文档内容：\n{doc_content}"
        context_list.append(one_context)
    context = "\n\n".join(context_list)

    # 检索内容转换为多模态模型 user content 的格式
    # 当访问外部多模态模型服务时，图片链接的 host 不能替换为 host.docker.internal
    replace_host = False if MLLM_MESSAGE_CONVERT else True
    retrieved_content = convert_to_multimodal_format(context, task="for_mllm", replace_host=replace_host)
    retrieved_content = [{"type": "text", "text": "请参考以下检索到的'参考资料'作答："}] + retrieved_content
    # print(retrieved_content)

    # 获取最近的用户的消息
    last_message = state["messages"][-1]
    query_content = last_message.content

    user_content = retrieved_content + [{"type": "text", "text": "用户原始的提问如下："}] + query_content
    if MLLM_MESSAGE_CONVERT:  # 当访问外部多模态模型服务时，把消息中的 image_url 转换为 base64
        convert_content(user_content, replace_host=True)
    # print(user_content)
    user_message = HumanMessage(content=user_content)

    # 构建系统消息
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    # 构建用于对话生成的消息
    history_messages = state["messages"][:-1]
    messages_for_mllm = [system_message] + history_messages + [user_message]

    # 多模态大语言模型推理
    response = mllm.invoke(messages_for_mllm)

    return {"messages": [response]}


# 测试多模态交互模型
# input_message = HumanMessage(content=[
#     {"type": "text", "text": "帮我讲解下ViT模型的网络结构"},
# ])
# input_ = {"messages": [input_message]}
# ret = retrieve_docs(input_)
# input_["files"], input_["documents"] = ret["files"], ret["documents"]
# ret2 = mllm_call(input_)
# print(ret2)


def delete_messages(state: State):
    messages = state['messages']
    if len(messages) > 10:
        # remove the earliest messages
        recent_messages = messages[-10:]
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages]}
    return {"messages": []}


def build_workflow():
    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("mllm_call", mllm_call)
    workflow.add_node("trim_messages", delete_messages)

    # Add edges to connect nodes
    workflow.add_edge(START, "retrieve_docs")
    workflow.add_edge("retrieve_docs", "mllm_call")
    workflow.add_edge("mllm_call", "trim_messages")
    workflow.add_edge("trim_messages", END)

    checkpointer = InMemorySaver()
    # Compile the agent
    rag = workflow.compile(checkpointer=checkpointer)

    # Show the workflow
    # graph = rag.get_graph(xray=True)
    # mermaid_code = graph.draw_mermaid()
    # print(mermaid_code)

    return rag


def get_image_mime_type(img_path):
    # 映射扩展到MIME类型
    mime_types = {
        "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
        "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff"
    }
    ext = Path(img_path).suffix.lower()
    mime_type = mime_types.get(ext, 'image/jpeg')
    return mime_type


def convert_input_to_content(input_list: list[dict], chat_id: str):
    """
        转换输入
    :param input_list: [{"type": "text", "text": "..."}, {"type": "image", "local_path": "..."}, ...]
    :param chat_id: 本次对话的标识 id
    :return: 多模态模型的 content 格式
    """
    image_manager = ImageManager()

    content = []
    for block in input_list:
        if block["type"] == "image":
            # 图片上传到 MinIO 图片服务
            image_local_path = block["local_path"]
            with open(image_local_path, "rb") as img_file:
                img_bytes = img_file.read()

            object_name = f"{chat_id}/{os.path.basename(image_local_path)}"
            content_type = get_image_mime_type(image_local_path)
            try:
                image_url = image_manager.upload_image_bytes(img_bytes, object_name, content_type)
                image_url = image_url.replace("127.0.0.1", "host.docker.internal")
                # print(image_url)
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            except Exception as e:
                print(e)
        else:
            content.append(block)

    return content


if __name__ == '__main__':
    rag_workflow = build_workflow()

    config = {"configurable": {"thread_id": "1"}}
    test_chat_id = "0442fa9d-d01d-4ef6-bd46-b6fe02234c4a"

    # 构建多模态输入
    test_image_path = "./documents/doc_md/含图片/images/ViT.png"
    # test_input = [{"type": "text", "text": "帮我解析下这张图"}, {"type": "image", "local_path": test_image_path}]
    test_input = [{"type": "text", "text": "帮我详解下ViT模型的网络结构"}]
    test_content = convert_input_to_content(test_input, test_chat_id)
    test_messages = {"messages": [HumanMessage(content=test_content)]}

    # 调用多模态 RAG 工作流
    for stream_mode, data in rag_workflow.stream(test_messages, config, stream_mode=["updates", "messages"]):
        if stream_mode == "updates":
            for source, update in data.items():
                print("=== Node Name:", source)
                print("=== Updated State:", update)
        if stream_mode == "messages":
            msg, metadata = data
            if msg.content and metadata["langgraph_node"] == "mllm_call":
                print(msg.content, end="", flush=True)
