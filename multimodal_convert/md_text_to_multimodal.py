import re
from typing import Literal


def convert_to_multimodal_format(
        text: str,
        task: Literal["for_embed", "for_mllm"] = "for_embed",
        replace_host: bool = True,
        replace_src_host: str = "127.0.0.1",
        replace_tgt_host: str = "host.docker.internal"):
    """
        将包含Markdown图片的文本转换为多模态模型输入格式

    特点：
    - 在每张图片前插入描述性文本，包含原始图片URL
    - 将图片URL中的host替换为可访问的地址（如docker内部网络）

    Args:
        text: 原始文本，可能包含![alt](url)格式的图片
        task: 转换后的内容用于嵌入，还是多模态交互
        replace_host: 是否替换图片链接中的 host
        replace_src_host: 图片链接中的原 host
        replace_tgt_host: 替换后的 host

    Returns:
        多模态模型输入的content列表
    """

    # 定义匹配Markdown图片的正则表达式
    img_pattern = r'!\[(.*?)\]\((.*?)\)'

    # 如果没有图片，直接返回纯文本格式
    if not re.search(img_pattern, text):
        return [{"type": "text", "text": text}]

    content = []
    last_end = 0

    # 遍历所有匹配的图片
    for match in re.finditer(img_pattern, text):
        alt_text = match.group(1)  # 图片alt文本
        img_url = match.group(2)  # 原始图片URL
        start, end = match.span()

        # 添加图片前的文本（如果有）
        if start > last_end:
            text_before = text[last_end:start]
            if text_before.strip():  # 只添加非空文本
                content.append({"type": "text", "text": text_before})

        # 处理图片 URL：替换host以便模型能够访问
        processed_url = img_url
        if replace_host and replace_src_host in img_url:
            processed_url = img_url.replace(replace_src_host, replace_tgt_host)

        # 【关键改进】添加描述性文本，告诉模型图片的URL
        # 这样模型在回答时就有可能引用这个URL
        if task == "for_mllm":
            description = f"如图所示，图片链接为：{img_url}"
            if alt_text.strip():
                description += f"，图片alt文本为：{alt_text}"
            description += "，图片具体内容如下："

            content.append({"type": "text", "text": description})

        # 添加图片（使用处理后的URL，确保模型能访问到）
        content.append({
            "type": "image_url",
            "image_url": {"url": processed_url}
        })

        last_end = end

    # 添加剩余的文本
    if last_end < len(text):
        remaining_text = text[last_end:]
        if remaining_text.strip():
            content.append({"type": "text", "text": remaining_text})

    return content


if __name__ == '__main__':
    # ========== 测试 ==========
    test_chunk = 'Vision Transformer (ViT) 模型详解\n一、引言\nVision Transformer（ViT）是Google Research 于2020 年提出的突破性视觉模型，首次将自然语言处理领域大获成功的Transformer 架构直接应用于图像分类任务。不同于传统的卷积神经网络（CNN），ViT 完全摒弃了卷积操作，将图像分割成固定大小的Patches，通过自注意力机制（Self-Attention）捕获全局依赖关系，开创了视觉理解的新范式。\n二、整体架构概览\n![Page 1, Image 1](http://127.0.0.1:32810/doc-images/ViT%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/page_1_img_1.png)\n从ViT 的整体架构图可以看出，模型主要由三个核心组件构成：图像块嵌入（Patch\nEmbedding）、Transformer 编码器（Transformer Encoder）和分类头（MLP\nHead）。\n2.1 图像分块与线性投影'

    result = convert_to_multimodal_format(test_chunk, task="for_embed")
    print(result)
