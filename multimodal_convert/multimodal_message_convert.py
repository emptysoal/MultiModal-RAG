"""
    多模态消息转换，image_url 转 base64

[
    {"type": "image_url",
     "image_url": {"url": "http://host.docker.internal:32810/doc-images/ViT模型详解/page_1_img_1.png"}},
    {"type": "text", "text": "ViT模型的网络结构图"}
]

转换为：

[
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "ViT模型的网络结构图"}
]
"""

import base64
import requests
from pathlib import Path
from urllib.parse import urlparse


def url_to_base64(image_url: str) -> str:
    """
    将HTTP图片链接转换为base64编码的Data URI

    Args:
        image_url: 图片的HTTP/HTTPS链接

    Returns:
        base64编码的Data URI字符串
    """
    try:
        # 发送HTTP请求获取图片
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功

        # 获取图片内容
        image_content = response.content

        # 获取MIME类型
        # 优先从Content-Type头获取
        content_type = response.headers.get('Content-Type', '')
        if content_type and content_type.startswith('image/'):
            mime_type = content_type
        else:
            # 从URL路径猜测MIME类型
            parsed_url = urlparse(image_url)
            path = parsed_url.path
            # 获取文件扩展名
            ext = Path(path).suffix.lower()

            # 映射扩展到MIME类型
            mime_types = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
                '.svg': 'image/svg+xml',
                '.ico': 'image/x-icon',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')

        # 编码为base64
        encoded_string = base64.b64encode(image_content).decode("utf-8")

        # 返回完整的Data URI
        return f"data:{mime_type};base64,{encoded_string}"

    except requests.RequestException as e:
        raise Exception(f"下载图片失败: {e}")
    except Exception as e:
        raise Exception(f"转换图片失败: {e}")


def convert_content(content, replace_host=False, src_host="host.docker.internal", tgt_host="127.0.0.1"):
    """
        转换消息内容：image_url -> base64
    :param content: 消息内容 [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "http://..."}}]
    :param replace_host: 是否替换 image_url 中的 host
    :param src_host: image_url 中的原本 host
    :param tgt_host: image_url 中的 host 替换为什么样
    """
    for block in content:
        if block.get("image_url"):
            image_url = block["image_url"]["url"]
            if image_url.startswith(("http://", "https://")):  # 确定为 image url
                # 替换 host
                if replace_host and src_host in image_url:
                    image_url = image_url.replace(src_host, tgt_host)

                base64_image = url_to_base64(image_url)
                block["image_url"]["url"] = base64_image


if __name__ == '__main__':
    # img_url = "http://127.0.0.1:32810/doc-images/ViT%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/page_1_img_1.png"
    # ret = url_to_base64(img_url)
    # print(ret)

    # test_content = [
    #     {"type": "image_url",
    #      "image_url": {
    #          "url": "http://127.0.0.1:32810/doc-images/ViT%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/page_1_img_1.png"}},
    #     {"type": "text", "text": "ViT模型的网络结构图"}
    # ]
    # convert_content(test_content)
    # print(test_content)

    test_content = [
        {"type": "image_url",
         "image_url": {
             "url": "http://host.docker.internal:32810/doc-images/ViT%E6%A8%A1%E5%9E%8B%E8%AF%A6%E8%A7%A3/page_1_img_1.png"}},
        {"type": "text", "text": "ViT模型的网络结构图"}
    ]
    convert_content(test_content, replace_host=True)
    print(test_content)
