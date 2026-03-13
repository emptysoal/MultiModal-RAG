"""
    使用 PyMuPDF 加载带有图片的 PDF 文档（可用于单栏和两栏，无表格，无复杂数学公式）
对 PDF 中的图片，上传至 Minio 构建的图片服务，并把原文档中图片的位置替换为图片服务的链接
"""

import fitz  # PyMuPDF
from pathlib import Path

from upload_images import ImageManager

image_manager = ImageManager()


def get_image_position(page, xref):
    """
    获取图片在页面上的位置（x, y）
    返回值越小，位置越靠上（越靠前）
    """
    # 获取图片的矩形区域
    try:
        # 获取图片在页面上的矩形区域
        rects = page.get_image_rects(xref)
        if rects:
            # 返回顶部 Y 坐标（PDF坐标系：0在底部，所以用 page.height - y）
            return rects[0].x0, rects[0].y0  # 越小越靠上
    except Exception as e:
        print(e)

    # 如果无法获取位置，返回一个中间值
    return page.rect.width / 2, page.rect.height / 2


def get_and_upload_images(page, page_num, doc_id):
    image_list = page.get_images(full=True)
    # [(21, 0, 1005, 511, 8, 'DeviceRGB', '', 'Image21', 'FlateDecode', 0)]
    content_type_map = {
        "png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg",
        "gif": "image/gif", "bmp": "image/bmp", "tiff": "image/tiff"
    }

    image_blocks = []
    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        # print(base_image)
        # {'width': 1005, 'height': 511, 'ext': 'png', 'colorspace': 3, 'xres': 96, 'yres': 96, 'bpc': 8,
        #  'size': 101487, 'image': b'...'}

        if base_image:
            ext = base_image["ext"]
            image_bytes = base_image["image"]

            # 获取图片位置
            x_pos, y_pos = get_image_position(page, xref)

            # 上传到 MinIO
            object_name = f"{doc_id}/page_{page_num}_img_{img_index}.{ext}"
            content_type = content_type_map.get(ext.lower(), "image/png")

            try:
                image_url = image_manager.upload_image_bytes(image_bytes, object_name, content_type)

                image_blocks.append({
                    "type": "image",
                    "x": x_pos,  # 图像块左上角的横坐标
                    "y": y_pos,  # 图像块左上角的纵坐标
                    "url": image_url,
                    "description": f"Page {page_num}, Image {img_index}",
                    "object_name": object_name,
                    "size": len(image_bytes)
                })
                print(f"  ✓ Uploaded image {img_index} at y={y_pos:.1f}: {object_name}")

            except Exception as e:
                print(f"  ✗ Failed to upload image {img_index}: {e}")

    return image_blocks


def extract_content_in_reading_order(page, page_num, doc_id):
    """
    按阅读顺序提取页面内容（文本和图片混合）

    返回某一页的文档加载结果
    """
    content_blocks = []

    # 1. 获取所有文本块
    text_blocks = page.get_text("blocks")
    for text_block in text_blocks:
        content_blocks.append({
            "type": "text",
            "x": text_block[0],  # 文本块左上角的横坐标
            "y": text_block[1],  # 文本块左上角的纵坐标
            "content": text_block[4]
        })

    # 2. 获取所有图片块，上传图片到图片服务，替换图片内容为图片的 URL
    image_blocks = get_and_upload_images(page, page_num, doc_id)
    content_blocks.extend(image_blocks)

    # 3. 调整每个 block 的左上角横坐标，位于页面左侧统一调整为 10，位于页面右侧统一调整为 中间值 + 10
    for block in content_blocks:
        if block["x"] < (page.rect.width / 2):
            block["x"] = 10
        else:
            block["x"] = page.rect.width / 2 + 10

    # 4. 然后按 x * 1000 + y 根据坐标排序（实现先从上到下，再从左到右，模拟阅读顺序）
    content_blocks.sort(key=lambda b: b["x"] * 1000 + b["y"])

    # 5. 把文本块和图像块连接成一个完整的字符串
    parts = []
    for block in content_blocks:
        if block["type"] == "text":
            parts.append(block["content"])

        elif block["type"] == "image":
            # 原来的图片转为 Markdown 格式
            md_image = f"\n![{block['description']}]({block['url']})\n"
            parts.append(md_image)

    return "".join(parts)


def adjust_pdf_content(pdf_content: str) -> str:
    pdf_content = pdf_content.strip()
    temp_page_text = pdf_content.replace(" \n", "\t")  # " \n" 代表两个段落的换行，先替换为 "\t"
    temp_page_text2 = temp_page_text.replace("\n", "")  # 把所有非段落换行的 "\n" 去掉
    final_page_content = temp_page_text2.replace("\t", "\n")  # 再把之前的段落换行符替换为 "\n"
    return final_page_content


def process_pdf_with_inline_images(pdf_path: str, doc_id: str | None = None) -> str:
    """
    处理含图片的 PDF，

    Args:
        pdf_path: PDF 文件路径
        doc_id: 文档标识

    Returns:
        str: 处理后的文档内容。原图片位置使用图片服务的链接替换
    """
    if doc_id is None:
        doc_id = Path(pdf_path).stem

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    print(f"Processing PDF: {pdf_path.name}")
    print(f"Total pages: {len(doc)}")

    pdf_content = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_idx = page_num + 1

        print(f"\nProcessing Page {page_idx}...")

        # 按阅读顺序提取内容（文本+图片混合）
        page_content = extract_content_in_reading_order(page, page_idx, doc_id)

        pdf_content += f"\n{page_content}"

    doc.close()

    return adjust_pdf_content(pdf_content)


if __name__ == '__main__':
    pdf_file = "../documents/doc_pdf/ViT模型详解-两栏.pdf"
    ret = process_pdf_with_inline_images(pdf_file)
    print(ret)
