"""
    把 PDF 文档内容转换为 Markdown 格式
1. 加载文档
2. 转换为图像数据
3. 调用 DeepSeek-OCR-2 模型服务转换文档
4. 对 PDF 中的图片，上传至 Minio 构建的图片服务，并把原文档中图片的位置替换为图片服务的链接
"""

import io
import re
import base64
import fitz
from pathlib import Path
from PIL import Image

from client_deepseek_ocr import DeepSeekOCR2Client
from upload_images import ImageManager

deepseek_ocr2_client = DeepSeekOCR2Client()
image_manager = ImageManager()


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return label_type, cor_list


def extract_images(image, refs, page_idx, doc_id):
    image_width, image_height = image.size
    img_idx = 1   # 图片从 1 开始计数
    img_idx_url_dict = {}  # 记录文档中图片的 id 和它对应的图片服务的 url

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            # cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")

                            buffered = io.BytesIO()
                            cropped.save(buffered, format="PNG")
                            img_bytes = buffered.getvalue()
                            # 上传到 MinIO
                            object_name = f"{doc_id}/page_{page_idx}_img_{img_idx}.png"
                            content_type = "image/png"
                            image_url = image_manager.upload_image_bytes(img_bytes, object_name, content_type)
                            img_idx_url_dict[img_idx] = image_url

                            print(f"  ✓ Uploaded image {img_idx} at: {object_name}")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
        except:
            continue

    return img_idx_url_dict


def page_to_image(page, dpi=144, image_format="png"):
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    # 渲染页面为图片
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    img_data = pixmap.tobytes(image_format)
    # base64编码，用于调用 DeepSeek-OCR-2 大模型
    img_base64 = base64.b64encode(img_data).decode("utf-8")

    # 转换为 PIL 格式，用于处理文档中的图片
    img_pil = Image.open(io.BytesIO(img_data))

    return img_base64, img_pil


def extract_content_in_reading_order(page, page_idx, doc_id):
    img_base64, img_pil = page_to_image(page)

    # 调用 DeepSeek-OCR-2 模型服务转换文档
    response = deepseek_ocr2_client.convert_image_to_markdown(img_base64)
    page_content = response.choices[0].message.content

    # 提取 page 中的图片，上传至 Minio 构建的图片服务，并获取图片的链接
    matches_ref, matches_images, matches_other = re_match(page_content)
    img_idx_url_dict = extract_images(img_pil, matches_ref, page_idx, doc_id)

    # 原文档中图片的位置替换为在图片服务中的链接
    for idx, a_match_image in enumerate(matches_images, start=1):
        page_content = page_content.replace(a_match_image, f'![]({img_idx_url_dict[idx]})\n')

    for idx, a_match_other in enumerate(matches_other):
        page_content = page_content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return page_content


def process_pdf(pdf_path: str, doc_id: str | None = None) -> str:
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

        # 提取内容
        page_content = extract_content_in_reading_order(page, page_idx, doc_id)

        pdf_content += f"\n{page_content}"

    doc.close()

    return pdf_content


if __name__ == '__main__':
    pdf_file = "../documents/doc_pdf/ViT模型详解-表格2.pdf"
    ret = process_pdf(pdf_file)
    print(ret)
