from openai import OpenAI

from config import *


class DeepSeekOCR2Client:
    def __init__(self, url=DEEPSEEK_OCR2_ENDPOINT, api_key=DEEPSEEK_OCR2_API_KEY):
        self.client = OpenAI(
            api_key=api_key,
            base_url=url,
        )
        self.model_name = "deepseek-ocr-2"
        self.prompt = "<image>\n<|grounding|>Convert the document to markdown."

    def convert_image_to_markdown(self, image_base64_str, stream=False):
        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64_str}"}},
                    ]
                },
            ],
            temperature=0.0,
            stream=stream
        )

        return chat_response


if __name__ == '__main__':
    import base64


    def encode_image(image_path):
        with open(image_path, "rb") as file:
            encode_string = base64.b64encode(file.read()).decode("utf-8")
        return encode_string


    image_file = "../documents/doc_pdf/table.png"
    base64_string = encode_image(image_file)

    deepseek_ocr2_client = DeepSeekOCR2Client()

    # 非流式输出
    # response = deepseek_ocr2_client.convert_image_to_markdown(base64_string)
    # print("Chat response:\n", response.choices[0].message.content)

    # 迭代处理流式响应
    response = deepseek_ocr2_client.convert_image_to_markdown(base64_string, stream=True)
    for chunk in response:
        # 每个 chunk 是一个 ChatCompletionChunk 对象
        if chunk.choices[0].delta.content is not None:
            # 打印当前得到的文本片段
            print(chunk.choices[0].delta.content, end="", flush=True)
