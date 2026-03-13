"""
    上传图像到 Minio 构建的图片服务
"""

import io
from minio import Minio
import urllib.parse

from config import *


class ImageManager:
    def __init__(
            self,
            minio_endpoint=MINIO_ENDPOINT,
            minio_access_key=MINIO_ACCESS_KEY,
            minio_secret_key=MINIO_SECRET_KEY,
            minio_secure=MINIO_SECURE,
            minio_bucket=MINIO_BUCKET
    ):
        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure
        )

        self.minio_endpoint = minio_endpoint
        self.bucket_name = minio_bucket

    def ensure_bucket_exists(self):
        """确保 bucket 存在并设置为公开读取"""
        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                    }
                ]
            }
            import json
            self.minio_client.set_bucket_policy(self.bucket_name, json.dumps(policy))

    def upload_image_bytes(self, image_bytes, object_name, content_type="image/png"):
        """上传图片字节到 MinIO，返回可访问的 URL"""
        self.ensure_bucket_exists()

        self.minio_client.put_object(
            self.bucket_name,
            object_name,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=content_type
        )

        encoded_name = urllib.parse.quote(object_name)
        return f"http://{self.minio_endpoint}/{self.bucket_name}/{encoded_name}"


if __name__ == '__main__':
    image_manager = ImageManager()

    img_local_path = "./documents/doc_md/含图片/images/ViT.png"
    with open(img_local_path, "rb") as img_file:
        img_bytes = img_file.read()

    image_url = image_manager.upload_image_bytes(img_bytes, "测试图片/ViT.png")
    print(image_url)
