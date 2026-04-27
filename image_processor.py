"""图片预处理：提取图片中的文字信息"""

import base64
import io
import logging
import re
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

logger = logging.getLogger("see-photo")

# 懒加载 OCR 引擎
_ocr = None
_ocr_engine = None


def _get_paddleocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="ch", use_angle_cls=False, show_log=False)


def _get_easyocr():
    import easyocr
    return easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)


def _load_ocr(engine: str):
    global _ocr, _ocr_engine
    if _ocr is not None and _ocr_engine == engine:
        return _ocr
    _ocr_engine = engine
    if engine == "easyocr":
        _ocr = _get_easyocr()
    else:
        _ocr = _get_paddleocr()
    return _ocr


def _ocr_image(image: Image.Image, engine: str) -> str:
    """对单张图片执行 OCR，返回识别文字"""
    ocr = _load_ocr(engine)

    if engine == "easyocr":
        img_array = _image_to_array(image)
        results = ocr.readtext(img_array)
        texts = [item[1] for item in results if item[2] > 0.5]
        return "\n".join(texts)

    # PaddleOCR
    img_array = _image_to_array(image)
    result = ocr.ocr(img_array)
    if not result or not result[0]:
        return ""
    texts = [line[1][0] for line in result[0]]
    return "\n".join(texts)


def _image_to_array(image: Image.Image):
    """PIL Image → numpy array (RGB)"""
    import numpy as np
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def _load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    # 确保 RGB 并限制尺寸
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img


def _limit_size(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = max_size / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def extract_text(image_data: bytes, max_size: int = 2048, engine: str = "paddleocr") -> str:
    """从图片二进制数据中提取文字"""
    img = _load_image_from_bytes(image_data)
    img = _limit_size(img, max_size)
    return _ocr_image(img, engine)


def decode_image_url(url: str) -> bytes | None:
    """解析图片 URL，返回原始字节。
    支持: http/https URL, data:image/...;base64,... 格式
    """
    if url.startswith("data:"):
        match = re.match(r"data:image/\w+;base64,(.+)", url, re.DOTALL)
        if not match:
            return None
        try:
            return base64.b64decode(match.group(1))
        except Exception:
            return None

    if url.startswith(("http://", "https://")):
        try:
            req = Request(url, headers={"User-Agent": "SeePhoto/1.0"})
            with urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            logger.warning(f"下载图片失败: {url}, {e}")
            return None

    # 本地文件路径
    path = Path(url)
    if path.exists():
        return path.read_bytes()

    return None


def process_messages(messages: list, max_size: int = 2048, engine: str = "paddleocr") -> list:
    """处理消息列表，将图片内容替换为 OCR 识别文字。

    输入格式兼容 OpenAI Vision API:
      content: [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "..."}}
      ]
    也支持纯文本:
      content: "hello"
    """
    processed = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # 纯文本，直接保留
            processed.append(msg)
            continue

        if not isinstance(content, list):
            processed.append(msg)
            continue

        # 解析 content 列表
        text_parts = []
        image_count = 0
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part["text"])
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if not url:
                    continue
                raw = decode_image_url(url)
                if raw is None:
                    text_parts.append("[图片: 无法加载]")
                    continue
                try:
                    ocr_text = extract_text(raw, max_size=max_size, engine=engine)
                    if ocr_text.strip():
                        text_parts.append(f"[图片文字内容]:\n{ocr_text}")
                    else:
                        text_parts.append("[图片: 未检测到文字]")
                except Exception as e:
                    logger.error(f"OCR 失败: {e}")
                    text_parts.append("[图片: 识别失败]")
                image_count += 1

        new_content = "\n".join(text_parts)
        processed.append({**msg, "content": new_content})

    return processed
