#!/usr/bin/env python3
"""SeePhoto Lite — 单文件图片预处理代理

纯本地运行，不依赖任何外部服务。
原理: 拦截 API 请求 → 图片 OCR 提取文字 → 转发纯文本给 DeepSeek

依赖: Python 3.10+, Pillow（PIL）
OCR:  系统需安装 tesseract 命令行工具（含中文语言包）

用法:
  # 1. 设置环境变量
  export DEEPSEEK_KEY=sk-your-key
  # 2. 启动
  python proxy_lite.py
  # 3. 客户端 API 地址改为 http://localhost:8000

各平台安装 tesseract:
  Ubuntu:    sudo apt install tesseract-ocr tesseract-ocr-chi-sim
  macOS:     brew install tesseract tesseract-lang
  Termux:    pkg install tesseract
  Windows:   https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ── 配置 ─────────────────────────────────────────────────
TARGET_URL = os.environ.get("TARGET_URL", "https://api.deepseek.com").rstrip("/")
TARGET_KEY = os.environ.get("DEEPSEEK_KEY", "")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "deepseek-chat")
OCR_LANG = os.environ.get("OCR_LANG", "chi_sim+eng")  # tesseract 语言代码
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "2048"))
LISTEN_HOST = os.environ.get("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("see-photo")

# ── OCR ──────────────────────────────────────────────────
_HAS_PIL = False
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    pass


def _check_tesseract() -> bool:
    """检测 tesseract 是否可用"""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_HAS_TESSERACT = _check_tesseract()


def extract_text_from_image(image_bytes: bytes) -> str:
    """从图片二进制提取文字，返回识别的文本"""
    if not _HAS_PIL:
        return "[需要安装 Pillow: pip install Pillow]"

    # 解码并限制尺寸
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # 转为 JPEG 字节（tesseract 支持更好）
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_bytes = buf.getvalue()

    # 调用 tesseract
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", OCR_LANG, "--psm", "3"],
            input=img_bytes,
            capture_output=True,
            timeout=30,
        )
        text = result.stdout.decode("utf-8", errors="replace").strip()
        return text if text else ""
    except FileNotFoundError:
        return "[tesseract 未安装，请安装后重试]"
    except Exception as e:
        log.error(f"OCR 失败: {e}")
        return ""


# ── 图片解码 ─────────────────────────────────────────────

def decode_image_url(url: str) -> bytes | None:
    """解析 data: URL 或 HTTP URL，返回图片字节"""
    # base64 data URL
    if url.startswith("data:"):
        m = re.match(r"data:image/\w+;base64,(.+)", url, re.DOTALL)
        if m:
            try:
                return base64.b64decode(m.group(1))
            except Exception:
                return None
        return None

    # HTTP(S) URL
    if url.startswith(("http://", "https://")):
        try:
            req = Request(url, headers={"User-Agent": "SeePhoto-Lite/1.0"})
            with urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            log.warning(f"下载图片失败: {url[:80]}... — {e}")
            return None

    # 本地文件
    p = Path(url)
    if p.exists():
        return p.read_bytes()

    return None


# ── 消息处理 ─────────────────────────────────────────────

def process_messages(messages: list) -> list:
    """遍历消息列表，替换图片为 OCR 文字"""
    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append(msg)
            continue
        if not isinstance(content, list):
            out.append(msg)
            continue

        parts = []
        for item in content:
            t = item.get("type", "")
            if t == "text":
                parts.append(item["text"])
            elif t == "image_url":
                url = item.get("image_url", {}).get("url", "")
                raw = decode_image_url(url)
                if raw is None:
                    parts.append("[图片无法加载]")
                    continue
                ocr_text = extract_text_from_image(raw)
                if ocr_text:
                    parts.append(f"[图片文字内容]:\n{ocr_text}")
                else:
                    parts.append("[图片中未检测到文字]")

        out.append({**msg, "content": "\n".join(parts)})
    return out


# ── HTTP 服务器 ──────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    """OpenAI 兼容 API 代理"""

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "target": TARGET_URL,
                "ocr": "tesseract" if _HAS_TESSERACT else "unavailable",
                "pil": _HAS_PIL,
            })
        elif self.path == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{"id": TARGET_MODEL, "object": "model"}]
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "only /v1/chat/completions"})
            return

        # 读取请求体
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        # 处理图片
        if "messages" in body:
            body["messages"] = process_messages(body["messages"])

        stream = body.get("stream", False)

        if stream:
            self._proxy_stream(body)
        else:
            self._proxy_normal(body)

    def _proxy_normal(self, body: dict):
        """普通请求 → 转发 → 返回 JSON"""
        body.pop("stream", None)
        req_body = json.dumps(body).encode()

        req = Request(
            f"{TARGET_URL}/v1/chat/completions",
            data=req_body,
            headers={
                "Authorization": f"Bearer {TARGET_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=120) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self._send_header_ct(resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(data)
        except HTTPError as e:
            self._json(e.code, {"error": str(e)})
        except Exception as e:
            self._json(502, {"error": f"代理请求失败: {e}"})

    def _proxy_stream(self, body: dict):
        """流式请求 → 转发 → SSE 流式返回"""
        body["stream"] = True
        req_body = json.dumps(body).encode()

        req = Request(
            f"{TARGET_URL}/v1/chat/completions",
            data=req_body,
            headers={
                "Authorization": f"Bearer {TARGET_KEY}",
                "Content-Type": "application/json",
            },
        )
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            with urlopen(req, timeout=120) as resp:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except Exception as e:
            err = json.dumps({"error": {"message": str(e)}})
            self.wfile.write(f"data: {err}\n\n".encode())
            self.wfile.flush()

    def _send_header_ct(self, ct: str):
        self.send_header("Content-Type", ct)
        self.send_header("Access-Control-Allow-Origin", "*")

    def _json(self, code: int, data: dict):
        self.send_response(code)
        self._send_header_ct("application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def log_message(self, fmt, *args):
        log.info(f"{self.client_address[0]} — {fmt % args}")


# ── 入口 ─────────────────────────────────────────────────

def main():
    print(r"""
╔══════════════════════════════════════╗
║     SeePhoto Lite — 本地图片代理    ║
║     图片 → OCR 文字 → DeepSeek     ║
╚══════════════════════════════════════╝
    """)
    print(f"  监听:    http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  目标:    {TARGET_URL}")
    print(f"  OCR:     {'✓ tesseract' if _HAS_TESSERACT else '✗ 未安装'}")
    print(f"  PIL:     {'✓' if _HAS_PIL else '✗ 需要 pip install Pillow'}")
    print(f"  API Key: {'已设置' if TARGET_KEY else '⚠ 未设置! export DEEPSEEK_KEY=sk-xxx'}")
    print()

    if not TARGET_KEY:
        print("错误: 请设置 DEEPSEEK_KEY 环境变量\n")
        sys.exit(1)
    if not _HAS_PIL:
        print("警告: Pillow 未安装，OCR 功能不可用\n  pip install Pillow\n")
    if not _HAS_TESSERACT:
        print("警告: tesseract 未安装，OCR 功能不可用\n")
        print("  Ubuntu:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim")
        print("  Termux:   pkg install tesseract")
        print("  macOS:    brew install tesseract tesseract-lang")
        print()

    server = HTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已关闭")
        server.server_close()


if __name__ == "__main__":
    main()
