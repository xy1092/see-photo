"""SeePhoto — 本地图片预处理代理

将图片 OCR 识别为文字后，转发给 DeepSeek 等纯文本模型。
用法: python proxy.py
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from image_processor import process_messages

# ── 加载配置 ─────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

TARGET_URL = config["target"]["url"].rstrip("/")
TARGET_KEY = config["target"]["api_key"]
TARGET_MODEL = config["target"].get("model", "deepseek-chat")
OCR_ENGINE = config["ocr"].get("engine", "paddleocr")
OCR_MAX_SIZE = config["ocr"].get("max_size", 2048)
SERVER_HOST = config["server"].get("host", "0.0.0.0")
SERVER_PORT = config["server"].get("port", 8000)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("see-photo")

# ── HTTP 客户端 ──────────────────────────────────────────
client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(120), follow_redirects=True)
    logger.info(f"SeePhoto 启动 → {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"目标 API: {TARGET_URL}")
    logger.info(f"OCR 引擎: {OCR_ENGINE}")
    yield
    if client:
        await client.aclose()
    logger.info("SeePhoto 已关闭")


app = FastAPI(title="SeePhoto", lifespan=lifespan)


# ── 路由 ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "target": TARGET_URL, "ocr": OCR_ENGINE}


@app.get("/v1/models")
async def list_models():
    """返回模型列表，让 ChatBox 等客户端能正常识别"""
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": TARGET_MODEL, "object": "model", "owned_by": "see-photo"}
        ]
    })


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = body.get("stream", False)

    # —— 处理图片 ——
    body = _process_body(body)

    if stream:
        return StreamingResponse(
            _stream_response(body),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )
    else:
        return await _normal_response(body)


# ── 核心逻辑 ──────────────────────────────────────────────

def _process_body(body: dict) -> dict:
    """处理请求体，替换图片为 OCR 文字"""
    messages = body.get("messages", [])
    if not messages:
        return body

    try:
        processed = process_messages(messages, max_size=OCR_MAX_SIZE, engine=OCR_ENGINE)
        body["messages"] = processed
    except Exception as e:
        logger.error(f"图片处理失败: {e}")

    # 同时注入模型名
    if "model" not in body:
        body["model"] = TARGET_MODEL
    return body


async def _normal_response(body: dict):
    """非流式响应"""
    try:
        resp = await client.post(
            f"{TARGET_URL}/v1/chat/completions",
            json=body,
            headers=_auth_header(),
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        logger.error(f"请求失败: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=502)


async def _stream_response(body: dict):
    """流式 SSE 响应"""
    body["stream"] = True
    try:
        async with client.stream(
            "POST",
            f"{TARGET_URL}/v1/chat/completions",
            json=body,
            headers=_auth_header(),
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
    except Exception as e:
        logger.error(f"流式请求失败: {e}")
        error_chunk = {
            "error": {"message": str(e), "type": "proxy_error"}
        }
        yield f"data: {json.dumps(error_chunk)}\n\n".encode()


def _auth_header() -> dict:
    return {
        "Authorization": f"Bearer {TARGET_KEY}",
        "Content-Type": "application/json",
    }


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("""
╔══════════════════════════════════╗
║       SeePhoto 图片代理         ║
║  图片 → OCR 文字 → DeepSeek    ║
╚══════════════════════════════════╝
    """)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
