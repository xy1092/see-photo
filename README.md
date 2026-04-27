# SeePhoto

本地图片预处理代理 — 将图片 OCR 识别为文字后转发给 DeepSeek 等纯文本模型。

## 原理

```
ChatBox / 任意客户端 ──▶ localhost:8000 ──▶ DeepSeek API
                           │
                    检测到图片 → OCR 提取文字
                    替换图片为文字内容
```

## 安装

```bash
git clone https://github.com/xy1092/see-photo
cd see-photo
pip install -r requirements.txt
```

## 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入你的 DeepSeek API Key
```

## 使用

```bash
# 启动代理
python proxy.py

# 客户端设置：
# API 地址: http://<本机IP>:8000
# API Key:  填写 config.yaml 中 target.api_key
# 模型:     deepseek-chat
```

## Docker

```bash
docker run -d --name see-photo \
  -p 8000:8000 \
  -v ./config.yaml:/app/config.yaml \
  xy1092/see-photo
```

## OCR 引擎

| 引擎 | 安装 | 中文精度 | 速度 |
|------|------|---------|------|
| PaddleOCR | `pip install paddleocr` | 高 | 首次慢 |
| EasyOCR | `pip install easyocr` | 中 | 较快 |

在 `config.yaml` 中切换：
```yaml
ocr:
  engine: paddleocr  # 或 easyocr
```

## 多设备使用

代理启动后，局域网内所有设备都可以连接：

| 设备 | 客户端 | API 地址示例 |
|------|--------|------------|
| PC | ChatBox / LobeChat | `http://localhost:8000` |
| 安卓 | ChatBox | `http://192.168.1.x:8000` |
| 鸿蒙 | 任意 OpenAI 客户端 | `http://192.168.1.x:8000` |

如需外网访问，建议部署到轻量云 VPS（¥35-40/月）。
