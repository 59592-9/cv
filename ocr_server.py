# -*- coding: utf-8 -*-
"""
TrOCR 常驻 HTTP 服务

启动后模型常驻 GPU 显存，通过 HTTP API 接收图片并返回识别结果。
支持 base64 JSON 和 multipart/form-data 两种上传方式。

用法:
    python ocr_server.py                    # 默认 GPU 模式，端口 5000
    python ocr_server.py --port 8080        # 指定端口
    python ocr_server.py --cpu              # CPU 模式

优化点：
- 使用 logging 模块统一日志输出
- 增加请求体大小限制，防止超大图片打爆内存
- 增加 /metrics 端点，暴露识别次数与平均耗时
- 输入验证更严格（base64 格式校验、图片解码校验）
- 线程安全的推理锁，防止多请求并发推理导致显存溢出
- 健康检查返回更多运行时信息
"""

import argparse
import base64
import logging
import re
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request

# ------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ocr_server")

# ------------------------------------------------------------------
# Flask 应用
# ------------------------------------------------------------------
app = Flask(__name__)
# 限制请求体最大 16 MB，防止超大图片耗尽内存
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ------------------------------------------------------------------
# 全局状态
# ------------------------------------------------------------------
_processor = None
_model = None
_device = None
_Image = None
_infer_lock = threading.Lock()   # 推理互斥锁，防止并发推理显存溢出
_start_time: float = 0.0

# 滑动窗口统计（最近 1000 次请求的耗时，单位 ms）
_latency_window: deque = deque(maxlen=1000)
_total_requests: int = 0
_failed_requests: int = 0
_stats_lock = threading.Lock()


# ------------------------------------------------------------------
# 模型初始化
# ------------------------------------------------------------------

def init_model(use_gpu: bool = True) -> None:
    """
    初始化 TrOCR 模型并加载到指定设备。

    Args:
        use_gpu: 是否优先使用 GPU
    """
    global _processor, _model, _device, _Image, _start_time

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image as PILImage

    _Image = PILImage

    if use_gpu and torch.cuda.is_available():
        _device = torch.device("cuda")
        logger.info("使用 GPU: %s", torch.cuda.get_device_name(0))
    else:
        _device = torch.device("cpu")
        if use_gpu:
            logger.warning("GPU 不可用，回退到 CPU 模式")
        else:
            logger.info("使用 CPU 模式")

    model_name = "microsoft/trocr-base-printed"
    logger.info("加载模型: %s", model_name)

    _processor = TrOCRProcessor.from_pretrained(model_name)
    _model = VisionEncoderDecoderModel.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()

    if _device.type == "cuda":
        import torch as _torch
        mem = _torch.cuda.memory_allocated() / 1024 / 1024
        logger.info("GPU 显存占用: %.0f MB", mem)

    _start_time = time.time()
    logger.info("模型加载完成！")


# ------------------------------------------------------------------
# 推理核心
# ------------------------------------------------------------------

def _recognize_text(img_bgr: np.ndarray) -> str:
    """
    使用 TrOCR 识别图片中的文字（线程安全）。

    Args:
        img_bgr: BGR 格式的 numpy 数组

    Returns:
        识别出的原始文本字符串
    """
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = _Image.fromarray(img_rgb)

    pixel_values = _processor(
        images=pil_image,
        return_tensors="pt",
    ).pixel_values.to(_device)

    with _infer_lock:
        with torch.no_grad():
            generated_ids = _model.generate(pixel_values)

    text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def _decode_image_from_b64(b64_str: str) -> Optional[np.ndarray]:
    """
    将 base64 字符串解码为 BGR numpy 数组。

    Args:
        b64_str: base64 编码的图片字符串

    Returns:
        BGR 图像数组；解码失败返回 None
    """
    try:
        img_bytes = base64.b64decode(b64_str, validate=True)
    except Exception:
        logger.warning("base64 解码失败")
        return None

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img  # 可能为 None（图片格式不支持）


def _decode_image_from_bytes(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    将原始字节解码为 BGR numpy 数组。

    Args:
        file_bytes: 图片原始字节

    Returns:
        BGR 图像数组；解码失败返回 None
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _update_stats(elapsed_ms: float, success: bool) -> None:
    """线程安全地更新请求统计信息。"""
    global _total_requests, _failed_requests
    with _stats_lock:
        _total_requests += 1
        if not success:
            _failed_requests += 1
        _latency_window.append(elapsed_ms)


# ------------------------------------------------------------------
# HTTP 路由
# ------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    健康检查接口。

    Returns:
        JSON 包含服务状态、设备信息、运行时长、请求统计。
    """
    uptime = time.time() - _start_time if _start_time else 0
    with _stats_lock:
        total = _total_requests
        failed = _failed_requests
        avg_ms = (
            round(sum(_latency_window) / len(_latency_window), 1)
            if _latency_window
            else 0
        )

    return jsonify({
        "status": "ok",
        "gpu": str(_device) if _device else "unknown",
        "model_loaded": _model is not None,
        "uptime_seconds": round(uptime, 1),
        "total_requests": total,
        "failed_requests": failed,
        "avg_latency_ms": avg_ms,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    指标接口，返回详细的性能统计数据。

    Returns:
        JSON 包含请求总数、失败数、最近延迟分布。
    """
    with _stats_lock:
        window = list(_latency_window)

    if window:
        sorted_w = sorted(window)
        n = len(sorted_w)
        p50 = sorted_w[int(n * 0.50)]
        p90 = sorted_w[int(n * 0.90)]
        p99 = sorted_w[min(int(n * 0.99), n - 1)]
        avg = sum(sorted_w) / n
    else:
        p50 = p90 = p99 = avg = 0

    with _stats_lock:
        total = _total_requests
        failed = _failed_requests

    return jsonify({
        "total_requests": total,
        "failed_requests": failed,
        "success_rate": round((total - failed) / total * 100, 2) if total else 100,
        "latency_ms": {
            "avg": round(avg, 1),
            "p50": round(p50, 1),
            "p90": round(p90, 1),
            "p99": round(p99, 1),
        },
        "window_size": len(window),
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    图片数字识别接口。

    支持两种上传方式：
    - JSON body: {"image_base64": "<base64字符串>"}
    - multipart/form-data: 字段名 "image"

    Returns:
        JSON:
            success   (bool)  - 是否识别成功
            number    (int)   - 提取的数字，失败为 -1
            raw_text  (str)   - OCR 原始文本
            time_ms   (float) - 本次请求耗时（毫秒）
            error     (str)   - 错误信息（仅失败时存在）
    """
    if _model is None:
        return jsonify({
            "success": False,
            "number": -1,
            "error": "模型尚未加载",
        }), 503

    t_start = time.perf_counter()
    img: Optional[np.ndarray] = None

    try:
        # 方式1：JSON base64
        if request.is_json:
            data = request.get_json(silent=True) or {}
            b64_str = data.get("image_base64", "")
            if not b64_str:
                return jsonify({
                    "success": False,
                    "number": -1,
                    "error": "缺少 image_base64 字段",
                }), 400
            img = _decode_image_from_b64(b64_str)

        # 方式2：文件上传
        elif "image" in request.files:
            file_bytes = request.files["image"].read()
            if not file_bytes:
                return jsonify({
                    "success": False,
                    "number": -1,
                    "error": "上传文件为空",
                }), 400
            img = _decode_image_from_bytes(file_bytes)

        else:
            return jsonify({
                "success": False,
                "number": -1,
                "error": "请求中未找到图片数据（需要 JSON image_base64 或 multipart image）",
            }), 400

        if img is None:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            _update_stats(elapsed_ms, success=False)
            return jsonify({
                "success": False,
                "number": -1,
                "error": "图片解码失败，请检查格式是否正确",
            }), 422

        # 执行识别
        raw_text = _recognize_text(img)

        # 提取数字
        digits = re.findall(r"\d+", raw_text)
        number = int("".join(digits)) if digits else -1

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        _update_stats(elapsed_ms, success=True)

        logger.debug("识别结果: '%s' -> %d (%.1f ms)", raw_text, number, elapsed_ms)

        return jsonify({
            "success": True,
            "number": number,
            "raw_text": raw_text,
            "time_ms": round(elapsed_ms, 1),
        })

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        _update_stats(elapsed_ms, success=False)
        logger.exception("识别请求异常")
        return jsonify({
            "success": False,
            "number": -1,
            "error": str(exc),
        }), 500


@app.errorhandler(413)
def request_entity_too_large(_):
    """处理超大请求体（> 16 MB）。"""
    return jsonify({
        "success": False,
        "number": -1,
        "error": "图片文件过大，最大支持 16 MB",
    }), 413


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrOCR 常驻 HTTP 服务")
    parser.add_argument(
        "--port", type=int, default=5000, help="服务端口（默认 5000）"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="强制使用 CPU"
    )
    parser.add_argument(
        "--debug", action="store_true", help="开启 Flask 调试模式（仅开发用）"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  TrOCR 常驻 HTTP 服务")
    print("=" * 50)

    init_model(use_gpu=not args.cpu)

    print(f"\n[OCR Server] 服务启动在 http://localhost:{args.port}")
    print(f"[OCR Server] 健康检查: GET  http://localhost:{args.port}/health")
    print(f"[OCR Server] 识别接口: POST http://localhost:{args.port}/recognize")
    print(f"[OCR Server] 性能指标: GET  http://localhost:{args.port}/metrics")
    print("=" * 50)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
