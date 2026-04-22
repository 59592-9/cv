# -*- coding: utf-8 -*-
"""
PaddleOCR 常驻 HTTP 服务

启动后模型常驻 GPU 显存，通过 HTTP API 接收图片并返回识别结果。
支持 base64 JSON 和 multipart/form-data 两种上传方式。

用法:
    python ocr_server.py                    # 默�� GPU 模式，端口 5000
    python ocr_server.py --port 8080        # 指定端口
    python ocr_server.py --cpu              # CPU 模式

优化点：
- 使用 PaddleOCR（PP-OCRv4 英文数字模型），对小区域纯数字识别更准确
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
_ocr = None                          # PaddleOCR 实例
_use_gpu: bool = True
_infer_lock = threading.Lock()       # 推理互斥锁，防止并发推理显存溢出
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
    初始化 PaddleOCR 模型并加载到指定设备。

    Args:
        use_gpu: 是否优先使用 GPU
    """
    global _ocr, _use_gpu, _start_time

    from paddleocr import PaddleOCR

    _use_gpu = use_gpu
    logger.info("正在加载 PaddleOCR PP-OCRv5（device=%s）...", "gpu" if use_gpu else "cpu")

    # PP-OCRv5 模型配置：
    #   text_detection_model_name:    PP-OCRv5_server_det  （高精度检测）
    #   text_recognition_model_name:  en_PP-OCRv5_mobile_rec（英文数字识别）
    # use_textline_orientation=False 关闭行方向分类，加快推理速度
    # enable_mkldnn=False 禁用 oneDNN，避免兼容性问题（CPU 模式下有效）
    # device="gpu"/"cpu" 为 PaddleOCR 3.x 新版设备参数
    _ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_textline_orientation=False,
        device="gpu" if use_gpu else "cpu",
        enable_mkldnn=False,
    )

    _start_time = time.time()
    logger.info("PaddleOCR 模型加载完成！")

    # GPU Warm-up：用一张空白图片做一次推理，触发 CUDA 内核编译和显存分配
    # 这样用户第一次请求时不会遇到 ~5000ms 的冷启动延迟
    logger.info("正在执行 GPU Warm-up 推理...")
    t_warmup = time.time()
    try:
        dummy = np.ones((64, 200, 3), dtype=np.uint8) * 200  # 浅灰色空白图
        _ocr.ocr(dummy)
        logger.info("GPU Warm-up 完成，耗时 %.2f 秒", time.time() - t_warmup)
    except Exception as exc:
        logger.warning("GPU Warm-up 失败（不影响正��使用）: %s", exc)


# ------------------------------------------------------------------
# 图像预处理
# ------------------------------------------------------------------

def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    图像预处理：自动检测深色背景并反色，放大至目标高度，锐化文字边缘。

    处理流程：
        1. 若图像平均亮度 < 128，判定为深色背景，执行反色
        2. 若图像高度 < 128px，按比例放大至 128px（最大放大 8x）
        3. 放大后做 Unsharp Mask 锐化，恢复文字边缘清晰度

    Args:
        img_bgr: BGR 格式原始图像

    Returns:
        预处理后的 BGR 图像
    """
    # 1. 深色背景检测与反色
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 128:
        img_bgr = cv2.bitwise_not(img_bgr)
        logger.debug("检测到深色背景（亮度=%.1f），已执行反色", mean_brightness)

    # 2. 小图放大（目标高度 128px，最大放大 8x）
    h, w = img_bgr.shape[:2]
    if h < 128:
        scale = min(128.0 / h, 8.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("小图放大 %.1fx: %dx%d → %dx%d", scale, w, h, new_w, new_h)

        # 3. Unsharp Mask 锐化：增强放大后的文字边缘
        blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
        img_bgr = cv2.addWeighted(img_bgr, 1.8, blurred, -0.8, 0)

    return img_bgr


# ------------------------------------------------------------------
# 推理核心
# ------------------------------------------------------------------

def _recognize_text(img_bgr: np.ndarray) -> tuple:
    """
    使用 PaddleOCR 识别图片中的文字（线程安全）。

    Args:
        img_bgr: BGR 格式的 numpy 数组

    Returns:
        (raw_text, number) 元组
        - raw_text: 所有识别片段拼接的原始文本
        - number:   从文本中提取的整数，未识别到返回 -1
    """
    # 预处理：反色 + 小图放大 + 锐化
    img_bgr = _preprocess(img_bgr)

    # PaddleOCR 3.x: ocr() 等同于 predict()，返回 list[OCRResult]
    # OCRResult 是类字典对象，文本在 rec_texts，置信度在 rec_scores
    with _infer_lock:
        result = _ocr.ocr(img_bgr)

    if not result:
        return "", -1

    ocr_res = result[0]  # 第一张图的结果，OCRResult 对象
    rec_texts = ocr_res.get("rec_texts", [])
    rec_scores = ocr_res.get("rec_scores", [])

    if not rec_texts:
        return "", -1

    texts = []
    for text, confidence in zip(rec_texts, rec_scores):
        if confidence >= 0.5:
            texts.append(text)
            logger.debug("识别片段: '%s' (置信度: %.2f)", text, confidence)

    raw_text = " ".join(texts)

    # 提取数字
    digits = re.findall(r"\d+", raw_text)
    number = int("".join(digits)) if digits else -1

    return raw_text, number


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
        "engine": "PaddleOCR",
        "gpu": _use_gpu,
        "model_loaded": _ocr is not None,
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
    if _ocr is None:
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
        raw_text, number = _recognize_text(img)

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
    parser = argparse.ArgumentParser(description="PaddleOCR 常驻 HTTP 服务")
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
    print("  PaddleOCR 常驻 HTTP 服务")
    print("=" * 50)

    init_model(use_gpu=not args.cpu)

    print(f"\n[OCR Server] 服务启动在 http://localhost:{args.port}")
    print(f"[OCR Server] 健康检查: GET  http://localhost:{args.port}/health")
    print(f"[OCR Server] 识别接口: POST http://localhost:{args.port}/recognize")
    print(f"[OCR Server] 性能指标: GET  http://localhost:{args.port}/metrics")
    print("=" * 50)

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
