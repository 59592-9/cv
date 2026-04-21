# -*- coding: utf-8 -*-
"""
视觉引擎模块 - TrOCR 版本

使用微软 TrOCR (Vision Transformer) 进行高精度数字识别。
支持本地模式（直接加载模型）和服务模式（HTTP 调用 ocr_server.py）。

优化点：
- 图像预处理增强（灰度化、二值化、降噪、放大）提升 OCR 准确率
- 服务模式增加重试机制与超时控制
- 统一返回值语义：识别失败统一返回 -1
- 日志使用 logging 模块，支持调试级别控制
- 截图对象复用，避免重复创建 mss 实例
"""

import os
import re
import time
import base64
import logging
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import mss

logger = logging.getLogger(__name__)


class VisionEngine:
    """
    视觉引擎 - 基于 TrOCR 的高精度数字识别

    支持两种运行模式：
    1. 本地模式（默认）：直接加载 TrOCR 模型到 GPU/CPU
    2. 服务模式：通过 HTTP 调用 ocr_server.py 的 API

    图像预处理流程（提升 OCR 准确率）：
        原图 → 灰度化 → 放大 2x → 自适应二值化 → 降噪
    """

    # 服务模式请求超时（秒）
    _REQUEST_TIMEOUT = 10
    # 服务模式最大重试次数
    _MAX_RETRIES = 3
    # 图像放大倍数（提升小字体识别率）
    _SCALE_FACTOR = 2.0

    def __init__(
        self,
        debug: bool = False,
        server_url: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        初始化视觉引擎。

        Args:
            debug:      是否开启调试模式（输出 OCR 原始文本等详细信息）
            server_url: OCR 服务地址（如 "http://localhost:5000"），
                        提供则使用服务模式，None 则使用本地模式
            use_gpu:    本地模式下是否优先使用 GPU（默认 True）
        """
        self._debug = debug
        self._server_url = server_url
        # 复用 mss 截图实例，避免每次截图重新初始化
        self._sct = mss.mss()

        if server_url:
            self._init_server_mode(server_url)
        else:
            self._init_local_mode(use_gpu)

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_server_mode(self, server_url: str) -> None:
        """初始化服务模式，验证远端 OCR 服务可用性。"""
        logger.info("使用 OCR 服务模式: %s", server_url)
        import requests

        self._session = requests.Session()
        # 设置连接池大小，提升并发性能
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=2,
            pool_maxsize=4,
            max_retries=0,  # 重试逻辑由本模块自行控制
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        try:
            resp = self._session.get(
                f"{server_url}/health", timeout=5
            )
            if resp.status_code == 200:
                info = resp.json()
                logger.info(
                    "OCR 服务连接成功 - GPU: %s", info.get("gpu", "unknown")
                )
            else:
                logger.warning("OCR 服务返回异常状态码: %d", resp.status_code)
        except Exception as exc:
            logger.warning("无法连接 OCR 服务: %s，将在调用时重试", exc)

    def _init_local_mode(self, use_gpu: bool) -> None:
        """初始化本地 TrOCR 模型。"""
        logger.info("正在初始化 TrOCR 引擎（首次运行需下载模型约 1GB）...")
        print("[VisionEngine] 正在初始化 TrOCR 引擎（首次运行需下载模型约 1GB）...")
        t_start = time.perf_counter()

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from PIL import Image

        self._Image = Image

        # 选择计算设备
        if use_gpu and torch.cuda.is_available():
            self._device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("使用 GPU: %s", gpu_name)
            print(f"[VisionEngine] 使用 GPU: {gpu_name}")
        else:
            self._device = torch.device("cpu")
            if use_gpu:
                logger.warning("GPU 不可用，回退到 CPU 模式")
                print("[VisionEngine] GPU 不可用，回退到 CPU 模式")
            else:
                logger.info("使用 CPU 模式")
                print("[VisionEngine] 使用 CPU 模式")

        model_name = "microsoft/trocr-base-printed"
        logger.info("加载模型: %s", model_name)
        print(f"[VisionEngine] 加载模型: {model_name}")

        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

        t_elapsed = time.perf_counter() - t_start
        logger.info("TrOCR 引擎初始化完成，耗时 %.2f 秒", t_elapsed)
        print(f"[VisionEngine] TrOCR 引擎初始化完成")
        print(f"  设备: {self._device}")
        print(f"  初始化耗时: {t_elapsed:.2f} 秒")

        if self._device.type == "cuda":
            import torch
            mem = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  GPU 显存占用: {mem:.0f} MB")

    # ------------------------------------------------------------------
    # 图像预处理
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        对截图进行预处理，提升 OCR 识别准确率。

        处理流程：
            1. 放大图像（小字体在高分辨率下识别更准）
            2. 转灰度
            3. 自适应二值化（应对不均匀光照）
            4. 形态学降噪（去除细小噪点）
            5. 转回 BGR（TrOCR 输入需要 RGB，后续在 _recognize_local 中转换）

        Args:
            img_bgr: BGR 格式的原始截图

        Returns:
            预处理后的 BGR 图像
        """
        # 1. 放大
        h, w = img_bgr.shape[:2]
        img_scaled = cv2.resize(
            img_bgr,
            (int(w * self._SCALE_FACTOR), int(h * self._SCALE_FACTOR)),
            interpolation=cv2.INTER_CUBIC,
        )

        # 2. 灰度化
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

        # 3. 自适应二值化（blockSize=15, C=8 适合数字场景）
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # 4. 形态学开运算降噪（去除孤立噪点）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 5. 转回 BGR（保持接口一致，_recognize_local 内部再转 RGB）
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # 识别核心
    # ------------------------------------------------------------------

    def _recognize_local(self, img_bgr: np.ndarray) -> str:
        """
        本地模式：使用 TrOCR 模型识别图片中的文字。

        Args:
            img_bgr: BGR 格式的 numpy 数组图片

        Returns:
            识别出的原始文本字符串
        """
        import torch

        img_preprocessed = self._preprocess(img_bgr)
        img_rgb = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)
        pil_image = self._Image.fromarray(img_rgb)

        pixel_values = self._processor(
            images=pil_image,
            return_tensors="pt",
        ).pixel_values.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return text.strip()

    def _recognize_via_server(
        self, img_bgr: np.ndarray
    ) -> Tuple[str, dict]:
        """
        服务模式：通过 HTTP 发送图片到 OCR 服务，带重试机制。

        Args:
            img_bgr: BGR 格式的 numpy 数组图片

        Returns:
            (识别文本, 详细结果字典) 的元组；失败时返回 ("", {})
        """
        img_preprocessed = self._preprocess(img_bgr)
        _, buffer = cv2.imencode(".png", img_preprocessed)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = self._session.post(
                    f"{self._server_url}/recognize",
                    json={"image_base64": img_base64},
                    timeout=self._REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("success"):
                    return result.get("raw_text", ""), result
                else:
                    logger.debug(
                        "服务端识别失败: %s", result.get("error", "unknown")
                    )
                    return "", result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "服务请求失败（第 %d/%d 次）: %s",
                    attempt, self._MAX_RETRIES, exc,
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(0.1 * attempt)  # 指数退避

        logger.error("服务请求全部失败: %s", last_exc)
        return "", {}

    # ------------------------------------------------------------------
    # 数字提取
    # ------------------------------------------------------------------

    def _extract_number(self, text: str) -> int:
        """
        从 OCR 识别文本中提取整数。

        策略：拼接所有连续数字片段，转换为整数。
        例如："¥ 150,000" → "150000" → 150000

        Args:
            text: OCR 识别的原始文本

        Returns:
            提取的整数；文本为空或无数字时返回 -1
        """
        if not text:
            return -1

        if self._debug:
            logger.debug("OCR 原始文本: '%s'", text)
            print(f"[VisionEngine] OCR 原始文本: '{text}'")

        digits = re.findall(r"\d+", text)
        if not digits:
            if self._debug:
                logger.debug("未从文本中提取到数字")
                print("[VisionEngine] 未从文本中提取到数字")
            return -1

        number_str = "".join(digits)
        if self._debug:
            logger.debug("提取数字: '%s'", number_str)
            print(f"[VisionEngine] 提取数字: '{number_str}'")

        try:
            return int(number_str)
        except ValueError:
            logger.warning("数字转换失败: '%s'", number_str)
            return -1

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def capture_and_recognize(self, monitor_dict: dict) -> int:
        """
        截取屏幕区域并识别数字。

        Args:
            monitor_dict: mss 格式的屏幕区域字典
                          {"top": int, "left": int, "width": int, "height": int}

        Returns:
            识别出的价格整数；失败返回 -1
        """
        try:
            screenshot = self._sct.grab(monitor_dict)
            img = np.array(screenshot)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if self._server_url:
                text, _ = self._recognize_via_server(img_bgr)
            else:
                text = self._recognize_local(img_bgr)

            return self._extract_number(text)

        except Exception as exc:
            logger.error("截屏识别异常: %s", exc)
            print(f"[VisionEngine] 截屏识别异常: {exc}")
            return -1

    def recognize_from_image(self, image_path: str) -> int:
        """
        从图片文件识别数字。

        Args:
            image_path: 图片文件路径

        Returns:
            识别出的整数；失败返回 -1
        """
        try:
            if not os.path.exists(image_path):
                logger.error("图片不存在: %s", image_path)
                print(f"[VisionEngine] 图片不存在: {image_path}")
                return -1

            img = cv2.imread(image_path)
            if img is None:
                logger.error("无法读取图片: %s", image_path)
                print(f"[VisionEngine] 无法读取: {image_path}")
                return -1

            if self._server_url:
                text, _ = self._recognize_via_server(img)
            else:
                text = self._recognize_local(img)

            return self._extract_number(text)

        except Exception as exc:
            logger.error("图片识别异常: %s", exc)
            print(f"[VisionEngine] 图片识别异常: {exc}")
            return -1

    def recognize_from_image_verbose(
        self, image_path: str
    ) -> Tuple[int, str, Any]:
        """
        从图片文件识别数字，返回详细结果。

        Args:
            image_path: 图片文件路径

        Returns:
            (数字, 原始文本, 详细信息) 三元组；
            失败时返回 (-1, "", [])
        """
        try:
            if not os.path.exists(image_path):
                logger.error("图片不存在: %s", image_path)
                return -1, "", []

            img = cv2.imread(image_path)
            if img is None:
                logger.error("无法读取图片: %s", image_path)
                return -1, "", []

            if self._server_url:
                text, details = self._recognize_via_server(img)
                number = self._extract_number(text)
                return number, text, details
            else:
                text = self._recognize_local(img)
                number = self._extract_number(text)
                return number, text, []

        except Exception as exc:
            logger.error("图片识别异常: %s", exc)
            print(f"[VisionEngine] 异常: {exc}")
            return -1, "", []
