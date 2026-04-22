# -*- coding: utf-8 -*-
"""
视觉引擎模块 - PaddleOCR 版本

使用 PaddleOCR（PP-OCRv5）进行高精度数字识别。
相比传统 OCR，PaddleOCR 对小区域纯数字场景识别准确率更高，速度更快。

支持本地模式（直接调用 PaddleOCR）和服务模式（HTTP 调用 ocr_server.py）。

优化点：
- 使用 PaddleOCR 英文数字模型（en），专为数字/字母场景优化
- 关闭方向分类（use_angle_cls=False），减少不必要推理
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
    视觉引擎 - 基于 PaddleOCR 的高精度数字识别

    支持两种运行模式：
    1. 本地模式（默认）：直接调用 PaddleOCR 模型（GPU/CPU）
    2. 服务模式：通过 HTTP 调用 ocr_server.py 的 API

    PaddleOCR 配置说明：
        - lang='en'：英文数字模型，专为数字/字母场景优化
        - use_angle_cls=False：关闭方向分类，加快推理速度
        - use_gpu：是否使用 GPU 加速
        - show_log=False：关闭 PaddleOCR 内部日志，避免刷屏
    """

    # 服务模式请求超时（秒）
    _REQUEST_TIMEOUT = 10
    # 服务模式最大重试次数
    _MAX_RETRIES = 3

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
        """初始化本地 PaddleOCR 引擎。"""
        logger.info("正在初始化 PaddleOCR 引擎...")
        print("[VisionEngine] 正在初始化 PaddleOCR 引擎...")
        t_start = time.perf_counter()

        from paddleocr import PaddleOCR

        # PP-OCRv5 模型配置：
        #   text_detection_model_name:    PP-OCRv5_server_det  （高精度检测）
        #   text_recognition_model_name:  en_PP-OCRv5_mobile_rec（英文数字识别）
        # use_textline_orientation=False 关闭行方向分类，加快推理速度
        # enable_mkldnn=False 禁用 oneDNN，避免兼容性问题（CPU 模式下有效）
        # device="gpu"/"cpu" 为 PaddleOCR 3.x 新版设备参数
        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            use_textline_orientation=False,
            device="gpu" if use_gpu else "cpu",
            enable_mkldnn=False,
        )

        t_elapsed = time.perf_counter() - t_start
        logger.info("PaddleOCR 引擎初始化完成，耗时 %.2f 秒", t_elapsed)
        print(f"[VisionEngine] PaddleOCR 引擎初始化完成")
        print(f"  模式: {'GPU' if use_gpu else 'CPU'}")
        print(f"  初始化耗时: {t_elapsed:.2f} 秒")

    # ------------------------------------------------------------------
    # 图像预处理
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        图像预处理：自动检测深色背景并反色，放大至目标高度，锐化文字边缘。

        处理流程：
            1. 若图像平均亮度 < 128，判定为深色背景，执行反色
               （深色背景白色文字 → 白底黑字，提升 PaddleOCR 检测率）
            2. 若图像高度 < 128px，按比例放大至 128px（最大放大 8x）
               （小图放大后识别率更高，128px 是 PaddleOCR 检测模型的推荐最小高度）
            3. 放大后对图像做 Unsharp Mask 锐化
               （放大引入模糊，锐化恢复文字边缘清晰度）

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
            if self._debug:
                logger.debug("检测到深色背景（亮度=%.1f），已执行反色", mean_brightness)

        # 2. 小图放大（目标高度 128px，最大放大 8x）
        h, w = img_bgr.shape[:2]
        if h < 128:
            scale = min(128.0 / h, 8.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_bgr = cv2.resize(
                img_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_CUBIC,
            )
            if self._debug:
                logger.debug("小图放大 %.1fx: %dx%d → %dx%d", scale, w, h, new_w, new_h)

            # 3. Unsharp Mask 锐化：增强放大后的文字边缘
            # 原理：锐化 = 原图 + α × (原图 - 高斯模糊)
            blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=1.5)
            img_bgr = cv2.addWeighted(img_bgr, 1.8, blurred, -0.8, 0)

        return img_bgr

    # ------------------------------------------------------------------
    # 识别核心
    # ------------------------------------------------------------------

    def _recognize_local(self, img_bgr: np.ndarray) -> str:
        """
        本地模式：使用 PaddleOCR 识别图片中的文字。

        Args:
            img_bgr: BGR 格式的 numpy 数组图片

        Returns:
            识别出的原始文本字符串（多行结果以空格拼接）
        """
        # 预处理：自动反色 + 小图放大
        img_bgr = self._preprocess(img_bgr)

        # PaddleOCR 3.x: ocr() 等同于 predict()，返回 list[OCRResult]
        # OCRResult 是类字典对象，文本在 rec_texts，置信度在 rec_scores
        result = self._ocr.ocr(img_bgr)

        if not result:
            return ""

        ocr_res = result[0]  # 第一张图的结果，OCRResult 对象
        rec_texts = ocr_res.get("rec_texts", [])
        rec_scores = ocr_res.get("rec_scores", [])

        if not rec_texts:
            return ""

        # 按置信度过滤并拼接所有识别到的文本
        texts = []
        for text, confidence in zip(rec_texts, rec_scores):
            if confidence >= 0.5:  # 置信度低于 50% 的结果丢弃
                texts.append(text)
                if self._debug:
                    logger.debug(
                        "PaddleOCR 识别: '%s' (置信度: %.2f)", text, confidence
                    )
                    print(f"[VisionEngine] 识别片段: '{text}' (置信度: {confidence:.2f})")

        return " ".join(texts)

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
        _, buffer = cv2.imencode(".png", img_bgr)
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

    def extract_price(self, text: str) -> float:
        """
        从 OCR 识别文本中提取价格（支持小数点）。

        策略：
            1. 优先匹配带小数点的数字，如 "1,234.56" → 1234.56
            2. 去除千位分隔符（逗号），保留小数点
            3. 若无小数点，返回整数值（float 类型）

        例如：
            "¥ 1,234.56" → 1234.56
            "150,000"    → 150000.0
            "99.9"       → 99.9

        Args:
            text: OCR 识别的原始文本

        Returns:
            提取的价格浮点数；文本为空或无数字时返回 -1.0
        """
        if not text:
            return -1.0

        if self._debug:
            logger.debug("OCR 原始文本（价格提取）: '%s'", text)

        # 先尝试匹配带小数点的完整数字（含千位逗号）
        # 例如 "1,234.56" 或 "1234.56"
        match = re.search(r"[\d,]+\.\d+", text)
        if match:
            num_str = match.group().replace(",", "")
            try:
                val = float(num_str)
                if self._debug:
                    logger.debug("提取价格（含小数）: %s → %.4f", match.group(), val)
                return val
            except ValueError:
                pass

        # 无小数点：拼接所有数字片段（去除逗号分隔符）
        # 例如 "150,000" → 150000
        clean = re.sub(r"[^\d]", "", text)
        if not clean:
            return -1.0
        try:
            val = float(clean)
            if self._debug:
                logger.debug("提取价格（整数）: '%s' → %.0f", clean, val)
            return val
        except ValueError:
            logger.warning("价格转换失败: '%s'", clean)
            return -1.0

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
