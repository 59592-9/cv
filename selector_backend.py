# -*- coding: utf-8 -*-
"""
区域 OCR 识别工具 - 后端逻辑模块 (selector_backend.py)

职责：
    - 全屏截图
    - 裁剪框选区域
    - 调用 OCR 服务或本地引擎识别
    - 返回结构化识别结果

本模块不包含任何 UI 代码，可独立测试和复用。
"""

import base64
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class ScreenRegion:
    """
    屏幕区域描述（相对于全屏截图的像素坐标）。

    Attributes:
        x1: 左上角 X
        y1: 左上角 Y
        x2: 右下角 X
        y2: 右下角 Y
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_valid(self) -> bool:
        """区域宽高均大于 4 像素才视为有效选区。"""
        return self.width > 4 and self.height > 4

    def to_mss_dict(self) -> dict:
        """
        转换为 main.py BotConfig.price_region 所需的 mss 格式字典。

        Returns:
            {"top": y1, "left": x1, "width": w, "height": h}
        """
        return {
            "top":    self.y1,
            "left":   self.x1,
            "width":  self.width,
            "height": self.height,
        }

    def to_config_str(self) -> str:
        """
        生成可直接粘贴到 BotConfig.price_region 的代码字符串。

        Returns:
            形如 '{"top": 400, "left": 800, "width": 200, "height": 50}'
        """
        d = self.to_mss_dict()
        return (
            f'{{"top": {d["top"]}, "left": {d["left"]}, '
            f'"width": {d["width"]}, "height": {d["height"]}}}'
        )


@dataclass
class OcrResult:
    """
    OCR 识别结果。

    Attributes:
        success:  是否识别成功
        raw_text: OCR 原始文本
        number:   从文本中提取的整数（-1 表示未提取到）
        error:    错误信息（仅失败时有值）
    """
    success: bool
    raw_text: str
    number: int
    error: str = ""

    def __str__(self) -> str:
        if not self.success:
            return f"识别失败：{self.error or '未识别到内容'}"
        msg = f"原始文本：{self.raw_text}"
        if self.number > 0:
            msg += f"\n提取数字：{self.number}"
        return msg


# ======================================================================
# 截图
# ======================================================================

def capture_fullscreen() -> Tuple[np.ndarray, int, int]:
    """
    截取整个虚拟屏幕（自动适配多显示器）。

    Returns:
        (img_bgr, screen_left, screen_top)
        - img_bgr:      BGR 格式的全屏截图 numpy 数组
        - screen_left:  虚拟屏幕左边界（多显示器时可能为负数）
        - screen_top:   虚拟屏幕上边界
    """
    with mss.mss() as sct:
        # monitors[0] 是所有显示器的合并虚拟区域
        mon = sct.monitors[0]
        shot = sct.grab(mon)
        img = np.array(shot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        logger.debug(
            "全屏截图完成: %dx%d, 偏移(%d, %d)",
            mon["width"], mon["height"], mon["left"], mon["top"],
        )
        return img_bgr, mon["left"], mon["top"]


def crop_region(img_bgr: np.ndarray, region: ScreenRegion) -> np.ndarray:
    """
    从全屏截图中裁剪指定区域。

    Args:
        img_bgr: 全屏 BGR 截图
        region:  要裁剪的区域

    Returns:
        裁剪后的 BGR 图像

    Raises:
        ValueError: 区域无效或裁剪结果为空
    """
    if not region.is_valid:
        raise ValueError(f"无效区域: {region}")

    h, w = img_bgr.shape[:2]
    x1 = max(0, region.x1)
    y1 = max(0, region.y1)
    x2 = min(w, region.x2)
    y2 = min(h, region.y2)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"裁剪结果为空: region={region}, img_size=({w},{h})")

    logger.debug("裁剪区域: (%d,%d)→(%d,%d), 大小: %dx%d", x1, y1, x2, y2, x2-x1, y2-y1)
    return crop


# ======================================================================
# OCR 识别
# ======================================================================

def recognize_via_server(img_bgr: np.ndarray, server_url: str) -> OcrResult:
    """
    将图片发送到 OCR HTTP 服务进行识别。

    Args:
        img_bgr:    BGR 格式图片
        server_url: OCR 服务地址（如 "http://localhost:5000"）

    Returns:
        OcrResult 识别结果
    """
    import requests

    _, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")

    try:
        resp = requests.post(
            f"{server_url}/recognize",
            json={"image_base64": b64},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("success"):
            logger.info(
                "OCR 服务识别成功: '%s' -> %d (%.1f ms)",
                data.get("raw_text", ""),
                data.get("number", -1),
                data.get("time_ms", 0),
            )
            return OcrResult(
                success=True,
                raw_text=data.get("raw_text", ""),
                number=data.get("number", -1),
            )
        else:
            err = data.get("error", "服务端返回失败")
            logger.warning("OCR 服务识别失败: %s", err)
            return OcrResult(success=False, raw_text="", number=-1, error=err)

    except Exception as exc:
        logger.error("OCR 服务请求异常: %s", exc)
        return OcrResult(success=False, raw_text="", number=-1, error=str(exc))


# 模块级 VisionEngine 单例缓存，避免每次识别都重新加载模型
_local_engine_cache: dict = {}  # key: use_gpu(bool) -> VisionEngine


def _get_local_engine(use_gpu: bool = True):
    """获取（或创建）本地 VisionEngine 单例。"""
    from vision import VisionEngine
    key = use_gpu
    if key not in _local_engine_cache:
        logger.info("首次初始化本地 VisionEngine (use_gpu=%s)...", use_gpu)
        _local_engine_cache[key] = VisionEngine(use_gpu=use_gpu)
        logger.info("VisionEngine 初始化完成，后续调用将复用此实例")
    return _local_engine_cache[key]


def recognize_local(img_bgr: np.ndarray, use_gpu: bool = True) -> OcrResult:
    """
    使用本地 VisionEngine（PaddleOCR PP-OCRv5）识别图片。
    VisionEngine 为模块级单例，只在首次调用时初始化，后续复用。

    Args:
        img_bgr:  BGR 格式图片
        use_gpu:  是否使用 GPU

    Returns:
        OcrResult 识别结果
    """
    import os
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, img_bgr)

    try:
        engine = _get_local_engine(use_gpu)
        number, raw_text, _ = engine.recognize_from_image_verbose(tmp.name)
        success = number > 0 or bool(raw_text)
        logger.info("本地 OCR 识别: '%s' -> %d", raw_text, number)
        return OcrResult(success=success, raw_text=raw_text, number=number)
    except Exception as exc:
        logger.error("本地 OCR 识别异常: %s", exc)
        return OcrResult(success=False, raw_text="", number=-1, error=str(exc))
    finally:
        os.unlink(tmp.name)


# ======================================================================
# 统一识别入口
# ======================================================================

class OcrBackend:
    """
    OCR 后端统一接口。

    根据配置自动选择服务模式或本地模式，对外暴露统一的 recognize() 方法。

    使用示例：
        backend = OcrBackend(server_url="http://localhost:5000")
        img, left, top = capture_fullscreen()
        region = ScreenRegion(100, 200, 400, 300)
        crop = crop_region(img, region)
        result = backend.recognize(crop)
        print(result)
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Args:
            server_url: OCR 服务地址，None 则使用本地模式
            use_gpu:    本地模式下是否使用 GPU
        """
        self._server_url = server_url
        self._use_gpu = use_gpu

        mode = f"服务模式 ({server_url})" if server_url else f"本地模式 ({'GPU' if use_gpu else 'CPU'})"
        logger.info("OcrBackend 初始化: %s", mode)

    @property
    def mode_description(self) -> str:
        """返回当前模式描述字符串。"""
        if self._server_url:
            return f"服务模式: {self._server_url}"
        return f"本地模式 ({'GPU' if self._use_gpu else 'CPU'})"

    def recognize(self, img_bgr: np.ndarray) -> OcrResult:
        """
        识别图片内容。

        Args:
            img_bgr: BGR 格式图片（通常为 crop_region() 的返回值）

        Returns:
            OcrResult 识别结果
        """
        if self._server_url:
            return recognize_via_server(img_bgr, self._server_url)
        return recognize_local(img_bgr, self._use_gpu)
