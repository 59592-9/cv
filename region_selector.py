# -*- coding: utf-8 -*-
"""
区域 OCR 识别工具 - 入口 / 应用控制器 (region_selector.py)

职责：
    - 解析命令行参数
    - 初始化后端（OcrBackend）和前端（MainWindow）
    - 作为 AppController 协调前后端交互

架构说明：
    ┌─────────────────────────────────────────────────────┐
    │                   region_selector.py                │
    │                   (AppController)                   │
    │                                                     │
    │   selector_backend.py          selector_ui.py       │
    │   ┌──────────────────┐        ┌──────────────────┐  │
    │   │  OcrBackend      │◄──────►│  MainWindow      │  │
    │   │  capture_full    │        │  FullscreenOverlay│  │
    │   │  crop_region     │        │                  │  │
    │   │  recognize()     │        │                  │  │
    │   └──────────────────┘        └──────────────────┘  │
    └─────────────────────────────────────────────────────┘

截屏时序（关键）：
    1. 主窗口先 withdraw() 隐藏
    2. 通过 root.after(80, ...) 延迟 80ms，等待屏幕真正刷新
    3. 后端截全屏（此时屏幕干净，无主窗口遮挡）
    4. 创建覆盖层展示截图
    5. 用户框选 → 识别 → 恢复主窗口

用法：
    python region_selector.py                              # 本地 PaddleOCR 模式
    python region_selector.py --server http://localhost:5000  # 服务模式（推荐）
    python region_selector.py --cpu                        # 本地 CPU 模式
"""

import argparse
import logging
from typing import Optional

import numpy as np

from selector_backend import (
    OcrBackend,
    ScreenRegion,
    capture_fullscreen,
    crop_region,
)
from selector_ui import FullscreenOverlay, MainWindow

# ------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("region_selector")


# ======================================================================
# 应用控制器
# ======================================================================

class AppController:
    """
    应用控制器：协调前端 UI 与后端逻辑。

    数据流：
        1. 用户点击"框选识别区域"
           → _on_select_clicked()
           → 主窗口隐藏
           → 延迟 80ms 后 _do_capture_and_overlay()
           → 后端截全屏（屏幕干净）
           → 打开 FullscreenOverlay

        2. 用户在覆盖层框选并点击"识别"
           → _on_region_confirmed(region, fullscreen_img)
           → 后端裁剪区域
           → 后端调用 OCR
           → 主窗口恢复并展示结果
    """

    def __init__(self, server_url: Optional[str] = None, use_gpu: bool = True):
        """
        Args:
            server_url: OCR 服务地址，None 则使用本地模式
            use_gpu:    本地模式下是否使用 GPU
        """
        self._backend = OcrBackend(server_url=server_url, use_gpu=use_gpu)
        self._window = MainWindow(
            mode_description=self._backend.mode_description,
            on_select_cb=self._on_select_clicked,
        )

    # ------------------------------------------------------------------
    # 事件处理（前端 → 控制器 → 后端）
    # ------------------------------------------------------------------

    def _on_select_clicked(self) -> None:
        """
        用户点击"框选识别区域"按钮时触发。

        正确时序：
            1. 主窗口先隐藏（withdraw），让屏幕真正刷新
            2. 通过 after(80ms) 延迟，等待 Windows 完成窗口重绘
            3. 再截全屏（此时屏幕干净，无主窗口遮挡）
        """
        logger.info("用户触发框选，隐藏主窗口准备截屏")
        self._window.set_result("正在截屏...")

        # 步骤1：先隐藏主窗口
        self._window.hide()

        # 步骤2：延迟 80ms 后再截屏，确保主窗口已从屏幕消失
        self._window.get_root().after(80, self._do_capture_and_overlay)

    def _do_capture_and_overlay(self) -> None:
        """
        延迟回调：截全屏并打开覆盖层。
        此时主窗口已隐藏，截图不含主窗口。
        """
        root = self._window.get_root()

        # 步骤3：截全屏
        try:
            fullscreen_img, screen_left, screen_top = capture_fullscreen()
        except Exception as exc:
            logger.error("截屏失败: %s", exc)
            self._window.restore()
            self._window.set_result(f"截屏失败：{exc}")
            return

        logger.info(
            "截全屏完成: %dx%d, 偏移(%d,%d)",
            fullscreen_img.shape[1], fullscreen_img.shape[0],
            screen_left, screen_top,
        )

        # 步骤4：创建覆盖层（主窗口已隐藏，Toplevel 仍可正常显示）
        overlay = FullscreenOverlay(
            parent_root=root,
            fullscreen_img=fullscreen_img,
            screen_left=screen_left,
            screen_top=screen_top,
            on_confirm_cb=self._on_region_confirmed,
        )

        # 步骤5：阻塞等待用户框选完成
        overlay.wait()

        # 步骤6：恢复主窗口
        self._window.restore()

    def _on_region_confirmed(
        self,
        region: ScreenRegion,
        fullscreen_img: np.ndarray,
    ) -> None:
        """
        用户在覆盖层确认框选区域后触发。

        Args:
            region:         用户框选的屏幕区域
            fullscreen_img: 全屏截图（由覆盖层原样传回）
        """
        logger.info(
            "用户确认框选区域: (%d,%d)→(%d,%d) %dx%d px",
            region.x1, region.y1, region.x2, region.y2,
            region.width, region.height,
        )

        self._window.set_result("识别中，请稍候...")

        # 后端裁剪
        try:
            crop = crop_region(fullscreen_img, region)
        except ValueError as exc:
            logger.error("裁剪失败: %s", exc)
            self._window.set_result(f"裁剪失败：{exc}")
            return

        # 后端识别
        result = self._backend.recognize(crop)
        logger.info("识别完成: %s", result)

        # 前端展示识别结果
        self._window.set_result(str(result))

        # 前端展示坐标（供填入 BotConfig.price_region）
        coord_str = region.to_config_str()
        self._window.set_coord(coord_str)
        logger.info("区域坐标: %s", coord_str)

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def run(self) -> None:
        """启动应用（阻塞直到窗口关闭）。"""
        self._window.run()


# ======================================================================
# 入口
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="区域 OCR 识别工具 - 框选屏幕区域并识别文字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python region_selector.py
  python region_selector.py --server http://localhost:5000
  python region_selector.py --cpu
        """,
    )
    parser.add_argument(
        "--server", type=str, default=None, metavar="URL",
        help="OCR 服务地址（如 http://localhost:5000），不指定则使用本地 PaddleOCR 模式",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="本地模式强制使用 CPU",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    AppController(
        server_url=args.server,
        use_gpu=not args.cpu,
    ).run()
