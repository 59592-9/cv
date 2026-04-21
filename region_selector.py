# -*- coding: utf-8 -*-
"""
区域 OCR 识别工具 (Region Selector)

功能流程：
  1. 点击"框选识别区域"按钮
  2. 全屏截图铺满屏幕，鼠标拖拽框选目标区域
  3. 点击"识别"按钮，将框选区域截图发送给 OCR 服务
  4. 主窗口显示识别结果

用法：
    python region_selector.py --server http://localhost:5000   # 服务模式（推荐）
    python region_selector.py                                  # 本地 TrOCR 模式
"""

import argparse
import base64
import sys
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
from typing import Optional, Tuple

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk


# ======================================================================
# 工具函数
# ======================================================================

def capture_fullscreen() -> Tuple[np.ndarray, int, int]:
    """
    截取整个虚拟屏幕（含多显示器）。

    Returns:
        (img_bgr, screen_left, screen_top)
    """
    with mss.mss() as sct:
        # monitor[0] 是所有显示器的合并区域
        mon = sct.monitors[0]
        shot = sct.grab(mon)
        img = np.array(shot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_bgr, mon["left"], mon["top"]


def ocr_via_server(img_bgr: np.ndarray, server_url: str) -> Tuple[bool, str, int]:
    """发送图片到 OCR 服务，返回 (success, raw_text, number)"""
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
        return data.get("success", False), data.get("raw_text", ""), data.get("number", -1)
    except Exception as exc:
        return False, str(exc), -1


def ocr_local(img_bgr: np.ndarray, use_gpu: bool = True) -> Tuple[bool, str, int]:
    """使用本地 VisionEngine 识别，返回 (success, raw_text, number)"""
    import tempfile, os
    from vision import VisionEngine
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, img_bgr)
    try:
        engine = VisionEngine(use_gpu=use_gpu)
        number, raw_text, _ = engine.recognize_from_image_verbose(tmp.name)
        return number > 0, raw_text, number
    finally:
        os.unlink(tmp.name)


# ======================================================================
# 全屏框选覆盖层
# ======================================================================

class FullscreenOverlay:
    """
    全屏截图覆盖层，用于拖拽框选识别区域。

    - 截取全屏后铺满屏幕显示
    - 鼠标拖拽绘制选框
    - 底部固定按钮栏：识别 / 重选 / 退出
    - 按 Esc 退出
    """

    BTN_BAR_H = 50  # 底部按钮栏高度

    def __init__(self, parent_root: tk.Tk, on_recognize_cb):
        """
        Args:
            parent_root:     主 Tk 根窗口
            on_recognize_cb: 识别回调 fn(crop_bgr: np.ndarray)
        """
        self._on_recognize_cb = on_recognize_cb

        # 截全屏
        self._src_bgr, self._screen_left, self._screen_top = capture_fullscreen()
        h, w = self._src_bgr.shape[:2]
        self._img_w = w
        self._img_h = h

        # 框选状态
        self._start_x = self._start_y = 0
        self._end_x = self._end_y = 0
        self._drawing = False
        self._rect_id = None
        self._has_selection = False

        # ---- 创建覆盖窗口 ----
        self._win = tk.Toplevel(parent_root)
        self._win.overrideredirect(True)       # 无边框
        self._win.attributes("-topmost", True) # 置顶
        # 覆盖层总高度 = 截图高度 + 按钮栏
        total_h = h + self.BTN_BAR_H
        self._win.geometry(f"{w}x{total_h}+{self._screen_left}+{self._screen_top}")

        # ---- Canvas 显示全屏截图 ----
        img_rgb = cv2.cvtColor(self._src_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self._canvas = tk.Canvas(
            self._win, width=w, height=h,
            cursor="crosshair", highlightthickness=0, bd=0,
        )
        self._canvas.pack(side=tk.TOP)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        # 提示文字（黄色，居中顶部）
        self._hint_id = self._canvas.create_text(
            w // 2, 28,
            text="拖拽鼠标框选要识别的区域，然后点击 [识别] 按钮",
            fill="yellow",
            font=("微软雅黑", 13, "bold"),
        )

        # ---- 底部按钮栏 ----
        btn_bar = tk.Frame(self._win, bg="#1e1e1e", height=self.BTN_BAR_H)
        btn_bar.pack(side=tk.BOTTOM, fill=tk.X)
        btn_bar.pack_propagate(False)

        btn_cfg = dict(font=("微软雅黑", 11), relief=tk.FLAT, width=10)

        self._btn_ok = tk.Button(
            btn_bar, text="✔ 识别",
            bg="#4CAF50", fg="white",
            command=self._do_recognize,
            state=tk.DISABLED,
            **btn_cfg,
        )
        self._btn_ok.pack(side=tk.LEFT, padx=14, pady=8)

        tk.Button(
            btn_bar, text="↺ 重选",
            bg="#2196F3", fg="white",
            command=self._clear_selection,
            **btn_cfg,
        ).pack(side=tk.LEFT, padx=4, pady=8)

        tk.Button(
            btn_bar, text="✖ 退出",
            bg="#f44336", fg="white",
            command=self._win.destroy,
            **btn_cfg,
        ).pack(side=tk.RIGHT, padx=14, pady=8)

        # ---- 绑定事件 ----
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._win.bind("<Escape>", lambda _: self._win.destroy())
        self._win.focus_force()

    # ------------------------------------------------------------------
    # 鼠标事件
    # ------------------------------------------------------------------

    def _on_press(self, event):
        self._start_x = event.x
        self._start_y = event.y
        self._drawing = True
        self._has_selection = False
        self._btn_ok.config(state=tk.DISABLED)
        if self._rect_id:
            self._canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if not self._drawing:
            return
        if self._rect_id:
            self._canvas.delete(self._rect_id)
        self._end_x = event.x
        self._end_y = event.y
        self._rect_id = self._canvas.create_rectangle(
            self._start_x, self._start_y,
            self._end_x, self._end_y,
            outline="#FF4444", width=2, dash=(4, 2),
        )

    def _on_release(self, event):
        if not self._drawing:
            return
        self._drawing = False
        self._end_x = event.x
        self._end_y = event.y

        x1, x2 = sorted([self._start_x, self._end_x])
        y1, y2 = sorted([self._start_y, self._end_y])

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            if self._rect_id:
                self._canvas.delete(self._rect_id)
                self._rect_id = None
            return

        self._has_selection = True
        self._btn_ok.config(state=tk.NORMAL)
        self._canvas.itemconfig(
            self._hint_id,
            text=f"已选: ({x1},{y1}) → ({x2},{y2})   {x2-x1}×{y2-y1}px   点击 [识别] 提交",
        )

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def _clear_selection(self):
        if self._rect_id:
            self._canvas.delete(self._rect_id)
            self._rect_id = None
        self._has_selection = False
        self._btn_ok.config(state=tk.DISABLED)
        self._canvas.itemconfig(
            self._hint_id,
            text="拖拽鼠标框选要识别的区域，然后点击 [识别] 按钮",
        )

    def _do_recognize(self):
        if not self._has_selection:
            return

        x1, x2 = sorted([self._start_x, self._end_x])
        y1, y2 = sorted([self._start_y, self._end_y])

        # 边界保护
        x1 = max(0, x1);          y1 = max(0, y1)
        x2 = min(self._img_w, x2); y2 = min(self._img_h, y2)

        crop = self._src_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            messagebox.showwarning("提示", "框选区域为空，请重新框选", parent=self._win)
            return

        self._win.destroy()
        self._on_recognize_cb(crop)

    def wait(self):
        """阻塞直到覆盖层关闭"""
        self._win.wait_window()


# ======================================================================
# 主应用
# ======================================================================

class RegionSelectorApp:
    """区域 OCR 识别工具主应用"""

    def __init__(self, server_url: Optional[str] = None, use_gpu: bool = True):
        self._server_url = server_url
        self._use_gpu = use_gpu

        self._root = tk.Tk()
        self._root.title("区域 OCR 识别工具")
        self._root.resizable(False, False)
        self._center_window(self._root, 480, 260)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self._root
        bold   = tkfont.Font(family="微软雅黑", size=11, weight="bold")
        normal = tkfont.Font(family="微软雅黑", size=10)
        large  = tkfont.Font(family="微软雅黑", size=14, weight="bold")

        tk.Label(root, text="区域 OCR 识别工具", font=large, pady=12).pack()

        mode = f"服务模式: {self._server_url}" if self._server_url else "本地模式 (TrOCR)"
        tk.Label(root, text=f"识别模式：{mode}", font=normal, fg="#555").pack()

        tk.Frame(root, height=1, bg="#cccccc").pack(fill=tk.X, padx=20, pady=10)

        # 框选按钮
        tk.Button(
            root, text="🖱  框选识别区域",
            font=tkfont.Font(family="微软雅黑", size=12, weight="bold"),
            bg="#FF9800", fg="white", relief=tk.FLAT,
            command=self._open_overlay,
            width=20, height=2,
        ).pack(pady=6)

        # 结果区域
        rf = tk.LabelFrame(root, text=" 识别结果 ", font=bold, padx=10, pady=6)
        rf.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        self._result_var = tk.StringVar(value="点击上方按钮开始框选...")
        tk.Label(
            rf, textvariable=self._result_var,
            font=tkfont.Font(family="微软雅黑", size=12),
            fg="#222", wraplength=420, justify=tk.LEFT,
        ).pack(anchor="w")

    # ------------------------------------------------------------------
    # 打开框选覆盖层
    # ------------------------------------------------------------------

    def _open_overlay(self):
        # 主窗口最小化，避免遮挡截图
        self._root.iconify()
        time.sleep(0.15)  # 等待最小化动画完成

        overlay = FullscreenOverlay(
            parent_root=self._root,
            on_recognize_cb=self._on_region_selected,
        )
        overlay.wait()

        # 覆盖层关闭后恢复主窗口
        self._root.deiconify()
        self._root.lift()
        self._root.focus_force()

    # ------------------------------------------------------------------
    # OCR 识别
    # ------------------------------------------------------------------

    def _on_region_selected(self, crop_bgr: np.ndarray):
        self._result_var.set("识别中，请稍候...")
        self._root.update()

        try:
            if self._server_url:
                success, raw_text, number = ocr_via_server(crop_bgr, self._server_url)
            else:
                success, raw_text, number = ocr_local(crop_bgr, self._use_gpu)

            if success and raw_text:
                msg = f"原始文本：{raw_text}"
                if number > 0:
                    msg += f"\n提取数字：{number}"
            else:
                msg = f"识别失败：{raw_text or '未识别到内容'}"

            self._result_var.set(msg)

        except Exception as exc:
            self._result_var.set(f"识别异常：{exc}")

    # ------------------------------------------------------------------

    @staticmethod
    def _center_window(win, w, h):
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def run(self):
        self._root.mainloop()


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="区域 OCR 识别工具")
    parser.add_argument(
        "--server", type=str, default=None, metavar="URL",
        help="OCR 服务地址（如 http://localhost:5000），不指定则使用本地 TrOCR 模式",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="本地模式强制使用 CPU",
    )
    args = parser.parse_args()

    RegionSelectorApp(
        server_url=args.server,
        use_gpu=not args.cpu,
    ).run()
