# -*- coding: utf-8 -*-
"""
区域 OCR 识别工具 - 前端 UI 模块 (selector_ui.py)

职责：
    - 主窗口布局与控件
    - 全屏框选覆盖层（FullscreenOverlay）
    - 用户交互事件处理
    - 将用户操作结果（框选区域）回调给外部逻辑

本模块不包含任何 OCR / 截图业务逻辑，所有数据处理均通过回调传递。

交互设计：
    - 鼠标拖拽框选后，松开鼠标立即自动触发识别，无需点击确认按钮
    - 底部按钮栏只保留"重选"和"退出"
"""

import signal
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from selector_backend import ScreenRegion


# ======================================================================
# 全屏框选覆盖层
# ======================================================================

class FullscreenOverlay:
    """
    全屏截图覆盖层，用于拖拽框选识别区域。

    职责（仅 UI）：
        - 将传入的全屏截图铺满屏幕
        - 响应鼠标拖拽，绘制选框
        - 鼠标释放后自动将 ScreenRegion 通过回调传出（无需点击确认）
        - 不执行任何截图或 OCR 操作

    回调约定：
        on_confirm_cb(region: ScreenRegion, fullscreen_img: np.ndarray)
    """

    # 底部按钮栏高度（像素）
    _BTN_BAR_H: int = 50

    def __init__(
        self,
        parent_root: tk.Tk,
        fullscreen_img: np.ndarray,
        screen_left: int,
        screen_top: int,
        on_confirm_cb: Callable[[ScreenRegion, np.ndarray], None],
    ):
        """
        Args:
            parent_root:     主 Tk 根窗口（仅用于事件循环，覆盖层独立显示）
            fullscreen_img:  全屏 BGR 截图（由后端提供）
            screen_left:     虚拟屏幕左边界坐标
            screen_top:      虚拟屏幕上边界坐标
            on_confirm_cb:   用户框选完成后的回调
                             fn(region: ScreenRegion, img: np.ndarray)
        """
        self._fullscreen_img = fullscreen_img
        self._on_confirm_cb = on_confirm_cb

        h, w = fullscreen_img.shape[:2]
        self._img_w = w
        self._img_h = h

        # 框选状态
        self._start_x: int = 0
        self._start_y: int = 0
        self._end_x: int = 0
        self._end_y: int = 0
        self._drawing: bool = False
        self._rect_id: Optional[int] = None
        self._has_selection: bool = False

        # ---- 覆盖窗口 ----
        # 窗口总高度 = 屏幕高度
        # Canvas 高度 = 屏幕高度 - 按钮栏高度（按钮栏在屏幕内可见）
        canvas_h = h - self._BTN_BAR_H

        self._win = tk.Toplevel(parent_root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.attributes("-alpha", 1.0)
        self._win.geometry(f"{w}x{h}+{screen_left}+{screen_top}")

        self._build_canvas(w, canvas_h)
        self._build_btn_bar()
        self._bind_events()

        # 强制渲染
        self._win.update_idletasks()
        self._win.update()
        self._win.lift()
        self._win.focus_force()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_canvas(self, w: int, h: int) -> None:
        """构建截图展示 Canvas 及提示文字。"""
        img_rgb = cv2.cvtColor(self._fullscreen_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)

        self._canvas = tk.Canvas(
            self._win, width=w, height=h,
            cursor="crosshair", highlightthickness=0, bd=0,
        )
        self._canvas.pack(side=tk.TOP)
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)

        self._hint_id = self._canvas.create_text(
            w // 2, 28,
            text="拖拽鼠标框选要识别的区域，松开鼠标后自动识别",
            fill="yellow",
            font=("微软雅黑", 13, "bold"),
        )

    def _build_btn_bar(self) -> None:
        """构建底部固定按钮栏（仅保留重选和退出）。"""
        bar = tk.Frame(self._win, bg="#1e1e1e", height=self._BTN_BAR_H)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)

        btn_cfg = dict(font=("微软雅黑", 11), relief=tk.FLAT, width=10)

        tk.Button(
            bar, text="↺ 重选",
            bg="#2196F3", fg="white",
            command=self._on_reset,
            **btn_cfg,
        ).pack(side=tk.LEFT, padx=14, pady=8)

        tk.Label(
            bar,
            text="松开鼠标后自动识别",
            bg="#1e1e1e", fg="#aaaaaa",
            font=("微软雅黑", 10),
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            bar, text="✖ 退出",
            bg="#f44336", fg="white",
            command=self._win.destroy,
            **btn_cfg,
        ).pack(side=tk.RIGHT, padx=14, pady=8)

    def _bind_events(self) -> None:
        """绑定鼠标和键盘事件。"""
        self._canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self._canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self._win.bind("<Escape>", lambda _: self._win.destroy())

    # ------------------------------------------------------------------
    # 鼠标事件处理
    # ------------------------------------------------------------------

    def _on_mouse_press(self, event: tk.Event) -> None:
        """鼠标按下：开始新的框选。"""
        self._start_x = event.x
        self._start_y = event.y
        self._drawing = True
        self._has_selection = False
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
            self._rect_id = None
        self._canvas.itemconfig(
            self._hint_id,
            text="拖拽鼠标框选要识别的区域，松开鼠标后自动识别",
        )

    def _on_mouse_drag(self, event: tk.Event) -> None:
        """鼠标拖拽：实时更新选框。"""
        if not self._drawing:
            return
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
        self._end_x = event.x
        self._end_y = event.y
        self._rect_id = self._canvas.create_rectangle(
            self._start_x, self._start_y,
            self._end_x, self._end_y,
            outline="#FF4444", width=2, dash=(4, 2),
        )

    def _on_mouse_release(self, event: tk.Event) -> None:
        """鼠标释放：框选完成，直接触发识别（无需点击按钮）。"""
        if not self._drawing:
            return
        self._drawing = False
        self._end_x = event.x
        self._end_y = event.y

        region = self._current_region()
        if not region.is_valid:
            if self._rect_id is not None:
                self._canvas.delete(self._rect_id)
                self._rect_id = None
            self._canvas.itemconfig(
                self._hint_id,
                text="选区太小，请重新拖拽框选",
            )
            return

        # 框选有效，更新提示并立即触发识别
        self._has_selection = True
        self._canvas.itemconfig(
            self._hint_id,
            text=f"已选: ({region.x1},{region.y1})→({region.x2},{region.y2})  "
                 f"{region.width}×{region.height}px  正在识别...",
        )
        self._win.update_idletasks()
        self._on_confirm()

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def _on_reset(self) -> None:
        """重选：清除当前选框，等待重新框选。"""
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
            self._rect_id = None
        self._has_selection = False
        self._canvas.itemconfig(
            self._hint_id,
            text="拖拽鼠标框选要识别的区域，松开鼠标后自动识别",
        )

    def _on_confirm(self) -> None:
        """触发识别：将选区和截图通过回调传出，关闭覆盖层。"""
        if not self._has_selection:
            return
        region = self._current_region()
        if not region.is_valid:
            messagebox.showwarning("提示", "框选区域无效，请重新框选", parent=self._win)
            return
        self._win.destroy()
        self._on_confirm_cb(region, self._fullscreen_img)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _current_region(self) -> ScreenRegion:
        """根据当前鼠标坐标构造 ScreenRegion（自动排序坐标，边界保护）。"""
        x1, x2 = sorted([self._start_x, self._end_x])
        y1, y2 = sorted([self._start_y, self._end_y])
        x1 = max(0, x1);            y1 = max(0, y1)
        x2 = min(self._img_w, x2);  y2 = min(self._img_h, y2)
        return ScreenRegion(x1=x1, y1=y1, x2=x2, y2=y2)

    def wait(self) -> None:
        """阻塞直到覆盖层窗口关闭。"""
        self._win.wait_window()


# ======================================================================
# 主窗口
# ======================================================================

class MainWindow:
    """
    区域 OCR 识别工具主窗口。

    职责（仅 UI）：
        - 展示工具标题和当前模式
        - 提供"框选识别区域"入口按钮
        - 展示识别结果
        - 通过回调与外部逻辑（AppController）解耦

    回调约定：
        on_select_cb()  — 用户点击"框选识别区域"时触发
    """

    def __init__(self, mode_description: str, on_select_cb: Callable[[], None]):
        """
        Args:
            mode_description: 识别模式描述字符串（由后端提供）
            on_select_cb:     用户点击框选按钮时的回调
        """
        self._on_select_cb = on_select_cb
        self._coord_str: str = ""  # 最近一次框选的坐标字符串

        self._root = tk.Tk()
        self._root.title("区域 OCR 识别工具")
        self._root.resizable(False, False)
        self._center(480, 320)

        self._build_ui(mode_description)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self, mode_description: str) -> None:
        root = self._root
        large  = tkfont.Font(family="微软雅黑", size=14, weight="bold")
        bold   = tkfont.Font(family="微软雅黑", size=11, weight="bold")
        normal = tkfont.Font(family="微软雅黑", size=10)
        mono   = tkfont.Font(family="Consolas", size=10)

        tk.Label(root, text="区域 OCR 识别工具", font=large, pady=12).pack()
        tk.Label(root, text=f"识别模式：{mode_description}", font=normal, fg="#555").pack()
        tk.Frame(root, height=1, bg="#cccccc").pack(fill=tk.X, padx=20, pady=10)

        tk.Button(
            root,
            text="🖱  框选识别区域",
            font=tkfont.Font(family="微软雅黑", size=12, weight="bold"),
            bg="#FF9800", fg="white", relief=tk.FLAT,
            command=self._on_select_cb,
            width=20, height=2,
        ).pack(pady=6)

        # ---- 识别结果区 ----
        rf = tk.LabelFrame(root, text=" 识别结果 ", font=bold, padx=10, pady=6)
        rf.pack(fill=tk.X, padx=20, pady=(4, 2))

        self._result_var = tk.StringVar(value="点击上方按钮开始框选...")
        tk.Label(
            rf,
            textvariable=self._result_var,
            font=tkfont.Font(family="微软雅黑", size=12),
            fg="#222", wraplength=420, justify=tk.LEFT,
        ).pack(anchor="w")

        # ---- 坐标区 ----
        cf = tk.LabelFrame(root, text=" 区域坐标（price_region） ", font=bold, padx=10, pady=6)
        cf.pack(fill=tk.X, padx=20, pady=(2, 8))

        self._coord_var = tk.StringVar(value="框选后自动显示坐标...")
        tk.Label(
            cf,
            textvariable=self._coord_var,
            font=mono,
            fg="#1565C0", wraplength=420, justify=tk.LEFT,
        ).pack(anchor="w", side=tk.LEFT, expand=True)

        self._copy_btn = tk.Button(
            cf,
            text="📋 复制",
            font=tkfont.Font(family="微软雅黑", size=10),
            bg="#4CAF50", fg="white", relief=tk.FLAT,
            command=self._on_copy_coord,
            state=tk.DISABLED,
            width=8,
        )
        self._copy_btn.pack(side=tk.RIGHT, padx=(8, 0))

    # ------------------------------------------------------------------
    # 公开接口（供 AppController 调用）
    # ------------------------------------------------------------------

    def set_result(self, text: str) -> None:
        """更新识别结果文本。"""
        self._result_var.set(text)
        self._root.update_idletasks()

    def set_coord(self, coord_str: str) -> None:
        """
        更新坐标显示并启用复制按钮。

        Args:
            coord_str: 形如 '{"top": 400, "left": 800, "width": 200, "height": 50}'
        """
        self._coord_str = coord_str
        self._coord_var.set(coord_str)
        self._copy_btn.config(state=tk.NORMAL)
        self._root.update_idletasks()

    def _on_copy_coord(self) -> None:
        """将坐标字符串复制到系统剪贴板。"""
        if not self._coord_str:
            return
        self._root.clipboard_clear()
        self._root.clipboard_append(self._coord_str)
        # 短暂改变按钮文字给用户反馈
        self._copy_btn.config(text="✔ 已复制", bg="#388E3C")
        self._root.after(1500, lambda: self._copy_btn.config(text="📋 复制", bg="#4CAF50"))

    def hide(self) -> None:
        """隐藏主窗口（不销毁，不影响子 Toplevel 的可见性）。"""
        self._root.withdraw()

    def restore(self) -> None:
        """恢复主窗口并置顶。"""
        self._root.deiconify()
        self._root.lift()
        self._root.focus_force()

    def get_root(self) -> tk.Tk:
        """返回 Tk 根窗口实例（供创建 Toplevel 时使用）。"""
        return self._root

    def run(self) -> None:
        """
        启动 Tk 主循环（阻塞）。

        通过每 200ms 调用一次空 after 回调，让 Python 有机会处理
        SIGINT（Ctrl+C）信号，从而支持在终端中用 Ctrl+C 退出。
        """
        def _handle_sigint(*_):
            self._root.destroy()

        signal.signal(signal.SIGINT, _handle_sigint)

        def _poll():
            self._root.after(200, _poll)

        self._root.after(200, _poll)
        self._root.mainloop()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _center(self, w: int, h: int) -> None:
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
