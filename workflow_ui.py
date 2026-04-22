# -*- coding: utf-8 -*-
"""
workflow_ui.py  -  工作流 GUI 组件

包含：
  ToolboxPanel   - 左侧工具箱（点击添加步骤）
  StepCard       - 单个步骤卡片（显示/编辑/移动/删除）
  WorkflowCanvas - 中间可滚动步骤列表
  LogPanel       - 底部运行日志区
  各步骤编辑对话框
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, List, Optional

from workflow_steps import (
    ClickStep, OcrStep, KeyboardStep, TimerStep, ConditionStep,
    ConditionAction, StepType, TOOLBOX_ITEMS, make_step,
)


# ======================================================================
# 颜色 / 字体常量
# ======================================================================

CLR_BG        = "#f5f5f5"   # 主背景（浅灰白）
CLR_PANEL     = "#ffffff"   # 面板背景（白）
CLR_CARD      = "#ebebeb"   # 步骤卡片背景
CLR_CARD_SEL  = "#d0d8ff"   # 选中卡片（淡蓝）
CLR_ACCENT    = "#3a5bd9"   # 强调色（蓝）
CLR_GREEN     = "#1a7f3c"   # 成功绿
CLR_RED       = "#cc2222"   # 错误红
CLR_YELLOW    = "#b07800"   # 警告黄
CLR_TEXT      = "#1a1a1a"   # 主文字（深黑）
CLR_SUBTEXT   = "#666666"   # 次要文字（灰）
CLR_BORDER    = "#cccccc"   # 边框

FONT_TITLE    = ("Microsoft YaHei UI", 11, "bold")
FONT_NORMAL   = ("Microsoft YaHei UI", 10)
FONT_SMALL    = ("Microsoft YaHei UI", 9)
FONT_MONO     = ("Consolas", 9)


# ======================================================================
# 工具函数
# ======================================================================

def _center_window(win: tk.Toplevel, w: int, h: int) -> None:
    """将 Toplevel 窗口居中显示。"""
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = (sw - w) // 2
    y = (sh - h) // 2
    win.geometry(f"{w}x{h}+{x}+{y}")


# ======================================================================
# ToolboxPanel  —  左侧工具箱
# ======================================================================

class ToolboxPanel(tk.Frame):
    """
    左侧工具箱面板。

    显示所有可用步骤类型，点击按钮即可将步骤追加到工作流画布。

    Args:
        parent:      父容器
        on_add_step: 回调 fn(step_type: StepType)，点击工具箱按钮时触发
    """

    def __init__(self, parent: tk.Widget, on_add_step: Callable[[StepType], None]):
        super().__init__(parent, bg=CLR_PANEL, padx=8, pady=8)
        self._on_add_step = on_add_step
        self._build()

    def _build(self) -> None:
        tk.Label(
            self, text="工 具 箱", bg=CLR_PANEL, fg=CLR_ACCENT,
            font=FONT_TITLE,
        ).pack(fill="x", pady=(0, 10))

        for item in TOOLBOX_ITEMS:
            self._make_btn(item["icon"], item["name"], item["type"])

    def _make_btn(self, icon: str, name: str, step_type: StepType) -> None:
        btn = tk.Button(
            self,
            text=f"  {icon}  {name}",
            anchor="w",
            bg=CLR_CARD,
            fg=CLR_TEXT,
            activebackground=CLR_ACCENT,
            activeforeground="#ffffff",
            relief="flat",
            font=FONT_NORMAL,
            cursor="hand2",
            padx=8,
            pady=6,
            command=lambda t=step_type: self._on_add_step(t),
        )
        btn.pack(fill="x", pady=3)

        # hover 效果
        btn.bind("<Enter>", lambda e, b=btn: b.config(bg=CLR_ACCENT))
        btn.bind("<Leave>", lambda e, b=btn: b.config(bg=CLR_CARD))


# ======================================================================
# StepCard  —  单个步骤卡片
# ======================================================================

class StepCard(tk.Frame):
    """
    工作流中单个步骤的卡片组件。

    显示步骤序号、名称、备注，提供编辑/上移/下移/删除按钮。

    Args:
        parent:      父容器（WorkflowCanvas 内部 frame）
        step:        步骤数据对象
        index:       步骤在列表中的索引（0-based）
        on_edit:     编辑回调 fn(index)
        on_move_up:  上移回调 fn(index)
        on_move_down:下移回调 fn(index)
        on_delete:   删除回调 fn(index)
        is_active:   是否为当前执行中的步骤（高亮显示）
    """

    def __init__(
        self,
        parent: tk.Widget,
        step,
        index: int,
        on_edit: Callable[[int], None],
        on_move_up: Callable[[int], None],
        on_move_down: Callable[[int], None],
        on_delete: Callable[[int], None],
        is_active: bool = False,
    ):
        bg = CLR_CARD_SEL if is_active else CLR_CARD
        super().__init__(parent, bg=bg, padx=8, pady=6, relief="flat")
        self._step    = step
        self._index   = index
        self._on_edit = on_edit
        self._on_move_up   = on_move_up
        self._on_move_down = on_move_down
        self._on_delete    = on_delete
        self._bg = bg
        self._build()

    def _build(self) -> None:
        # 左侧：序号 + 步骤名
        left = tk.Frame(self, bg=self._bg)
        left.pack(side="left", fill="both", expand=True)

        num_lbl = tk.Label(
            left,
            text=f"#{self._index + 1}",
            bg=self._bg,
            fg=CLR_ACCENT,
            font=FONT_SMALL,
            width=3,
            anchor="w",
        )
        num_lbl.pack(side="left")

        name_lbl = tk.Label(
            left,
            text=self._step.display_name(),
            bg=self._bg,
            fg=CLR_TEXT,
            font=FONT_NORMAL,
            anchor="w",
        )
        name_lbl.pack(side="left", fill="x", expand=True)

        if self._step.label:
            lbl = tk.Label(
                left,
                text=f"  ({self._step.label})",
                bg=self._bg,
                fg=CLR_SUBTEXT,
                font=FONT_SMALL,
                anchor="w",
            )
            lbl.pack(side="left")

        # 右侧：操作按钮
        right = tk.Frame(self, bg=self._bg)
        right.pack(side="right")

        for text, cmd, fg in [
            ("✏", lambda: self._on_edit(self._index),      CLR_YELLOW),
            ("↑", lambda: self._on_move_up(self._index),   CLR_TEXT),
            ("↓", lambda: self._on_move_down(self._index), CLR_TEXT),
            ("✕", lambda: self._on_delete(self._index),    CLR_RED),
        ]:
            b = tk.Button(
                right,
                text=text,
                bg=self._bg,
                fg=fg,
                activebackground=CLR_BORDER,
                activeforeground=fg,
                relief="flat",
                font=FONT_NORMAL,
                cursor="hand2",
                padx=4,
                command=cmd,
            )
            b.pack(side="left", padx=1)


# ======================================================================
# WorkflowCanvas  —  中间可滚动步骤列表
# ======================================================================

class WorkflowCanvas(tk.Frame):
    """
    工作流步骤列表（可滚动）。

    维护步骤列表，提供增删改查和重排序接口，
    并在步骤变化时自动刷新显示。

    Args:
        parent:      父容器
        on_change:   步骤列表变化时的回调 fn()
    """

    def __init__(self, parent: tk.Widget, on_change: Callable[[], None]):
        super().__init__(parent, bg=CLR_BG)
        self._steps: list = []
        self._active_index: int = -1
        self._on_change = on_change
        self._build()

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build(self) -> None:
        header = tk.Frame(self, bg=CLR_BG)
        header.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(
            header, text="工 作 流", bg=CLR_BG, fg=CLR_ACCENT,
            font=FONT_TITLE,
        ).pack(side="left")

        self._count_lbl = tk.Label(
            header, text="(0 步)", bg=CLR_BG, fg=CLR_SUBTEXT,
            font=FONT_SMALL,
        )
        self._count_lbl.pack(side="left", padx=6)

        # 清空按钮
        tk.Button(
            header,
            text="清空",
            bg=CLR_CARD,
            fg=CLR_RED,
            activebackground=CLR_RED,
            activeforeground="#fff",
            relief="flat",
            font=FONT_SMALL,
            cursor="hand2",
            padx=6,
            command=self._clear_all,
        ).pack(side="right")

        # 可滚动区域
        container = tk.Frame(self, bg=CLR_BG)
        container.pack(fill="both", expand=True, padx=8, pady=4)

        self._canvas = tk.Canvas(container, bg=CLR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=CLR_BG)
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )

        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

        # 空状态提示
        self._empty_lbl = tk.Label(
            self._inner,
            text="← 点击左侧工具箱添加步骤",
            bg=CLR_BG,
            fg=CLR_SUBTEXT,
            font=FONT_NORMAL,
        )
        self._empty_lbl.pack(pady=40)

    def _on_inner_configure(self, _event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event) -> None:
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list:
        return list(self._steps)

    def add_step(self, step) -> None:
        """追加步骤到列表末尾。"""
        self._steps.append(step)
        self._refresh()
        self._on_change()

    def set_active(self, index: int) -> None:
        """高亮显示当前执行中的步骤。"""
        self._active_index = index
        self._refresh()

    def clear_active(self) -> None:
        """清除高亮。"""
        self._active_index = -1
        self._refresh()

    # ------------------------------------------------------------------
    # 步骤操作（由 StepCard 回调触发）
    # ------------------------------------------------------------------

    def _edit_step(self, index: int) -> None:
        """
        编辑步骤。
        - ClickStep：直接进入全屏坐标拾取模式（左键记录，右键取消）
        - OcrStep：直接进入全屏框选模式
        - 其他步骤：弹出编辑对话框
        """
        step = self._steps[index]
        root = self.winfo_toplevel()

        if step.type == StepType.CLICK:
            # 隐藏主窗口，直接拾取坐标
            root.withdraw()
            root.update()
            picked = _pick_screen_point()
            root.deiconify()
            root.lift()
            root.focus_force()
            if picked is not None:
                self._steps[index] = ClickStep(
                    x=picked[0],
                    y=picked[1],
                    label=step.label,
                )
                self._refresh()
                self._on_change()
            return

        if step.type == StepType.OCR:
            # 隐藏主窗口，直接框选识别区域
            root.withdraw()
            root.update()
            region = _pick_ocr_region(root)
            root.deiconify()
            root.lift()
            root.focus_force()
            if region is not None:
                self._steps[index] = OcrStep(
                    region=region,
                    result_var=step.result_var,
                    label=step.label,
                )
                self._refresh()
                self._on_change()
            return

        edited = _open_edit_dialog(self, step, self._steps)
        if edited is not None:
            self._steps[index] = edited
            self._refresh()
            self._on_change()

    def _move_up(self, index: int) -> None:
        if index <= 0:
            return
        self._steps[index - 1], self._steps[index] = (
            self._steps[index], self._steps[index - 1]
        )
        self._refresh()
        self._on_change()

    def _move_down(self, index: int) -> None:
        if index >= len(self._steps) - 1:
            return
        self._steps[index], self._steps[index + 1] = (
            self._steps[index + 1], self._steps[index]
        )
        self._refresh()
        self._on_change()

    def _delete_step(self, index: int) -> None:
        if messagebox.askyesno("删除步骤", f"确认删除步骤 #{index + 1}？"):
            del self._steps[index]
            self._refresh()
            self._on_change()

    def _clear_all(self) -> None:
        if not self._steps:
            return
        if messagebox.askyesno("清空工作流", "确认清空所有步骤？"):
            self._steps.clear()
            self._refresh()
            self._on_change()

    # ------------------------------------------------------------------
    # 刷新显示
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """销毁所有卡片并重新渲染。"""
        for widget in self._inner.winfo_children():
            widget.destroy()

        if not self._steps:
            self._empty_lbl = tk.Label(
                self._inner,
                text="← 点击左侧工具箱添加步骤",
                bg=CLR_BG,
                fg=CLR_SUBTEXT,
                font=FONT_NORMAL,
            )
            self._empty_lbl.pack(pady=40)
            self._count_lbl.config(text="(0 步)")
            return

        self._count_lbl.config(text=f"({len(self._steps)} 步)")

        for i, step in enumerate(self._steps):
            card = StepCard(
                self._inner,
                step=step,
                index=i,
                on_edit=self._edit_step,
                on_move_up=self._move_up,
                on_move_down=self._move_down,
                on_delete=self._delete_step,
                is_active=(i == self._active_index),
            )
            card.pack(fill="x", pady=2)

            # 分隔线
            sep = tk.Frame(self._inner, bg=CLR_BORDER, height=1)
            sep.pack(fill="x", padx=4)


# ======================================================================
# LogPanel  —  底部运行日志
# ======================================================================

class LogPanel(tk.Frame):
    """
    底部运行日志面板（深色终端风格）。

    Args:
        parent: 父容器
    """

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, bg=CLR_BG)
        self._build()

    def _build(self) -> None:
        header = tk.Frame(self, bg=CLR_BG)
        header.pack(fill="x", padx=8, pady=(6, 2))

        tk.Label(
            header, text="运 行 日 志", bg=CLR_BG, fg=CLR_ACCENT,
            font=FONT_TITLE,
        ).pack(side="left")

        tk.Button(
            header,
            text="清空",
            bg=CLR_CARD,
            fg=CLR_SUBTEXT,
            activebackground=CLR_BORDER,
            relief="flat",
            font=FONT_SMALL,
            cursor="hand2",
            padx=6,
            command=self._clear,
        ).pack(side="right")

        self._text = tk.Text(
            self,
            bg="#0d0d1a",
            fg=CLR_TEXT,
            font=FONT_MONO,
            relief="flat",
            state="disabled",
            wrap="word",
            height=8,
            padx=8,
            pady=4,
        )
        self._text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 颜色标签
        self._text.tag_config("info",    foreground=CLR_TEXT)
        self._text.tag_config("success", foreground=CLR_GREEN)
        self._text.tag_config("error",   foreground=CLR_RED)
        self._text.tag_config("warn",    foreground=CLR_YELLOW)
        self._text.tag_config("accent",  foreground=CLR_ACCENT)

    def append(self, msg: str) -> None:
        """追加一行日志（线程安全，自动滚动到底部）。"""
        tag = "info"
        if "✓" in msg or "成功" in msg or "完成" in msg:
            tag = "success"
        elif "✗" in msg or "错误" in msg or "失败" in msg or "异常" in msg:
            tag = "error"
        elif "⚠" in msg or "警告" in msg:
            tag = "warn"
        elif "↩" in msg or "跳转" in msg or "■" in msg:
            tag = "accent"

        self._text.config(state="normal")
        self._text.insert("end", msg + "\n", tag)
        self._text.see("end")
        self._text.config(state="disabled")

    def _clear(self) -> None:
        self._text.config(state="normal")
        self._text.delete("1.0", "end")
        self._text.config(state="disabled")


# ======================================================================
# 全屏坐标拾取（模块级，供 ClickStep 直接使用）
# ======================================================================

def _pick_screen_point():
    """
    显示全屏半透明覆盖层，让用户点击屏幕任意位置记录坐标。
    调用方负责在调用前隐藏窗口、调用后恢复窗口。

    - 左键单击 → 记录坐标并退出
    - 右键单击 → 取消退出

    Returns:
        (x, y) 元组，或 None（取消时）
    """
    picked: list = [None]

    try:
        overlay = tk.Toplevel()
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#000000")
        overlay.attributes("-alpha", 0.35)
        overlay.config(cursor="crosshair")
        overlay.overrideredirect(True)

        tip = tk.Label(
            overlay,
            text="左键单击目标位置记录坐标    右键取消",
            bg="#000000", fg="#ffffff",
            font=("Microsoft YaHei UI", 16, "bold"),
            padx=24, pady=12,
        )
        tip.place(relx=0.5, rely=0.04, anchor="n")

        def _on_left(event):
            picked[0] = (event.x_root, event.y_root)
            overlay.destroy()

        def _on_right(_event):
            overlay.destroy()

        overlay.bind("<Button-1>", _on_left)
        overlay.bind("<Button-3>", _on_right)
        overlay.focus_force()
        overlay.grab_set()
        overlay.wait_window()

    except Exception:
        pass

    return picked[0]


def _pick_ocr_region(root: tk.Widget) -> Optional[dict]:
    """
    显示全屏截图覆盖层，让用户拖拽框选 OCR 识别区域。
    调用方负责在调用前隐藏窗口、调用后恢复窗口。

    Returns:
        mss 格式区域字典 {"top", "left", "width", "height"}，或 None（取消时）
    """
    from selector_backend import capture_fullscreen
    from selector_ui import FullscreenOverlay

    picked: list = [None]

    try:
        img, screen_left, screen_top = capture_fullscreen()

        def _on_confirmed(region, _img):
            picked[0] = region.to_mss_dict()

        overlay = FullscreenOverlay(
            parent_root=root,
            fullscreen_img=img,
            screen_left=screen_left,
            screen_top=screen_top,
            on_confirm_cb=_on_confirmed,
        )
        overlay.wait()

    except Exception as exc:
        import tkinter.messagebox as mb
        mb.showerror("错误", f"框选失败：{exc}")

    return picked[0]


# ======================================================================
# 步骤编辑对话框
# ======================================================================

def _open_edit_dialog(parent: tk.Widget, step, all_steps: list):
    """
    根据步骤类型打开对应的编辑对话框。

    Returns:
        编辑后的步骤对象，若用户取消则返回 None。
    """
    t = step.type
    if t == StepType.CLICK:
        return _edit_click(parent, step)
    elif t == StepType.OCR:
        return _edit_ocr(parent, step)
    elif t == StepType.KEYBOARD:
        return _edit_keyboard(parent, step)
    elif t == StepType.TIMER:
        return _edit_timer(parent, step)
    elif t == StepType.CONDITION:
        return _edit_condition(parent, step, all_steps)
    return None


# ------------------------------------------------------------------
# 公共对话框工具
# ------------------------------------------------------------------

def _make_dialog(parent: tk.Widget, title: str, w: int, h: int) -> tk.Toplevel:
    """创建统一风格的编辑对话框。"""
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg=CLR_PANEL)
    dlg.resizable(False, False)
    dlg.grab_set()
    _center_window(dlg, w, h)
    return dlg


def _lbl(parent, text: str) -> tk.Label:
    return tk.Label(parent, text=text, bg=CLR_PANEL, fg=CLR_TEXT, font=FONT_NORMAL, anchor="w")


def _entry(parent, textvariable=None, width=30) -> tk.Entry:
    return tk.Entry(
        parent,
        textvariable=textvariable,
        bg=CLR_CARD,
        fg=CLR_TEXT,
        insertbackground=CLR_TEXT,
        relief="flat",
        font=FONT_NORMAL,
        width=width,
    )


def _ok_cancel(parent, on_ok, on_cancel=None) -> None:
    """在 parent 底部添加确定/取消按钮行。"""
    bar = tk.Frame(parent, bg=CLR_PANEL)
    bar.pack(fill="x", padx=16, pady=(8, 16))

    tk.Button(
        bar, text="取 消",
        bg=CLR_CARD, fg=CLR_TEXT,
        activebackground=CLR_BORDER,
        relief="flat", font=FONT_NORMAL, cursor="hand2", padx=12,
        command=on_cancel or (lambda: parent.destroy()),
    ).pack(side="right", padx=(6, 0))

    tk.Button(
        bar, text="确 定",
        bg=CLR_ACCENT, fg="#ffffff",
        activebackground="#9d8fff",
        relief="flat", font=FONT_NORMAL, cursor="hand2", padx=12,
        command=on_ok,
    ).pack(side="right")


# ------------------------------------------------------------------
# ClickStep 编辑对话框
# ------------------------------------------------------------------

def _edit_click(parent: tk.Widget, step: ClickStep) -> Optional[ClickStep]:
    """编辑鼠标点击步骤：支持手动输入坐标，或点击屏幕任意位置记录坐标。"""
    result: list = [None]

    dlg = _make_dialog(parent, "编辑 — 鼠标点击", 420, 280)

    body = tk.Frame(dlg, bg=CLR_PANEL, padx=16, pady=12)
    body.pack(fill="both", expand=True)

    # 备注
    _lbl(body, "备注（可选）：").grid(row=0, column=0, sticky="w", pady=4)
    var_label = tk.StringVar(value=step.label)
    _entry(body, var_label).grid(row=0, column=1, sticky="ew", pady=4)

    # X 坐标
    _lbl(body, "X 坐标：").grid(row=1, column=0, sticky="w", pady=4)
    var_x = tk.StringVar(value=str(step.x))
    _entry(body, var_x, width=12).grid(row=1, column=1, sticky="w", pady=4)

    # Y 坐标
    _lbl(body, "Y 坐标：").grid(row=2, column=0, sticky="w", pady=4)
    var_y = tk.StringVar(value=str(step.y))
    _entry(body, var_y, width=12).grid(row=2, column=1, sticky="w", pady=4)

    body.columnconfigure(1, weight=1)

    hint = tk.Label(
        body,
        text="提示：点击下方按钮后，在屏幕任意位置单击即可记录坐标",
        bg=CLR_PANEL, fg=CLR_SUBTEXT, font=FONT_SMALL,
        wraplength=360, justify="left",
    )
    hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _pick_point():
        """隐藏对话框，调用全屏坐标拾取，完成后恢复对话框。"""
        dlg.withdraw()
        dlg.update()

        picked = _pick_screen_point()

        dlg.deiconify()
        dlg.lift()
        dlg.focus_force()

        if picked is not None:
            var_x.set(str(picked[0]))
            var_y.set(str(picked[1]))

    tk.Button(
        body,
        text="🖱  点击屏幕记录坐标",
        bg=CLR_CARD, fg=CLR_ACCENT,
        activebackground=CLR_ACCENT, activeforeground="#fff",
        relief="flat", font=FONT_NORMAL, cursor="hand2", padx=8, pady=4,
        command=_pick_point,
    ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _on_ok():
        try:
            x = int(var_x.get())
            y = int(var_y.get())
        except ValueError:
            messagebox.showerror("输入错误", "X/Y 坐标必须为整数", parent=dlg)
            return
        result[0] = ClickStep(x=x, y=y, label=var_label.get().strip())
        dlg.destroy()

    _ok_cancel(dlg, _on_ok, dlg.destroy)
    dlg.wait_window()
    return result[0]


# ------------------------------------------------------------------
# OcrStep 编辑对话框
# ------------------------------------------------------------------

def _edit_ocr(parent: tk.Widget, step: OcrStep) -> Optional[OcrStep]:
    """编辑价格识别步骤：框选识别区域。"""
    result: list = [None]

    dlg = _make_dialog(parent, "编辑 — 价格识别", 420, 280)

    body = tk.Frame(dlg, bg=CLR_PANEL, padx=16, pady=12)
    body.pack(fill="both", expand=True)

    # 备注
    _lbl(body, "备注（可选）：").grid(row=0, column=0, sticky="w", pady=4)
    var_label = tk.StringVar(value=step.label)
    _entry(body, var_label).grid(row=0, column=1, sticky="ew", pady=4)

    # 结果变量名
    _lbl(body, "结果变量名：").grid(row=1, column=0, sticky="w", pady=4)
    var_result = tk.StringVar(value=step.result_var)
    _entry(body, var_result, width=16).grid(row=1, column=1, sticky="w", pady=4)

    body.columnconfigure(1, weight=1)

    # 当前区域显示
    region_var = [dict(step.region)]

    def _region_text():
        r = region_var[0]
        if r:
            return f"left={r.get('left',0)}, top={r.get('top',0)}, w={r.get('width',0)}, h={r.get('height',0)}"
        return "（未设置）"

    region_lbl = tk.Label(
        body, text=_region_text(),
        bg=CLR_CARD, fg=CLR_TEXT, font=FONT_SMALL,
        padx=6, pady=4, anchor="w",
    )
    region_lbl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 4))

    def _pick_region():
        """隐藏对话框，调用全屏框选，完成后恢复对话框。"""
        dlg.withdraw()
        dlg.update()
        region = _pick_ocr_region(dlg)
        dlg.deiconify()
        dlg.lift()
        dlg.focus_force()
        if region is not None:
            region_var[0] = region
            region_lbl.config(text=_region_text())

    tk.Button(
        body,
        text="🔍  框选识别区域",
        bg=CLR_CARD, fg=CLR_ACCENT,
        activebackground=CLR_ACCENT, activeforeground="#fff",
        relief="flat", font=FONT_NORMAL, cursor="hand2", padx=8, pady=4,
        command=_pick_region,
    ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)

    def _on_ok():
        var_name = var_result.get().strip()
        if not var_name:
            messagebox.showerror("输入错误", "结果变量名不能为空", parent=dlg)
            return
        result[0] = OcrStep(
            region=region_var[0],
            result_var=var_name,
            label=var_label.get().strip(),
        )
        dlg.destroy()

    _ok_cancel(dlg, _on_ok, dlg.destroy)
    dlg.wait_window()
    return result[0]


# ------------------------------------------------------------------
# KeyboardStep 编辑对话框
# ------------------------------------------------------------------

def _edit_keyboard(parent: tk.Widget, step: KeyboardStep) -> Optional[KeyboardStep]:
    """编辑键盘按键组合步骤。"""
    result: list = [None]

    dlg = _make_dialog(parent, "编辑 — 按键组合", 420, 260)

    body = tk.Frame(dlg, bg=CLR_PANEL, padx=16, pady=12)
    body.pack(fill="both", expand=True)

    _lbl(body, "备注（可选）：").grid(row=0, column=0, sticky="w", pady=4)
    var_label = tk.StringVar(value=step.label)
    _entry(body, var_label).grid(row=0, column=1, sticky="ew", pady=4)

    _lbl(body, "按键组合：").grid(row=1, column=0, sticky="w", pady=4)
    var_keys = tk.StringVar(value=step.keys)
    _entry(body, var_keys).grid(row=1, column=1, sticky="ew", pady=4)

    body.columnconfigure(1, weight=1)

    hint = tk.Label(
        body,
        text="示例：enter  /  ctrl+c  /  ctrl+shift+s  /  esc",
        bg=CLR_PANEL, fg=CLR_SUBTEXT, font=FONT_SMALL,
    )
    hint.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _on_ok():
        keys = var_keys.get().strip()
        if not keys:
            messagebox.showerror("输入错误", "按键组合不能为空", parent=dlg)
            return
        result[0] = KeyboardStep(keys=keys, label=var_label.get().strip())
        dlg.destroy()

    _ok_cancel(dlg, _on_ok, dlg.destroy)
    dlg.wait_window()
    return result[0]


# ------------------------------------------------------------------
# TimerStep 编辑对话框
# ------------------------------------------------------------------

def _edit_timer(parent: tk.Widget, step: TimerStep) -> Optional[TimerStep]:
    """编辑倒计时步骤。"""
    result: list = [None]

    dlg = _make_dialog(parent, "编辑 — 倒计时", 380, 220)

    body = tk.Frame(dlg, bg=CLR_PANEL, padx=16, pady=12)
    body.pack(fill="both", expand=True)

    _lbl(body, "备注（可选）：").grid(row=0, column=0, sticky="w", pady=4)
    var_label = tk.StringVar(value=step.label)
    _entry(body, var_label).grid(row=0, column=1, sticky="ew", pady=4)

    _lbl(body, "等待秒数：").grid(row=1, column=0, sticky="w", pady=4)
    var_sec = tk.StringVar(value=str(step.seconds))
    _entry(body, var_sec, width=10).grid(row=1, column=1, sticky="w", pady=4)

    body.columnconfigure(1, weight=1)

    tk.Label(
        body, text="支持小数，如 0.5 表示 500ms",
        bg=CLR_PANEL, fg=CLR_SUBTEXT, font=FONT_SMALL,
    ).grid(row=2, column=0, columnspan=2, sticky="w")

    def _on_ok():
        try:
            sec = float(var_sec.get())
            if sec < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "等待秒数必须为正数", parent=dlg)
            return
        result[0] = TimerStep(seconds=sec, label=var_label.get().strip())
        dlg.destroy()

    _ok_cancel(dlg, _on_ok, dlg.destroy)
    dlg.wait_window()
    return result[0]


# ------------------------------------------------------------------
# ConditionStep 编辑对话框
# ------------------------------------------------------------------

def _edit_condition(
    parent: tk.Widget,
    step: ConditionStep,
    all_steps: list,
) -> Optional[ConditionStep]:
    """编辑条件判断步骤。"""
    result: list = [None]

    dlg = _make_dialog(parent, "编辑 — 价格判断", 480, 420)

    body = tk.Frame(dlg, bg=CLR_PANEL, padx=16, pady=12)
    body.pack(fill="both", expand=True)

    # 备注
    _lbl(body, "备注（可选）：").grid(row=0, column=0, sticky="w", pady=4)
    var_label = tk.StringVar(value=step.label)
    _entry(body, var_label).grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)

    # 条件表达式：变量名 运算符 数值
    _lbl(body, "条件：").grid(row=1, column=0, sticky="w", pady=4)

    var_var = tk.StringVar(value=step.var)
    _entry(body, var_var, width=10).grid(row=1, column=1, sticky="w", pady=4, padx=(0, 4))

    var_op = tk.StringVar(value=step.op)
    op_cb = ttk.Combobox(
        body, textvariable=var_op,
        values=["<", ">", "==", "<=", ">=", "!="],
        width=5, state="readonly",
    )
    op_cb.grid(row=1, column=2, sticky="w", pady=4, padx=4)

    var_val = tk.StringVar(value=str(step.value))
    _entry(body, var_val, width=10).grid(row=1, column=3, sticky="w", pady=4)

    body.columnconfigure(1, weight=1)

    # 条件成立时
    _lbl(body, "成立时：").grid(row=2, column=0, sticky="w", pady=4)
    var_true = tk.StringVar(value=step.on_true)
    true_cb = ttk.Combobox(
        body, textvariable=var_true,
        values=[ConditionAction.CONTINUE, ConditionAction.STOP, ConditionAction.LOOP],
        width=12, state="readonly",
    )
    true_cb.grid(row=2, column=1, columnspan=3, sticky="w", pady=4)

    # 条件不成立时
    _lbl(body, "不成立时：").grid(row=3, column=0, sticky="w", pady=4)
    var_false = tk.StringVar(value=step.on_false)
    false_cb = ttk.Combobox(
        body, textvariable=var_false,
        values=[ConditionAction.CONTINUE, ConditionAction.STOP, ConditionAction.LOOP],
        width=12, state="readonly",
    )
    false_cb.grid(row=3, column=1, columnspan=3, sticky="w", pady=4)

    # 循环跳回步骤
    _lbl(body, "循环跳回步骤：").grid(row=4, column=0, sticky="w", pady=4)

    step_labels = [f"步骤 {i+1}: {s.display_name()}" for i, s in enumerate(all_steps)]
    var_loop = tk.StringVar()
    loop_cb = ttk.Combobox(
        body, textvariable=var_loop,
        values=step_labels,
        width=30, state="readonly",
    )
    # 设置当前值
    if 0 <= step.loop_to < len(step_labels):
        loop_cb.current(step.loop_to)
    elif step_labels:
        loop_cb.current(0)
    loop_cb.grid(row=4, column=1, columnspan=3, sticky="w", pady=4)

    hint = tk.Label(
        body,
        text="（仅当成立/不成立动作为 loop 时生效）",
        bg=CLR_PANEL, fg=CLR_SUBTEXT, font=FONT_SMALL,
    )
    hint.grid(row=5, column=0, columnspan=4, sticky="w")

    # 动作说明
    action_hint = tk.Label(
        body,
        text="continue=继续下一步  stop=停止工作流  loop=跳回指定步骤",
        bg=CLR_PANEL, fg=CLR_SUBTEXT, font=FONT_SMALL,
    )
    action_hint.grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))

    def _on_ok():
        var_name = var_var.get().strip()
        if not var_name:
            messagebox.showerror("输入错误", "变量名不能为空", parent=dlg)
            return
        try:
            threshold = float(var_val.get())
        except ValueError:
            messagebox.showerror("输入错误", "比较值必须为数字", parent=dlg)
            return

        loop_idx = loop_cb.current()
        if loop_idx < 0:
            loop_idx = 0

        result[0] = ConditionStep(
            var=var_name,
            op=var_op.get(),
            value=threshold,
            on_true=var_true.get(),
            on_false=var_false.get(),
            loop_to=loop_idx,
            label=var_label.get().strip(),
        )
        dlg.destroy()

    _ok_cancel(dlg, _on_ok, dlg.destroy)
    dlg.wait_window()
    return result[0]