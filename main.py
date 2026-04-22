# -*- coding: utf-8 -*-
"""
main.py  -  工作流自动化工具主入口

WorkflowApp 主窗口布局：
  ┌─────────────────────────────────────────────────────┐
  │  工具箱 (左)  │  工作流画布 (中)  │  控制区 (右)    │
  ├───────────────┴──────────────────┴──────────────────┤
  │                   运行日志 (底部)                    │
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox

from selector_backend import OcrBackend
from driver import MouseDriver
from workflow_steps import StepType, make_step
from workflow_engine import WorkflowEngine
from workflow_ui import (
    ToolboxPanel,
    WorkflowCanvas,
    LogPanel,
    CLR_BG, CLR_PANEL, CLR_CARD, CLR_ACCENT,
    CLR_GREEN, CLR_RED, CLR_TEXT, CLR_SUBTEXT, CLR_BORDER,
    FONT_TITLE, FONT_NORMAL, FONT_SMALL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("WorkflowApp")


# ======================================================================
# WorkflowApp  —  主窗口
# ======================================================================

class WorkflowApp:
    """
    工作流自动化工具主窗口。

    组装 ToolboxPanel + WorkflowCanvas + LogPanel，
    并管理 WorkflowEngine 的生命周期。
    """

    def __init__(self):
        self._engine: WorkflowEngine | None = None

        # 初始化后端（懒加载，首次执行时才真正加载模型）
        self._ocr_backend: OcrBackend | None = None
        self._mouse_driver: MouseDriver | None = None

        self._root = tk.Tk()
        self._root.title("工作流自动化工具")
        self._root.configure(bg=CLR_BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._center(1100, 720)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """构建主界面布局。"""

        # ── 顶部标题栏 ──────────────────────────────────────────────
        title_bar = tk.Frame(self._root, bg=CLR_PANEL, pady=8)
        title_bar.pack(fill="x")

        tk.Label(
            title_bar,
            text="⚡  工作流自动化工具",
            bg=CLR_PANEL, fg=CLR_ACCENT,
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left", padx=16)

        tk.Label(
            title_bar,
            text="设计工作流 → 点击执行",
            bg=CLR_PANEL, fg=CLR_SUBTEXT,
            font=FONT_SMALL,
        ).pack(side="left", padx=4)

        # ── 主体区域（三列）────────────────────────────────────────
        body = tk.Frame(self._root, bg=CLR_BG)
        body.pack(fill="both", expand=True)

        # 左：工具箱
        self._toolbox = ToolboxPanel(body, on_add_step=self._on_add_step)
        self._toolbox.pack(side="left", fill="y", padx=(8, 4), pady=8)

        # 分隔线
        tk.Frame(body, bg=CLR_BORDER, width=1).pack(side="left", fill="y", pady=8)

        # 中：工作流画布
        self._canvas = WorkflowCanvas(body, on_change=self._on_workflow_change)
        self._canvas.pack(side="left", fill="both", expand=True, padx=4, pady=8)

        # 分隔线
        tk.Frame(body, bg=CLR_BORDER, width=1).pack(side="left", fill="y", pady=8)

        # 右：控制面板
        self._ctrl_panel = self._build_ctrl_panel(body)
        self._ctrl_panel.pack(side="left", fill="y", padx=(4, 8), pady=8)

        # ── 底部日志区 ──────────────────────────────────────────────
        tk.Frame(self._root, bg=CLR_BORDER, height=1).pack(fill="x")
        self._log_panel = LogPanel(self._root)
        self._log_panel.pack(fill="x")

    def _build_ctrl_panel(self, parent: tk.Widget) -> tk.Frame:
        """构建右侧控制面板（执行/停止按钮 + 状态显示）。"""
        panel = tk.Frame(parent, bg=CLR_PANEL, padx=12, pady=12, width=160)
        panel.pack_propagate(False)

        tk.Label(
            panel, text="控 制", bg=CLR_PANEL, fg=CLR_ACCENT,
            font=FONT_TITLE,
        ).pack(fill="x", pady=(0, 12))

        # 执行按钮
        self._btn_run = tk.Button(
            panel,
            text="▶  执 行",
            bg=CLR_GREEN,
            fg="#ffffff",
            activebackground="#155c2e",
            activeforeground="#ffffff",
            relief="flat",
            font=("Microsoft YaHei UI", 11, "bold"),
            cursor="hand2",
            pady=8,
            command=self._on_run,
        )
        self._btn_run.pack(fill="x", pady=4)

        # 停止按钮
        self._btn_stop = tk.Button(
            panel,
            text="■  停 止",
            bg=CLR_CARD,
            fg=CLR_RED,
            activebackground=CLR_RED,
            activeforeground="#fff",
            relief="flat",
            font=("Microsoft YaHei UI", 11, "bold"),
            cursor="hand2",
            pady=8,
            state="disabled",
            command=self._on_stop,
        )
        self._btn_stop.pack(fill="x", pady=4)

        # 分隔线
        tk.Frame(panel, bg=CLR_BORDER, height=1).pack(fill="x", pady=12)

        # 状态标签
        tk.Label(
            panel, text="状态", bg=CLR_PANEL, fg=CLR_SUBTEXT,
            font=FONT_SMALL,
        ).pack(anchor="w")

        self._status_lbl = tk.Label(
            panel,
            text="就绪",
            bg=CLR_PANEL,
            fg=CLR_TEXT,
            font=FONT_NORMAL,
            anchor="w",
            wraplength=140,
            justify="left",
        )
        self._status_lbl.pack(fill="x", pady=(2, 12))

        # 步骤计数
        tk.Label(
            panel, text="步骤数", bg=CLR_PANEL, fg=CLR_SUBTEXT,
            font=FONT_SMALL,
        ).pack(anchor="w")

        self._step_count_lbl = tk.Label(
            panel, text="0",
            bg=CLR_PANEL, fg=CLR_TEXT,
            font=FONT_NORMAL, anchor="w",
        )
        self._step_count_lbl.pack(fill="x", pady=(2, 0))

        return panel

    # ------------------------------------------------------------------
    # 工具箱回调
    # ------------------------------------------------------------------

    def _on_add_step(self, step_type: StepType) -> None:
        """工具箱点击 → 追加默认步骤到画布。"""
        step = make_step(step_type)
        self._canvas.add_step(step)
        self._log(f"已添加步骤：{step.display_name()}")

    # ------------------------------------------------------------------
    # 工作流变化回调
    # ------------------------------------------------------------------

    def _on_workflow_change(self) -> None:
        """步骤列表变化时更新步骤计数。"""
        count = len(self._canvas.steps)
        self._step_count_lbl.config(text=str(count))

    # ------------------------------------------------------------------
    # 执行 / 停止
    # ------------------------------------------------------------------

    def _on_run(self) -> None:
        """点击执行按钮。"""
        steps = self._canvas.steps
        if not steps:
            messagebox.showwarning("工作流为空", "请先在左侧工具箱添加步骤", parent=self._root)
            return

        # 检查是否有未配置的步骤
        unconfigured = [
            f"步骤 {i+1}: {s.display_name()}"
            for i, s in enumerate(steps)
            if not s.is_configured()
        ]
        if unconfigured:
            msg = "以下步骤尚未配置，请先编辑：\n" + "\n".join(unconfigured)
            messagebox.showwarning("步骤未配置", msg, parent=self._root)
            return

        # 懒加载后端（使用本地 OCR 服务，避免每次重新加载模型）
        if self._ocr_backend is None:
            self._log("正在连接 OCR 服务 (http://127.0.0.1:5000)...")
            try:
                self._ocr_backend = OcrBackend(server_url="http://127.0.0.1:5000")
            except Exception as exc:
                self._log(f"✗ OCR 后端初始化失败：{exc}")
                messagebox.showerror("初始化失败", f"OCR 后端初始化失败：{exc}", parent=self._root)
                return

        if self._mouse_driver is None:
            self._mouse_driver = MouseDriver()

        # 创建并启动引擎
        self._engine = WorkflowEngine(
            steps=steps,
            ocr_backend=self._ocr_backend,
            mouse_driver=self._mouse_driver,
            on_log=self._on_engine_log,
            on_step=self._on_engine_step,
            on_done=self._on_engine_done,
        )

        self._set_running(True)
        self._canvas.clear_active()
        self._engine.start()

    def _on_stop(self) -> None:
        """点击停止按钮。"""
        if self._engine and self._engine.is_running:
            self._engine.stop()

    # ------------------------------------------------------------------
    # 引擎回调（后台线程调用，需通过 after 切回主线程）
    # ------------------------------------------------------------------

    def _on_engine_log(self, msg: str) -> None:
        self._root.after(0, self._log, msg)

    def _on_engine_step(self, index: int) -> None:
        self._root.after(0, self._canvas.set_active, index)
        self._root.after(0, self._status_lbl.config, {"text": f"执行步骤 {index + 1}"})

    def _on_engine_done(self, reason: str) -> None:
        self._root.after(0, self._handle_done, reason)

    def _handle_done(self, reason: str) -> None:
        self._canvas.clear_active()
        self._set_running(False)

        reason_map = {
            "done":      ("✓ 工作流执行完成", CLR_GREEN),
            "stopped":   ("■ 工作流已停止",   CLR_TEXT),
            "cancelled": ("■ 已手动取消",     CLR_TEXT),
            "error":     ("✗ 执行出错",       CLR_RED),
        }
        text, color = reason_map.get(reason, (f"结束: {reason}", CLR_TEXT))
        self._status_lbl.config(text=text, fg=color)
        self._log(text)

    # ------------------------------------------------------------------
    # UI 状态切换
    # ------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        if running:
            self._btn_run.config(state="disabled", bg=CLR_CARD, fg=CLR_SUBTEXT)
            self._btn_stop.config(state="normal")
            self._status_lbl.config(text="运行中...", fg=CLR_GREEN)
        else:
            self._btn_run.config(state="normal", bg=CLR_GREEN, fg="#ffffff")
            self._btn_stop.config(state="disabled")

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self._log_panel.append(msg)
        logger.info(msg)

    # ------------------------------------------------------------------
    # 窗口管理
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._engine and self._engine.is_running:
            if not messagebox.askyesno("确认退出", "工作流正在运行，确认退出？", parent=self._root):
                return
            self._engine.stop()
        if self._mouse_driver:
            self._mouse_driver.close()
        self._root.destroy()

    def _center(self, w: int, h: int) -> None:
        self._root.update_idletasks()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self._root.geometry(f"{w}x{h}+{x}+{y}")

    def run(self) -> None:
        self._root.mainloop()


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    app = WorkflowApp()
    app.run()
