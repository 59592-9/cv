# -*- coding: utf-8 -*-
"""
工作流执行引擎 (workflow_engine.py)

负责按顺序执行工作流中的步骤，支持：
  - 顺序执行
  - 条件判断（继续 / 停止 / 循环跳转）
  - 后台线程运行（不阻塞 GUI）
  - 线程安全停止

执行上下文（context）是一个共享字典，步骤间通过变量名传递数据：
    {"price": 5639.0, ...}
"""

import logging
import re
import threading
import time
from typing import Callable, List, Optional

from workflow_steps import (
    ClickStep, OcrStep, KeyboardStep, TimerStep, ConditionStep,
    ConditionAction, StepType,
)

logger = logging.getLogger("WorkflowEngine")


# ======================================================================
# 执行结果
# ======================================================================

class StepResult:
    """单步执行结果。"""
    NEXT    = "next"    # 继续执行下一步
    STOP    = "stop"    # 停止工作流
    JUMP    = "jump"    # 跳转到指定步骤
    ERROR   = "error"   # 执行出错

    def __init__(self, action: str, jump_to: int = 0, message: str = ""):
        self.action  = action
        self.jump_to = jump_to   # 仅 JUMP 时有效
        self.message = message


# ======================================================================
# 工作流引擎
# ======================================================================

class WorkflowEngine:
    """
    工作流执行引擎。

    在后台线程中顺序执行步骤列表，通过回调向 GUI 汇报进度。

    Args:
        steps:        步骤列表（ClickStep / OcrStep / ...）
        ocr_backend:  OcrBackend 实例（用于 OcrStep）
        mouse_driver: MouseDriver 实例（用于 ClickStep）
        on_log:       日志回调 fn(msg: str)
        on_step:      步骤开始回调 fn(index: int)
        on_done:      工作流结束回调 fn(reason: str)
    """

    def __init__(
        self,
        steps: list,
        ocr_backend,
        mouse_driver,
        on_log: Callable[[str], None],
        on_step: Callable[[int], None],
        on_done: Callable[[str], None],
    ):
        self._steps        = steps
        self._ocr_backend  = ocr_backend
        self._mouse_driver = mouse_driver
        self._on_log       = on_log
        self._on_step      = on_step
        self._on_done      = on_done

        self._stop_event   = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def start(self) -> None:
        """在后台线程启动工作流。"""
        if self._thread and self._thread.is_alive():
            self._log("工作流已在运行中")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """请求停止工作流（当前步骤执行完后生效）。"""
        self._stop_event.set()
        self._log("收到停止信号...")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # 执行主循环
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """后台线程：顺序执行步骤，支持循环跳转。"""
        context: dict = {}
        i = 0
        total = len(self._steps)

        self._log(f"工作流开始执行，共 {total} 个步骤")

        while i < total:
            # 每步开始前检查停止信号
            if self._stop_event.is_set():
                break

            step = self._steps[i]
            self._on_step(i)
            self._log(f"[步骤 {i+1}/{total}] {step.display_name()}")

            try:
                result = self._execute_step(step, context, i)
            except Exception as exc:
                self._log(f"  ✗ 步骤执行异常: {exc}")
                logger.exception("步骤 %d 执行异常", i)
                result = StepResult(StepResult.ERROR, message=str(exc))

            # 每步结束后再次检查停止信号（OCR 等阻塞操作完成后立即响应）
            if self._stop_event.is_set():
                self._log("工作流已手动停止")
                self._on_done("cancelled")
                return

            if result.action == StepResult.NEXT:
                i += 1
            elif result.action == StepResult.JUMP:
                self._log(f"  ↩ 跳转到步骤 {result.jump_to + 1}")
                i = result.jump_to
            elif result.action == StepResult.STOP:
                self._log(f"  ■ 工作流停止：{result.message}")
                self._on_done("stopped")
                return
            elif result.action == StepResult.ERROR:
                self._log(f"  ✗ 工作流因错误停止")
                self._on_done("error")
                return

        if self._stop_event.is_set():
            self._log("工作流已手动停止")
            self._on_done("cancelled")
        else:
            self._log("✓ 工作流执行完成")
            self._on_done("done")

    # ------------------------------------------------------------------
    # 单步执行分发
    # ------------------------------------------------------------------

    def _execute_step(self, step, context: dict, index: int) -> StepResult:
        """根据步骤类型分发到对应执行方法。"""
        t = step.type
        if t == StepType.CLICK:
            return self._exec_click(step)
        elif t == StepType.OCR:
            return self._exec_ocr(step, context)
        elif t == StepType.KEYBOARD:
            return self._exec_keyboard(step)
        elif t == StepType.TIMER:
            return self._exec_timer(step)
        elif t == StepType.CONDITION:
            return self._exec_condition(step, context)
        else:
            return StepResult(StepResult.ERROR, message=f"未知步骤类型: {t}")

    # ------------------------------------------------------------------
    # 各步骤执行实现
    # ------------------------------------------------------------------

    def _exec_click(self, step: ClickStep) -> StepResult:
        """执行鼠标点击。"""
        if not step.is_configured():
            return StepResult(StepResult.ERROR, message="点击坐标未设置")
        self._log(f"  → 点击坐标 ({step.x}, {step.y})")
        ok = self._mouse_driver.move_and_click(step.x, step.y)
        if ok:
            self._log(f"  ✓ 点击成功")
            return StepResult(StepResult.NEXT)
        else:
            return StepResult(StepResult.ERROR, message="鼠标点击失败")

    def _exec_ocr(self, step: OcrStep, context: dict) -> StepResult:
        """执行价格识别，结果存入 context。"""
        if not step.is_configured():
            return StepResult(StepResult.ERROR, message="识别区域未设置")

        import numpy as np
        import cv2
        import mss

        try:
            with mss.mss() as sct:
                screenshot = sct.grab(step.region)
                img = np.array(screenshot)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            ocr_result = self._ocr_backend.recognize(img_bgr)
            raw_text = ocr_result.raw_text
            price = _extract_price(raw_text)

            context[step.result_var] = price
            self._log(f"  → 识别结果: '{raw_text}' → {price}")

            if price < 0:
                self._log(f"  ⚠ 未识别到有效价格，{step.result_var} = {price}")
            else:
                self._log(f"  ✓ {step.result_var} = {price}")

            return StepResult(StepResult.NEXT)

        except Exception as exc:
            return StepResult(StepResult.ERROR, message=f"OCR 识别失败: {exc}")

    def _exec_keyboard(self, step: KeyboardStep) -> StepResult:
        """执行键盘按键组合。"""
        if not step.is_configured():
            return StepResult(StepResult.ERROR, message="按键未设置")

        try:
            from pynput import keyboard as kb
            _press_keys(step.keys)
            self._log(f"  ✓ 按键 [{step.keys}] 执行完成")
            return StepResult(StepResult.NEXT)
        except Exception as exc:
            return StepResult(StepResult.ERROR, message=f"按键执行失败: {exc}")

    def _exec_timer(self, step: TimerStep) -> StepResult:
        """执行倒计时等待。"""
        if step.seconds <= 0:
            return StepResult(StepResult.NEXT)

        total = step.seconds
        self._log(f"  → 倒计时 {total}s...")

        # 分段等待，每 0.1s 检查一次停止信号
        elapsed = 0.0
        interval = 0.1
        while elapsed < total and not self._stop_event.is_set():
            time.sleep(min(interval, total - elapsed))
            elapsed += interval
            remaining = max(0.0, total - elapsed)
            if remaining > 0 and int(elapsed) != int(elapsed - interval):
                self._log(f"  ⏱ 剩余 {remaining:.1f}s")

        if self._stop_event.is_set():
            return StepResult(StepResult.STOP, message="倒计时被中断")

        self._log(f"  ✓ 倒计时结束")
        return StepResult(StepResult.NEXT)

    def _exec_condition(self, step: ConditionStep, context: dict) -> StepResult:
        """执行条件判断。"""
        result = step.evaluate(context)
        val = context.get(step.var, "未定义")
        self._log(
            f"  → 判断: {step.var}({val}) {step.op} {step.value} → {'成立' if result else '不成立'}"
        )

        action = step.on_true if result else step.on_false

        if action == ConditionAction.CONTINUE:
            self._log(f"  ✓ 条件{'成立' if result else '不成立'}，继续执行")
            return StepResult(StepResult.NEXT)
        elif action == ConditionAction.STOP:
            self._log(f"  ■ 条件{'成立' if result else '不成立'}，停止工作流")
            return StepResult(StepResult.STOP, message="条件触发停止")
        elif action == ConditionAction.LOOP:
            self._log(f"  ↩ 条件{'成立' if result else '不成立'}，跳回步骤 {step.loop_to + 1}")
            return StepResult(StepResult.JUMP, jump_to=step.loop_to)
        else:
            return StepResult(StepResult.ERROR, message=f"未知动作: {action}")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """线程安全地发送日志。"""
        logger.info(msg)
        if self._on_log:
            self._on_log(msg)


# ======================================================================
# 工具函数
# ======================================================================

def _extract_price(text: str) -> float:
    """从 OCR 文本提取价格（支持小数点和千位逗号）。"""
    if not text:
        return -1.0
    m = re.search(r"[\d,]+\.\d+", text)
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            pass
    digits = re.sub(r"[^\d]", "", text)
    if digits:
        try:
            return float(digits)
        except ValueError:
            pass
    return -1.0


def _press_keys(keys_str: str) -> None:
    """
    模拟键盘按键组合。

    支持格式：
        "enter"         → 单键
        "ctrl+c"        → 组合键
        "ctrl+shift+s"  → 多键组合

    Args:
        keys_str: 按键字符串（不区分大小写）
    """
    from pynput import keyboard as kb
    from pynput.keyboard import Key, Controller

    controller = Controller()

    # 特殊键名映射
    SPECIAL_KEYS = {
        "enter":     Key.enter,
        "esc":       Key.esc,
        "escape":    Key.esc,
        "tab":       Key.tab,
        "space":     Key.space,
        "backspace": Key.backspace,
        "delete":    Key.delete,
        "up":        Key.up,
        "down":      Key.down,
        "left":      Key.left,
        "right":     Key.right,
        "home":      Key.home,
        "end":       Key.end,
        "pageup":    Key.page_up,
        "pagedown":  Key.page_down,
        "f1":  Key.f1,  "f2":  Key.f2,  "f3":  Key.f3,  "f4":  Key.f4,
        "f5":  Key.f5,  "f6":  Key.f6,  "f7":  Key.f7,  "f8":  Key.f8,
        "f9":  Key.f9,  "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
        "ctrl":  Key.ctrl,  "control": Key.ctrl,
        "shift": Key.shift,
        "alt":   Key.alt,
        "win":   Key.cmd,
    }

    parts = [p.strip().lower() for p in keys_str.split("+")]
    resolved = []
    for p in parts:
        if p in SPECIAL_KEYS:
            resolved.append(SPECIAL_KEYS[p])
        elif len(p) == 1:
            resolved.append(p)
        else:
            raise ValueError(f"未知按键: '{p}'")

    # 按下所有键，再全部释放
    for k in resolved:
        controller.press(k)
    for k in reversed(resolved):
        controller.release(k)
