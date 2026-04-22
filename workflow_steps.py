# -*- coding: utf-8 -*-
"""
工作流步骤数据模型 (workflow_steps.py)

定义所有可用步骤的数据结构，每个步骤是一个 dataclass，
可序列化为 dict 方便保存/加载工作流配置。

步骤类型：
    CLICK      - 鼠标左键点击指定坐标
    OCR        - 框选区域识别价格，结果存入上下文变量
    KEYBOARD   - 模拟键盘按键组合
    TIMER      - 倒计时等待
    CONDITION  - 条件判断（基于上下文变量），支持继续/停止/循环跳转
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ======================================================================
# 步骤类型枚举
# ======================================================================

class StepType(str, Enum):
    CLICK     = "click"      # 鼠标点击
    OCR       = "ocr"        # 价格识别
    KEYBOARD  = "keyboard"   # 按键组合
    TIMER     = "timer"      # 倒计时
    CONDITION = "condition"  # 条件判断


# ======================================================================
# 条件判断结果动作
# ======================================================================

class ConditionAction(str, Enum):
    CONTINUE = "continue"  # 继续执行下一步
    STOP     = "stop"      # 停止整个工作流
    LOOP     = "loop"      # 跳回指定步骤（循环）


# ======================================================================
# 步骤数据类
# ======================================================================

@dataclass
class ClickStep:
    """
    鼠标左键点击步骤。

    Attributes:
        x:     点击的屏幕 X 坐标
        y:     点击的屏幕 Y 坐标
        label: 用户自定义备注（显示在步骤卡片上）
    """
    type: str = StepType.CLICK
    x: int = 0
    y: int = 0
    label: str = ""

    def display_name(self) -> str:
        return f"🖱 点击 ({self.x}, {self.y})" if self.x or self.y else "🖱 点击（未设置坐标）"

    def is_configured(self) -> bool:
        return self.x > 0 or self.y > 0


@dataclass
class OcrStep:
    """
    价格识别步骤。

    Attributes:
        region:     mss 格式区域 {"top", "left", "width", "height"}
        result_var: 识别结果存入的上下文变量名（供 ConditionStep 引用）
        label:      用户自定义备注
    """
    type: str = StepType.OCR
    region: dict = field(default_factory=dict)
    result_var: str = "price"
    label: str = ""

    def display_name(self) -> str:
        if self.region:
            r = self.region
            return f"🔍 识别价格 ({r.get('left',0)},{r.get('top',0)} {r.get('width',0)}×{r.get('height',0)})"
        return "🔍 识别价格（未设置区域）"

    def is_configured(self) -> bool:
        return bool(self.region)


@dataclass
class KeyboardStep:
    """
    键盘按键组合步骤。

    Attributes:
        keys:  按键字符串，如 "ctrl+c"、"enter"、"esc"、"ctrl+shift+s"
        label: 用户自定义备注
    """
    type: str = StepType.KEYBOARD
    keys: str = ""
    label: str = ""

    def display_name(self) -> str:
        return f"⌨ 按键 [{self.keys}]" if self.keys else "⌨ 按键（未设置）"

    def is_configured(self) -> bool:
        return bool(self.keys.strip())


@dataclass
class TimerStep:
    """
    倒计时等待步骤。

    Attributes:
        seconds: 等待秒数（支持小数，如 0.5）
        label:   用户自定义备注
    """
    type: str = StepType.TIMER
    seconds: float = 1.0
    label: str = ""

    def display_name(self) -> str:
        return f"⏱ 等待 {self.seconds}s"

    def is_configured(self) -> bool:
        return self.seconds > 0


@dataclass
class ConditionStep:
    """
    条件判断步骤。

    根据上下文变量的值决定工作流走向：
        - on_true:  条件成立时的动作
        - on_false: 条件不成立时的动作
        - loop_to:  动作为 LOOP 时跳回的步骤索引（0-based）

    Attributes:
        var:      引用的上下文变量名（如 "price"）
        op:       比较运算符："<" ">" "==" "<=" ">=" "!="
        value:    比较值（浮点数）
        on_true:  条件成立时的动作（ConditionAction）
        on_false: 条件不成立时的动作（ConditionAction）
        loop_to:  循环跳回的步骤索引（on_true/on_false 为 LOOP 时有效）
        label:    用户自定义备注
    """
    type: str = StepType.CONDITION
    var: str = "price"
    op: str = "<"
    value: float = 0.0
    on_true: str = ConditionAction.CONTINUE
    on_false: str = ConditionAction.LOOP
    loop_to: int = 0
    label: str = ""

    def display_name(self) -> str:
        return f"❓ 若 {self.var} {self.op} {self.value}"

    def is_configured(self) -> bool:
        return bool(self.var.strip()) and bool(self.op)

    def evaluate(self, context: dict) -> bool:
        """
        对上下文变量求值。

        Args:
            context: 运行时变量字典，如 {"price": 5639.0}

        Returns:
            条件是否成立
        """
        val = context.get(self.var, None)
        if val is None:
            return False
        try:
            val = float(val)
            threshold = float(self.value)
            ops = {
                "<":  val < threshold,
                ">":  val > threshold,
                "==": val == threshold,
                "<=": val <= threshold,
                ">=": val >= threshold,
                "!=": val != threshold,
            }
            return ops.get(self.op, False)
        except (TypeError, ValueError):
            return False


# ======================================================================
# 步骤工厂
# ======================================================================

# 步骤类型 → 数据类的映射
STEP_CLASS_MAP = {
    StepType.CLICK:     ClickStep,
    StepType.OCR:       OcrStep,
    StepType.KEYBOARD:  KeyboardStep,
    StepType.TIMER:     TimerStep,
    StepType.CONDITION: ConditionStep,
}

# 工具箱展示顺序和元信息
TOOLBOX_ITEMS = [
    {"type": StepType.CLICK,     "icon": "🖱",  "name": "鼠标点击"},
    {"type": StepType.OCR,       "icon": "🔍",  "name": "价格识别"},
    {"type": StepType.KEYBOARD,  "icon": "⌨",  "name": "按键组合"},
    {"type": StepType.TIMER,     "icon": "⏱",  "name": "倒计时"},
    {"type": StepType.CONDITION, "icon": "❓",  "name": "价格判断"},
]


def make_step(step_type: StepType):
    """根据步骤类型创建默认步骤实例。"""
    cls = STEP_CLASS_MAP.get(step_type)
    if cls is None:
        raise ValueError(f"未知步骤类型: {step_type}")
    return cls()


def step_to_dict(step) -> dict:
    """将步骤实例序列化为字典。"""
    return asdict(step)


def step_from_dict(data: dict):
    """从字典反序列化步骤实例。"""
    step_type = StepType(data.get("type", ""))
    cls = STEP_CLASS_MAP.get(step_type)
    if cls is None:
        raise ValueError(f"未知步骤类型: {step_type}")
    # 只传入该类已知的字段
    import dataclasses
    known = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)
