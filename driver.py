# -*- coding: utf-8 -*-
"""
硬件控制抽象模块 (Mouse Driver)

本模块封装鼠标移动与点击操作，提供统一的硬件控制接口。
当前为模拟实现，后续可替换为真实硬件驱动（如 KmBox / 罗技驱动 / 串口通信）。

优化点：
- 使用 logging 模块统一日志输出
- 增加坐标合法性校验，防止传入负数或超大坐标
- 增加 click_count 统计，便于调试
- 预留 close() 方法，供串口等资源释放使用
- 支持上下文管理器（with 语句），确保资源安全释放
"""

import logging
import random
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 屏幕坐标合法范围（超出则警告，但仍执行）
_MAX_COORD = 32767  # Win32 虚拟屏幕最大坐标


class MouseDriver:
    """
    鼠标驱动抽象类

    封装鼠标的移动和点击操作。当前使用模拟延迟实现，
    预留了串口连接等配置参数，方便后续接入真实硬件驱动。

    使用示例：
        driver = MouseDriver()
        driver.move_and_click(960, 600)
        driver.close()

        # 或使用上下文管理器
        with MouseDriver() as driver:
            driver.move_and_click(960, 600)
    """

    def __init__(self):
        """
        初始化鼠标驱动。

        当前为空实现，预留以下配置参数供后续扩展：
        - 串口号 (COM port)
        - 波特率 (baud rate)
        - 设备类型 (KmBox / 罗技驱动 / Arduino 等)
        """
        # TODO: 后续可在此处添加串口连接初始化代码
        # 示例:
        #   self._serial_port = "COM3"
        #   self._baud_rate = 115200
        #   self._connection = serial.Serial(self._serial_port, self._baud_rate)

        self._click_count: int = 0
        logger.debug("MouseDriver 初始化完成")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def move_and_click(self, x: int, y: int) -> bool:
        """
        移动鼠标到指定坐标并执行左键单击。

        使用 pynput 实现真实鼠标控制。
        后续可替换为串口通信代码（KmBox / 罗技驱动等）。

        Args:
            x: 目标点击位置的屏幕 X 坐标
            y: 目标点击位置的屏幕 Y 坐标

        Returns:
            True 表示点击成功，False 表示参数非法或执行失败
        """
        if not self._validate_coords(x, y):
            return False

        try:
            from pynput.mouse import Button, Controller as MouseController
            mouse = MouseController()

            self._click_count += 1
            logger.info("点击坐标: (%d, %d)（第 %d 次）", x, y, self._click_count)

            # 移动到目标位置
            mouse.position = (x, y)
            time.sleep(0.05)

            # 按下并释放左键
            press_duration = random.uniform(0.03, 0.08)
            mouse.press(Button.left)
            time.sleep(press_duration)
            mouse.release(Button.left)

            logger.debug("点击完成 (%d, %d)", x, y)
            return True

        except Exception as exc:
            logger.error("点击操作异常: %s", exc)
            return False

    @property
    def click_count(self) -> int:
        """返回累计点击次数。"""
        return self._click_count

    def close(self) -> None:
        """
        释放硬件资源（串口连接等）。

        当前为空实现，后续接入真实硬件时在此处关闭连接。
        """
        # TODO: self._connection.close()
        logger.debug("MouseDriver 资源已释放（累计点击 %d 次）", self._click_count)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_coords(x: int, y: int) -> bool:
        """
        校验坐标合法性。

        Args:
            x: X 坐标
            y: Y 坐标

        Returns:
            True 表示坐标合法
        """
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            logger.error("坐标类型非法: x=%r, y=%r", x, y)
            return False
        if x < 0 or y < 0:
            logger.error("坐标不能为负数: (%d, %d)", x, y)
            return False
        if x > _MAX_COORD or y > _MAX_COORD:
            logger.warning("坐标超出常规范围: (%d, %d)，仍将执行", x, y)
        return True

    # ------------------------------------------------------------------
    # 上下文管理器支持
    # ------------------------------------------------------------------

    def __enter__(self) -> "MouseDriver":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
