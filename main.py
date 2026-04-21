# -*- coding: utf-8 -*-
"""
主控逻辑模块 (Sniper Bot)

本模块为"极速价格识别与自动点击"系统的主控入口。
核心流程：截屏 → TrOCR 价格识别 → 条件判定 → 自动点击购买。
支持 F9 热键安全退出。

识别引擎：TrOCR（微软 Vision Transformer，高准确率，支持 GPU 加速）
支持本地模式（直接加载模型）和服务模式（HTTP 调用 ocr_server.py）

优化点：
- 配置集中到 BotConfig 数据类，便于外部修改和单元测试
- 增加运行统计（总帧数、触发次数、平均识别耗时）
- 购买后可配置最大触发次数，防止无限重复购买
- 使用 logging 模块统一日志，支持日志级别控制
- 线程安全的停止标志（threading.Event）
- 主循环异常计数，连续异常超阈值时自动退出，避免死循环
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from pynput import keyboard

from vision import VisionEngine
from driver import MouseDriver

# ------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SniperBot")


# ------------------------------------------------------------------
# 配置数据类
# ------------------------------------------------------------------

@dataclass
class BotConfig:
    """
    狙击机器人运行配置。

    将所有可调参数集中在此处，方便修改和测试。
    """

    # 目标触发价格（当识别价格 <= 此值时触发购买）
    target_price: int = 150_000

    # 价格识别区域（mss 格式）
    price_region: dict = field(default_factory=lambda: {
        "top": 400,
        "left": 800,
        "width": 200,
        "height": 50,
    })

    # 购买按钮的屏幕物理坐标 (x, y)
    buy_button_pos: tuple = (960, 600)

    # 目标帧率（每秒识别次数）
    # TrOCR GPU 模式下单次推理约 50~200ms，2 FPS 即可满足 3 秒响应需求
    target_fps: float = 2.0

    # OCR 服务地址（None=本地模式，设置 URL=服务模式）
    # 服务模式需先启动: python ocr_server.py
    ocr_server_url: Optional[str] = None

    # 是否使用 GPU（本地模式下有效）
    use_gpu: bool = True

    # 购买后冷却时间（秒），防止重复触发
    buy_cooldown: float = 2.0

    # 最大购买次数（0 = 不限制）
    max_buy_count: int = 0

    # 连续异常帧数阈值，超过后自动退出主循环
    max_consecutive_errors: int = 10


# ------------------------------------------------------------------
# 主控类
# ------------------------------------------------------------------

class SniperBot:
    """
    狙击机器人主控类

    负责协调视觉引擎和鼠标驱动，实现价格监控与自动购买逻辑。
    当识别到的价格低于或等于目标价格时，自动触发点击购买操作。
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """
        初始化狙击机器人。

        Args:
            config: 运行配置，None 时使用默认配置
        """
        self.config = config or BotConfig()

        # 线程安全的停止事件（比 bool 标志更可靠）
        self._stop_event = threading.Event()

        # 运行统计
        self._stats = {
            "total_frames": 0,
            "recognized_frames": 0,
            "buy_count": 0,
            "total_recognize_ms": 0.0,
        }

        # 初始化子模块
        logger.info("初始化视觉引擎...")
        self._vision = VisionEngine(
            server_url=self.config.ocr_server_url,
            use_gpu=self.config.use_gpu,
        )

        logger.info("初始化鼠标驱动...")
        self._driver = MouseDriver()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """主循环是否正在运行。"""
        return not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_mode_info(self) -> str:
        """返回当前运行模式描述字符串。"""
        if self.config.ocr_server_url:
            return f"服务模式 ({self.config.ocr_server_url})"
        return f"本地模式 ({'GPU' if self.config.use_gpu else 'CPU'})"

    def _should_buy(self, price: int) -> bool:
        """
        判断是否应该触发购买。

        Args:
            price: 当前识别价格

        Returns:
            True 表示应触发购买
        """
        if price <= 0:
            return False
        if price > self.config.target_price:
            return False
        if self.config.max_buy_count > 0:
            if self._stats["buy_count"] >= self.config.max_buy_count:
                logger.info(
                    "已达最大购买次数 %d，不再触发", self.config.max_buy_count
                )
                return False
        return True

    def _do_buy(self, price: int) -> None:
        """
        执行购买操作。

        Args:
            price: 触发购买时的识别价格
        """
        self._stats["buy_count"] += 1
        logger.info(
            "★ 触发购买！识别价格: %d <= 目标价格: %d（第 %d 次）",
            price,
            self.config.target_price,
            self._stats["buy_count"],
        )
        print(
            f"[SniperBot] ★ 触发购买！识别价格: {price} <= "
            f"目标价格: {self.config.target_price}"
            f"（第 {self._stats['buy_count']} 次）"
        )

        self._driver.move_and_click(
            self.config.buy_button_pos[0],
            self.config.buy_button_pos[1],
        )

        # 冷却等待，防止重复触发
        self._stop_event.wait(timeout=self.config.buy_cooldown)

    def _print_stats(self) -> None:
        """打印运行统计摘要。"""
        s = self._stats
        avg_ms = (
            s["total_recognize_ms"] / s["recognized_frames"]
            if s["recognized_frames"] > 0
            else 0
        )
        print("\n" + "=" * 60)
        print("  运行统计摘要")
        print("=" * 60)
        print(f"  总帧数:       {s['total_frames']}")
        print(f"  有效识别帧:   {s['recognized_frames']}")
        print(f"  触发购买次数: {s['buy_count']}")
        print(f"  平均识别耗时: {avg_ms:.0f} ms")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def start_loop(self) -> None:
        """
        启动主监控循环（阻塞调用）。

        以目标帧率运行，持续截屏识别价格。
        当价格满足触发条件时，自动执行购买点击。
        通过 stop() 方法或 _stop_event 安全终止。
        """
        self._stop_event.clear()
        cfg = self.config
        frame_interval = 1.0 / cfg.target_fps
        consecutive_errors = 0

        logger.info(
            "主循环已启动，目标帧率: %.1f FPS，模式: %s",
            cfg.target_fps,
            self._get_mode_info(),
        )
        print(f"[SniperBot] 主循环已启动，目标帧率: {cfg.target_fps} FPS")

        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            try:
                # ===== 截屏并识别价格 =====
                current_price = self._vision.capture_and_recognize(
                    cfg.price_region
                )
                recognize_ms = (time.perf_counter() - t_start) * 1000

                # 更新统计
                self._stats["total_frames"] += 1
                if current_price > 0:
                    self._stats["recognized_frames"] += 1
                    self._stats["total_recognize_ms"] += recognize_ms
                    logger.debug(
                        "当前价格: %d（耗时: %.0f ms）", current_price, recognize_ms
                    )
                    print(
                        f"[SniperBot] 当前识别价格: {current_price}"
                        f"（耗时: {recognize_ms:.0f} ms）"
                    )

                # ===== 购买判定 =====
                if self._should_buy(current_price):
                    self._do_buy(current_price)

                    # 达到最大购买次数后自动退出
                    if (
                        cfg.max_buy_count > 0
                        and self._stats["buy_count"] >= cfg.max_buy_count
                    ):
                        logger.info("已完成全部购买任务，自动退出")
                        self._stop_event.set()
                        break

                # 重置连续错误计数
                consecutive_errors = 0

            except Exception as exc:
                consecutive_errors += 1
                logger.error(
                    "主循环单帧异常（连续 %d 次）: %s",
                    consecutive_errors,
                    exc,
                )
                print(f"[SniperBot] 主循环单帧异常: {exc}")

                if consecutive_errors >= cfg.max_consecutive_errors:
                    logger.critical(
                        "连续异常次数达到阈值 %d，自动退出",
                        cfg.max_consecutive_errors,
                    )
                    print(
                        f"[SniperBot] 连续异常 {cfg.max_consecutive_errors} 次，"
                        "自动退出"
                    )
                    self._stop_event.set()
                    break

            # ===== 帧率控制 =====
            elapsed = time.perf_counter() - t_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                # 使用 Event.wait 代替 time.sleep，可被 stop() 立即中断
                self._stop_event.wait(timeout=remaining)

        logger.info("主循环已安全退出")
        print("[SniperBot] 主循环已安全退出")
        self._print_stats()

    def stop(self) -> None:
        """
        安全停止主监控循环。

        设置停止事件，主循环将在当前帧结束后立即退出（无需等待 sleep 结束）。
        """
        logger.info("收到停止信号，正在退出...")
        print("[SniperBot] 收到停止信号，正在退出...")
        self._stop_event.set()


# ------------------------------------------------------------------
# 程序入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 在此处修改配置参数
    config = BotConfig(
        target_price=150_000,
        price_region={"top": 400, "left": 800, "width": 200, "height": 50},
        buy_button_pos=(960, 600),
        target_fps=2.0,
        ocr_server_url="http://localhost:5000",   # 例如 "http://localhost:5000"
        use_gpu=True,
        buy_cooldown=2.0,
        max_buy_count=0,       # 0 = 不限制
        max_consecutive_errors=10,
    )

    bot = SniperBot(config)

    def on_press(key):
        """键盘热键回调：F9 安全退出。"""
        if key == keyboard.Key.f9:
            print("\n[热键] 检测到 F9 按下，正在安全退出...")
            bot.stop()
            return False  # 停止 pynput 监听器线程

    # 启动后台键盘监听线程
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 打印启动信息
    print("=" * 60)
    print("  极速价格识别与自动点击系统 - 已启动")
    print(f"  识别引擎: TrOCR（微软 Vision Transformer）")
    print(f"  运行模式: {bot._get_mode_info()}")
    print("=" * 60)
    print(f"  目标价格:   <= {config.target_price}")
    print(f"  识别区域:   {config.price_region}")
    print(f"  购买按钮:   {config.buy_button_pos}")
    print(f"  目标帧率:   {config.target_fps} FPS")
    print(f"  购买冷却:   {config.buy_cooldown} 秒")
    print(f"  最大购买:   {'不限' if config.max_buy_count == 0 else config.max_buy_count} 次")
    print(f"  退出热键:   F9")
    print("=" * 60)

    # 启动主监控循环（阻塞主线程）
    bot.start_loop()

    print("[主程序] 脚本已完全退出")
