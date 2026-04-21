# -*- coding: utf-8 -*-
"""
数字识别测试脚本 - TrOCR 版本

用法:
    python test_recognize.py <图片路径>                                  # 本地 GPU 模式
    python test_recognize.py <图片路径> --server http://localhost:5000   # 服务模式
    python test_recognize.py <图片路径> --cpu                            # 本地 CPU 模式
    python test_recognize.py <图片路径> --repeat 5                       # 重复识别 5 次（测试稳定性）
    python test_recognize.py <图片路径> --debug                          # 开启调试输出

示例:
    python test_recognize.py 1.png
    python test_recognize.py 1.png --server http://localhost:5000
    python test_recognize.py 1.png --repeat 10 --debug

优化点：
- 增加 --repeat 参数，支持多次识别以测试稳定性和平均耗时
- 增加 --debug 参数，透传给 VisionEngine 开启详细日志
- 增加识别结果一致性统计（多次识别时）
- 错误信息更友好，区分文件不存在、引擎初始化失败、识别失败
- 使用 logging 模块，支持调试级别控制
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

# ------------------------------------------------------------------
# 日志配置（在 import vision 之前设置，确保子模块日志也生效）
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # 默认只显示警告及以上，--debug 时切换为 DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_recognize")


# ------------------------------------------------------------------
# 核心测试函数
# ------------------------------------------------------------------

def test_recognize(
    image_path: str,
    server_url: Optional[str] = None,
    use_gpu: bool = True,
    repeat: int = 1,
    debug: bool = False,
) -> int:
    """
    测试图片数字识别，支持多次重复以评估稳定性。

    Args:
        image_path:  图片文件路径
        server_url:  OCR 服务地址（None=本地模式）
        use_gpu:     是否使用 GPU（本地模式有效）
        repeat:      重复识别次数（>=1）
        debug:       是否开启调试输出

    Returns:
        最后一次识别结果（成功返回正整数，失败返回 -1）
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 确定运行模式描述
    if server_url:
        mode_desc = f"服务模式 ({server_url})"
    elif use_gpu:
        mode_desc = "本地 GPU 模式"
    else:
        mode_desc = "本地 CPU 模式"

    _print_header("数字识别测试（TrOCR 引擎）")
    print(f"  输入图片: {image_path}")
    print(f"  运行模式: {mode_desc}")
    print(f"  重复次数: {repeat}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 步骤1：初始化引擎
    # ------------------------------------------------------------------
    _print_section("步骤1", "初始化 TrOCR 引擎")

    t_init = time.perf_counter()
    try:
        from vision import VisionEngine
        engine = VisionEngine(
            debug=debug,
            server_url=server_url,
            use_gpu=use_gpu,
        )
    except Exception as exc:
        print(f"  ✗ 引擎初始化失败: {exc}")
        logger.exception("引擎初始化失败")
        return -1

    init_elapsed = time.perf_counter() - t_init
    print(f"  引擎初始化耗时: {init_elapsed:.2f} 秒")

    # ------------------------------------------------------------------
    # 步骤2：执行识别（支持多次重复）
    # ------------------------------------------------------------------
    _print_section("步骤2", f"执行 TrOCR 识别（共 {repeat} 次）")

    results: List[int] = []
    raw_texts: List[str] = []
    elapsed_list: List[float] = []

    for i in range(1, repeat + 1):
        t_rec = time.perf_counter()
        number, raw_text, details = engine.recognize_from_image_verbose(image_path)
        rec_elapsed = time.perf_counter() - t_rec

        results.append(number)
        raw_texts.append(raw_text)
        elapsed_list.append(rec_elapsed)

        status = "✓" if number > 0 else "✗"
        print(
            f"  [{i:02d}/{repeat:02d}] {status} "
            f"结果: {number if number > 0 else '失败':<10} "
            f"原始文本: '{raw_text}'  "
            f"耗时: {rec_elapsed:.3f}s"
        )

        # 服务模式打印服务端耗时
        if details and isinstance(details, dict):
            srv_ms = details.get("time_ms")
            if srv_ms is not None:
                print(f"         服务端耗时: {srv_ms} ms")

    # ------------------------------------------------------------------
    # 步骤3：统计分析
    # ------------------------------------------------------------------
    _print_section("步骤3", "统计分析")

    valid_results = [r for r in results if r > 0]
    avg_elapsed = sum(elapsed_list) / len(elapsed_list)
    min_elapsed = min(elapsed_list)
    max_elapsed = max(elapsed_list)

    print(f"  识别成功率: {len(valid_results)}/{repeat} "
          f"({len(valid_results)/repeat*100:.0f}%)")
    print(f"  平均耗时:   {avg_elapsed:.3f}s")
    print(f"  最快耗时:   {min_elapsed:.3f}s")
    print(f"  最慢耗时:   {max_elapsed:.3f}s")

    if repeat > 1 and valid_results:
        unique_results = set(valid_results)
        consistency = len(valid_results) - len(unique_results) + 1
        print(f"  结果一致性: {len(valid_results)} 次有效识别中有 "
              f"{len(unique_results)} 种不同结果")
        if len(unique_results) == 1:
            print(f"  ✓ 所有有效识别结果一致: {valid_results[0]}")
        else:
            print(f"  ⚠ 识别结果存在差异: {sorted(unique_results)}")

    # ------------------------------------------------------------------
    # 步骤4：最终结论
    # ------------------------------------------------------------------
    _print_section("步骤4", "最终结论")

    final_number = results[-1]
    if final_number > 0:
        print(f"  ✓ 最终识别结果: {final_number}")
    else:
        print("  ✗ 识别失败，未能提取有效数字")
        if not valid_results:
            print("  建议：检查图片质量、识别区域是否包含清晰数字")
        else:
            print(f"  注意：部分识别成功，最常见结果: "
                  f"{max(set(valid_results), key=valid_results.count)}")

    print("=" * 50)
    return final_number


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _print_header(title: str) -> None:
    """打印标题横幅。"""
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


def _print_section(step: str, desc: str) -> None:
    """打印步骤分隔线。"""
    print(f"\n{'=' * 50}")
    print(f"[{step}] {desc}")
    print("=" * 50)


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrOCR 数字识别测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_recognize.py 1.png
  python test_recognize.py 1.png --server http://localhost:5000
  python test_recognize.py 1.png --cpu --repeat 5
  python test_recognize.py 1.png --debug
        """,
    )
    parser.add_argument("image", help="图片文件路径")
    parser.add_argument(
        "--server", type=str, default=None,
        metavar="URL",
        help="OCR 服务地址（如 http://localhost:5000），不指定则使用本地模式",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="强制使用 CPU（本地模式下有效）",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        metavar="N",
        help="重复识别次数，用于测试稳定性（默认 1）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="开启调试输出（显示 OCR 原始文本等详细信息）",
    )

    args = parser.parse_args()

    # 参数校验
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在: {args.image}")
        sys.exit(1)

    if args.repeat < 1:
        print("错误: --repeat 必须 >= 1")
        sys.exit(1)

    result = test_recognize(
        image_path=args.image,
        server_url=args.server,
        use_gpu=not args.cpu,
        repeat=args.repeat,
        debug=args.debug,
    )

    # 以识别结果决定退出码（方便 CI/脚本调用）
    sys.exit(0 if result > 0 else 1)
