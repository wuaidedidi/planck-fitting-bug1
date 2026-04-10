"""
日志配置模块

提供结构化日志输出，支持控制台和文件双通道。
日志格式包含时间戳、级别、模块名，便于问题定位。
"""

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "planck_fitting",
    level: int = logging.INFO,
    log_dir: str = "output/logs",
) -> logging.Logger:
    """
    初始化并返回一个配置完整的 Logger 实例。

    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志文件输出目录

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_path / "planck_fitting.log", encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    获取子模块 Logger。

    Args:
        module_name: 模块名称

    Returns:
        子模块 Logger 实例
    """
    return logging.getLogger(f"planck_fitting.{module_name}")
