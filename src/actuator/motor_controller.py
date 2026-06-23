"""DRV2605 震动马达控制器封装。

从 test3.py 提取核心控制逻辑（I2C 通信、DRV2605 初始化、效果播放），
不直接 import 原脚本。

支持 real / mock 两种模式：
  - real: 通过 smbus2 驱动真实 DRV2605 硬件
  - mock: 仅打印控制指令，不驱动硬件

冷却机制：
  - 同一等级重复触发需间隔 >= min_interval_sec
  - 高风险（level=2）可打断中风险（level=1）
  - 低风险（level=0）不重复发送静默命令
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.actuator.alert_pattern import pattern_for_level, ALERT_LABELS
from src.fusion.data_types import MotorCommand

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(path: str) -> str:
    """将相对路径解析为基于项目根的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return str(p)


def _load_motor_config(config_path: str = "configs/risk_params.yaml") -> dict:
    """加载马达控制配置。"""
    with open(_resolve_path(config_path), "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params.get("motor", {})


class MotorController:
    """DRV2605 震动马达控制器。"""

    def __init__(
        self,
        mode: str = "mock",
        i2c_bus: int = 3,
        i2c_addr: int = 0x5A,
        config_path: str = "configs/risk_params.yaml",
    ) -> None:
        self.mode = mode
        self.i2c_bus = i2c_bus
        self.i2c_addr = i2c_addr
        self._bus = None

        # 加载配置
        cfg = _load_motor_config(config_path)
        self.min_interval_sec = cfg.get("min_interval_sec", 0.5)
        self.cooldown_sec = cfg.get("cooldown_sec", 2.0)
        self.high_priority_interrupt = cfg.get("high_priority_interrupt", True)

        medium = cfg.get("medium", {})
        self.medium_effect_ids = medium.get("effect_ids", [1])
        self.medium_duration = medium.get("pulse_duration_sec", 0.05)
        self.medium_repeat = medium.get("repeat_count", 3)

        high = cfg.get("high", {})
        self.high_effect_ids = high.get("effect_ids", [64])
        self.high_duration = high.get("pulse_duration_sec", 0.5)

        # 冷却状态
        self._last_level: int = -1  # 上次触发的等级
        self._last_level_time: float = 0.0  # 上次触发时间
        self._last_any_time: float = 0.0  # 上次任何触发时间
        self._is_medium_playing: bool = False  # 中风险是否正在播放

        # 最新指令
        self._latest: Optional[MotorCommand] = None

    # ---- 生命周期 ----

    def start(self) -> None:
        """初始化马达连接。"""
        if self.is_real:
            try:
                from smbus2 import SMBus

                self._bus = SMBus(self.i2c_bus)
                self._drv2605_init()
                print(f"[MotorController] 已初始化 DRV2605 (bus={self.i2c_bus}, addr=0x{self.i2c_addr:02X})")
            except ImportError:
                print("[MotorController] smbus2 未安装，无法驱动真实硬件。自动切换至 mock 模式。")
                self.mode = "mock"
            except Exception as e:
                print(f"[MotorController] 硬件初始化失败: {e}。自动切换至 mock 模式。")
                self.mode = "mock"

        if self.is_mock:
            print("[MotorController] mock 模式启动")

    def stop(self) -> None:
        """释放马达资源。"""
        if self._bus is not None:
            try:
                # 停止震动
                self._bus.write_byte_data(self.i2c_addr, 0x0C, 0x00)
                self._bus.close()
                print("[MotorController] I2C 已关闭")
            except Exception:
                pass
            self._bus = None
        else:
            print("[MotorController] 已停止")

    # ---- 核心接口 ----

    def execute(self, command: MotorCommand) -> None:
        """执行马达控制指令。

        含冷却机制：
          - level=0 且与上次相同 → 不重复发送
          - 同一等级冷却未到期 → 跳过
          - level=2 可打断正在执行的 level=1
        """
        now = time.time()
        self._latest = command

        # level=0：不重复发送静默
        if command.risk_level == 0:
            if self._last_level == 0:
                return
            self._print_action(command)
            self._last_level = 0
            self._last_level_time = now
            self._is_medium_playing = False
            return

        # 检查冷却（level=2 高风险不受同级冷却限制，保证持续强震）
        if command.risk_level != 2 and self._last_level == command.risk_level:
            elapsed = now - self._last_level_time
            if elapsed < self.min_interval_sec:
                return  # 冷却未到

        # 检查打断：高风险打断中风险
        if command.risk_level == 2 and self._last_level == 1:
            if self.high_priority_interrupt:
                self._print_action(command, interrupt=True)
                self._execute_hardware(command)
                self._last_level = 2
                self._last_level_time = now
                self._last_any_time = now
                self._is_medium_playing = False
                return

        # 正常执行
        if command.risk_level >= 1:
            self._print_action(command)
            self._execute_hardware(command)
            self._last_level = command.risk_level
            self._last_level_time = now
            self._last_any_time = now
            self._is_medium_playing = (command.risk_level == 1)

    def alert_low(self) -> None:
        """低风险（level=0）：静默。"""
        self.execute(MotorCommand(risk_level=0, pattern="silent"))

    def alert_medium(self, risk_score: float = 0.0) -> None:
        """中风险（level=1）：短促轻震。"""
        cmd = pattern_for_level(
            1, risk_score,
            medium_effect_ids=self.medium_effect_ids,
            medium_duration=self.medium_duration,
            medium_repeat=self.medium_repeat,
        )
        self.execute(cmd)

    def alert_high(self, risk_score: float = 0.0) -> None:
        """高风险（level=2）：持续强震。"""
        cmd = pattern_for_level(
            2, risk_score,
            high_effect_ids=self.high_effect_ids,
            high_duration=self.high_duration,
        )
        self.execute(cmd)

    def get_latest(self) -> Optional[MotorCommand]:
        """返回最近一次执行的马达指令。"""
        return self._latest

    # ---- 内部 ----

    def _execute_hardware(self, cmd: MotorCommand) -> None:
        """实际驱动硬件。"""
        if self.is_mock or self._bus is None:
            return

        try:
            if cmd.risk_level == 1:
                for _ in range(self.medium_repeat):
                    for eid in self.medium_effect_ids:
                        self._play_effect(eid, self.medium_duration)
            elif cmd.risk_level == 2:
                for eid in self.high_effect_ids:
                    self._play_effect(eid, self.high_duration)
        except Exception as e:
            print(f"[MotorController] 硬件执行异常: {e}")

    def _print_action(self, cmd: MotorCommand, interrupt: bool = False) -> None:
        """打印控制指令（mock/real 均打印）。"""
        label = ALERT_LABELS.get(cmd.risk_level, "未知")
        interrupt_tag = " [打断中风险]" if interrupt else ""
        if self.is_mock:
            print(f"[MotorController Mock]{interrupt_tag} "
                  f"level={cmd.risk_level} ({label}) | "
                  f"pattern={cmd.pattern} | "
                  f"duration={cmd.duration_ms}ms")
        else:
            print(f"[MotorController]{interrupt_tag} {cmd.risk_level} ({label})")

    def _drv2605_init(self) -> None:
        """DRV2605 初始化序列（来自 test3.py）。"""
        if self._bus is None:
            return
        self._bus.write_byte_data(self.i2c_addr, 0x01, 0x00)  # 复位
        self._bus.write_byte_data(self.i2c_addr, 0x1D, 0x01)  # LRA 模式
        self._bus.write_byte_data(self.i2c_addr, 0x03, 0x01)  # 内置效果库

    def _play_effect(self, effect_id: int, duration: float) -> None:
        """播放指定效果（来自 test3.py）。"""
        if self._bus is None:
            return
        self._bus.write_byte_data(self.i2c_addr, 0x04, effect_id)
        self._bus.write_byte_data(self.i2c_addr, 0x0C, 0x01)  # GO
        time.sleep(duration)
        self._bus.write_byte_data(self.i2c_addr, 0x0C, 0x00)  # 停止

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def is_real(self) -> bool:
        return self.mode == "real"
