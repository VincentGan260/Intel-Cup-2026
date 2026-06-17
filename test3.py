from smbus2 import SMBus
import time

# DRV2605 配置
DRV2605_ADDR = 0x5A
bus = SMBus(3)

def drv2605_init():
    bus.write_byte_data(DRV2605_ADDR, 0x01, 0x00)  # 复位
    bus.write_byte_data(DRV2605_ADDR, 0x1D, 0x01)  # 配置为 LRA 线性马达模式
    bus.write_byte_data(DRV2605_ADDR, 0x03, 0x01)   # 选择内置效果库

def play_effect(effect_id, duration):
    """播放指定效果，持续指定时长"""
    bus.write_byte_data(DRV2605_ADDR, 0x04, effect_id)
    bus.write_byte_data(DRV2605_ADDR, 0x0C, 0x01)
    time.sleep(duration)
    bus.write_byte_data(DRV2605_ADDR, 0x0C, 0x00)

# --- 震动分级定义（按你说的逻辑） ---
def level_1():
    """强度1：最轻微，短粗+间隔短"""
    print("强度 1：短粗轻震")
    for _ in range(3):
        play_effect(1, 0.05)   # 极短脉冲
        time.sleep(0.05)

def level_2():
    """强度2：比1稍强，单脉冲略长"""
    print("强度 2：中等短震")
    play_effect(14, 0.15)    # 稍长的单震

def level_3():
    """强度3：持续震动，时间变长"""
    print("强度 3：持续震动")
    play_effect(64, 0.3)     # 持续中等震动

def level_4():
    """强度4：长震动，时间最长，区分度最大"""
    print("强度 4：长震警报")
    play_effect(64, 0.5)     # 长时间强震

if __name__ == "__main__":
    drv2605_init()
    print("DRV2605 震动分级测试开始...")
    try:
        while True:
            # 按强度从弱到强循环
            level_1()
            time.sleep(1)

            level_2()
            time.sleep(1)

            level_3()
            time.sleep(1)

            level_4()
            time.sleep(2)

    except KeyboardInterrupt:
        print("程序结束")
