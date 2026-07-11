"""
HLK-LD2451 雷达模块测试脚本
============================
通信协议版本: V1.03 (2024-7-1)
当前真机参数: 256000 baud, 1 stop bit, no parity

命令格式:
  帧头(4B) + 数据长度(2B, LE) + 命令字(2B) + 命令值(NB) + 帧尾(4B)
  帧头: FD FC FB FA
  帧尾: 04 03 02 01

数据输出格式:
  帧头(4B) + 数据长度(2B, LE) + 目标数(1B) + 报警信息(1B) + 目标信息*N(5B) + 帧尾(4B)
  帧头: F4 F3 F2 F1
  帧尾: F8 F7 F6 F5

每个目标信息(5字节):
  [角度(1B)] [距离(1B)] [速度方向(1B)] [速度值(1B)] [信噪比(1B)]
  实际角度 = 原始值 - 0x80 (范围: -128~127度)
  当前前向安装真机标定: 01=靠近, 00=远离；原始正角需取反为项目坐标
  距离: 0~100米
  速度: 0~120 km/h
"""

import serial
import serial.tools.list_ports
import struct
import time
import sys
from threading import Thread, Event


# ============================================================
#  协议常量
# ============================================================
FRAME_HEADER = b"\xFD\xFC\xFB\xFA"
FRAME_END = b"\x04\x03\x02\x01"
DATA_HEADER = b"\xF4\xF3\xF2\xF1"
DATA_END = b"\xF8\xF7\xF6\xF5"

# 命令字
CMD_ENABLE_CONFIG = 0x00FF  # 使能配置
CMD_END_CONFIG = 0x00FE  # 结束配置
CMD_READ_FIRMWARE = 0x00A0  # 读取固件版本
CMD_SET_DETECTION_PARAM = 0x0002  # 设置目标检测参数
CMD_READ_DETECTION_PARAM = 0x0012  # 读取目标检测参数
CMD_SET_SENSITIVITY = 0x0003  # 设置灵敏度
CMD_READ_SENSITIVITY = 0x0013  # 读取灵敏度
CMD_SET_BAUDRATE = 0x00A1  # 设置波特率
CMD_FACTORY_RESET = 0x00A2  # 恢复出厂设置
CMD_REBOOT = 0x00A3  # 重启模块


def build_command(cmd_word: int, cmd_value: bytes = b"") -> bytes:
    """构建命令帧"""
    payload = struct.pack("<H", cmd_word) + cmd_value
    length = len(payload)
    frame = FRAME_HEADER + struct.pack("<H", length) + payload + FRAME_END
    return frame


def parse_ack(data: bytes):
    """解析雷达 ACK 回复"""
    if len(data) < 10:
        return None, None, "数据太短"
    if not data.startswith(FRAME_HEADER):
        return None, None, f"帧头错误: {data[:4].hex()}"
    if not data.endswith(FRAME_END):
        return None, None, f"帧尾错误: {data[-4:].hex()}"

    # 数据长度(小端)
    payload_len = struct.unpack("<H", data[4:6])[0]
    payload = data[6 : 6 + payload_len]

    if len(payload) < 2:
        return None, None, "有效载荷太短"

    # ACK 回复: 发送命令字 & 0x0100
    resp_cmd = struct.unpack("<H", payload[:2])[0]
    return_val = payload[2:]

    return resp_cmd, return_val, None


# ============================================================
#  扫描串口
# ============================================================
def scan_ports():
    """扫描并列出所有可用串口"""
    ports = serial.tools.list_ports.comports()
    print(f"\n{'='*60}")
    print(f"  可用串口列表:")
    print(f"{'='*60}")
    if not ports:
        print("  (未检测到串口设备)")
    for p in ports:
        print(f"  {p.device:8s} - {p.description}")
    print(f"{'='*60}\n")
    return ports


# ============================================================
#  雷达基本通信测试
# ============================================================
def test_connection(port: str, baudrate: int = 256000, timeout: float = 0.5):
    """测试与雷达模块的基本连接"""
    print(f"\n>>> 正在连接 {port} (波特率: {baudrate})...")
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"    连接成功! (端口: {ser.port}, 波特率: {ser.baudrate})")
        return ser
    except serial.SerialException as e:
        print(f"    [错误] 连接失败: {e}")
        return None


def send_and_wait_ack(ser: serial.Serial, cmd: bytes, timeout: float = 1.0) -> bytes:
    """发送命令并等待 ACK（会过滤掉中间的数据帧）"""
    # 先排空输入缓冲区中的旧数据帧
    if ser.in_waiting:
        _ = ser.read(ser.in_waiting)
    ser.reset_output_buffer()

    print(f"    发送: {cmd.hex().upper()}")
    ser.write(cmd)

    # 持续读取，直到找到 ACK 帧（以 FD FC FB FA 开头）或超时
    buffer = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting)
            buffer.extend(chunk)
            # 查找 ACK 帧头
            idx = buffer.find(FRAME_HEADER)
            if idx >= 0:
                # 跳过前面的非帧头数据
                remaining = buffer[idx:]
                if len(remaining) >= 10:
                    payload_len = struct.unpack("<H", remaining[4:6])[0]
                    frame_len = 4 + 2 + payload_len + 4
                    if len(remaining) >= frame_len and remaining[frame_len - 4 : frame_len] == FRAME_END:
                        ack = bytes(remaining[:frame_len])
                        print(f"    回复: {ack.hex().upper()}")
                        return ack
        time.sleep(0.05)

    # 超时，返回已收到的数据（用于调试）
    resp = bytes(buffer)
    if resp:
        print(f"    回复(非ACK): {resp.hex().upper()}")
    else:
        print(f"    回复: (无响应)")
    return resp


def test_enable_config(ser: serial.Serial):
    """测试使能配置命令（失败时自动重试一次）"""
    for attempt in range(2):
        print(f"\n  [1] 使能配置模式...{' (重试)' if attempt > 0 else ''}")
        cmd = build_command(CMD_ENABLE_CONFIG, b"\x01\x00")
        resp = send_and_wait_ack(ser, cmd, timeout=1.5)

        resp_cmd, ret_val, err = parse_ack(resp)
        if not err and resp_cmd == (CMD_ENABLE_CONFIG | 0x0100):
            status = struct.unpack("<H", ret_val[:2])[0]
            if status == 0:
                ver = struct.unpack("<H", ret_val[2:4])[0]
                print(f"      成功! 协议版本: V{ver}")
                return True
        if attempt == 0:
            # 第一次失败，等待雷达安静后再试
            time.sleep(0.3)
    print(f"      [错误] 使能配置失败")
    return False


def test_end_config(ser: serial.Serial):
    """测试结束配置命令"""
    print(f"\n  [2] 结束配置模式...")
    cmd = build_command(CMD_END_CONFIG)
    resp = send_and_wait_ack(ser, cmd)

    resp_cmd, ret_val, err = parse_ack(resp)
    if err:
        print(f"      解析失败: {err}")
        return False
    if resp_cmd == (CMD_END_CONFIG | 0x0100):
        status = struct.unpack("<H", ret_val[:2])[0]
        if status == 0:
            print(f"      成功!")
            return True
        else:
            print(f"      失败! 状态码: {status}")
            return False
    return False


def test_read_firmware(ser: serial.Serial):
    """测试读取固件版本"""
    print(f"\n  [3] 读取固件版本...")
    # 先使能配置
    if not test_enable_config(ser):
        return None
    # 等待雷达完全进入配置模式
    time.sleep(0.3)

    cmd = build_command(CMD_READ_FIRMWARE)
    resp = send_and_wait_ack(ser, cmd, timeout=2.0)

    resp_cmd, ret_val, err = parse_ack(resp)
    if err or resp_cmd != (CMD_READ_FIRMWARE | 0x0100):
        print(f"      读取固件版本失败: {err}")
        test_end_config(ser)
        return None

    status = struct.unpack("<H", ret_val[:2])[0]
    if status != 0:
        print(f"      失败! 状态码: {status}")
        test_end_config(ser)
        return None

    # 固件类型(2B) + 主版本号(2B) + 次版本号(4B)
    fw_type = struct.unpack("<H", ret_val[2:4])[0]
    # 主版本号2字节: 分别作为 major.minor (BCD格式)
    major_ver = ret_val[4]
    minor_ver = ret_val[5]
    # 次版本号4字节: 需反转顺序显示 (协议小端存储)
    minor_bytes = ret_val[6:10][::-1]
    minor_str = "".join(f"{b:02X}" for b in minor_bytes)

    print(f"      固件类型: 0x{fw_type:04X}")
    print(f"      版本号: V{major_ver}.{minor_ver:02d}.{minor_str}")

    test_end_config(ser)
    return f"V{major_ver}.{minor_ver:02d}.{minor_str}"


def test_read_detection_params(ser: serial.Serial):
    """测试读取目标检测参数"""
    print(f"\n  [4] 读取目标检测参数...")
    if not test_enable_config(ser):
        return
    time.sleep(0.1)

    cmd = build_command(CMD_READ_DETECTION_PARAM)
    resp = send_and_wait_ack(ser, cmd)

    resp_cmd, ret_val, err = parse_ack(resp)
    if err or resp_cmd != (CMD_READ_DETECTION_PARAM | 0x0100):
        print(f"      读取参数失败: {err}")
        test_end_config(ser)
        return

    status = struct.unpack("<H", ret_val[:2])[0]
    if status != 0:
        print(f"      失败! 状态码: {status}")
        test_end_config(ser)
        return

    # 4字节配置值
    max_dist = ret_val[2]  # 最远检测距离(m)
    move_dir = ret_val[3]  # 运动方向: 0=远离, 1=靠近, 2=均检测
    min_speed = ret_val[4]  # 最小运动速度(km/h)
    delay = ret_val[5]  # 无目标延迟时间(s)

    dir_map = {0: "只检测远离", 1: "只检测靠近", 2: "均检测"}
    print(f"      最远检测距离: {max_dist} m")
    print(f"      运动方向: {dir_map.get(move_dir, f'未知({move_dir})')}")
    print(f"      最小运动速度: {min_speed} km/h")
    print(f"      无目标延迟时间: {delay} s")

    test_end_config(ser)


def test_read_sensitivity(ser: serial.Serial):
    """测试读取灵敏度参数"""
    print(f"\n  [5] 读取灵敏度参数...")
    if not test_enable_config(ser):
        return
    time.sleep(0.1)

    cmd = build_command(CMD_READ_SENSITIVITY)
    resp = send_and_wait_ack(ser, cmd)

    resp_cmd, ret_val, err = parse_ack(resp)
    if err or resp_cmd != (CMD_READ_SENSITIVITY | 0x0100):
        print(f"      读取灵敏度失败: {err}")
        test_end_config(ser)
        return

    status = struct.unpack("<H", ret_val[:2])[0]
    if status != 0:
        print(f"      失败! 状态码: {status}")
        test_end_config(ser)
        return

    trigger_count = ret_val[2]
    snr_threshold = ret_val[3]
    ext1 = ret_val[4]
    ext2 = ret_val[5]

    print(f"      累积有效触发次数: {trigger_count}")
    print(f"      信噪比阈值等级: {snr_threshold} (值越大越不灵敏)")
    print(f"      扩展参数: {ext1} {ext2}")

    test_end_config(ser)


# ============================================================
#  实时数据读取
# ============================================================
def parse_radar_data(data: bytes):
    """解析雷达输出的探测数据"""
    if len(data) < 10:
        return None, "数据太短"

    if not data.startswith(DATA_HEADER):
        return None, f"帧头错误: {data[:4].hex()}"

    if not data.endswith(DATA_END):
        return None, f"帧尾错误: {data[-4:].hex()}"

    payload_len = struct.unpack("<H", data[4:6])[0]
    payload = data[6 : 6 + payload_len]

    # 当没有目标时，payload 长度为 0（协议默认行为）
    if payload_len == 0:
        return {"target_count": 0, "has_alarm": False, "targets": []}, None

    if len(payload) < 2:
        return None, f"有效载荷太短 ({len(payload)} 字节)"

    target_count = payload[0]
    alarm_info = payload[1]

    expected_payload_len = 2 + target_count * 5
    if len(payload) < expected_payload_len:
        return None, f"有效载荷不完整: 应有{expected_payload_len}字节, 实际{len(payload)}字节"

    targets = []
    for i in range(target_count):
        offset = 2 + i * 5
        t = payload[offset : offset + 5]
        angle_raw = t[0]
        distance = t[1]
        speed_dir = t[2]
        speed_val = t[3]
        snr = t[4]

        angle = -(angle_raw - 0x80)
        dir_str = "靠近" if speed_dir == 1 else "远离" if speed_dir == 0 else f"未知({speed_dir})"

        targets.append(
            {
                "angle": angle,
                "distance": distance,
                "speed": speed_val,
                "direction": dir_str,
                "snr": snr,
            }
        )

    return {
        "target_count": target_count,
        "has_alarm": bool(alarm_info & 0x01),
        "targets": targets,
    }, None


running = False


def read_radar_data_loop(ser: serial.Serial):
    """实时读取并解析雷达数据"""
    global running
    buffer = bytearray()
    running = True
    frame_count = 0
    error_count = 0
    last_heartbeat = time.time()
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0.0

    print(f"\n{'='*60}")
    print(f"  实时雷达数据监控 (按 Ctrl+C 停止)")
    print(f"{'='*60}")
    print(f"  {'目标':>4s} | {'角度(°)':>8s} | {'距离(m)':>8s} | {'速度':>10s} | {'方向':>6s} | {'信噪比':>6s}")
    print(f"{'-'*60}")
    print(f"  >> 天线面(正面)朝向活动区域才可检测到 <<")
    print(f"  >> 帧率受硬件限制 ~1Hz，非串口瓶颈          <<")
    print(f"{'-'*60}")

    try:
        while running:
            now = time.time()
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                buffer.extend(chunk)

                # 查找数据帧头
                while True:
                    idx = buffer.find(DATA_HEADER)
                    if idx < 0:
                        if len(buffer) > 1024:
                            buffer.clear()
                        break

                    # 移除帧头前的无用数据
                    if idx > 0:
                        del buffer[:idx]
                        continue

                    # 至少需要: 帧头(4) + 长度(2) + 帧尾(4) = 10 (无目标最小帧)
                    if len(buffer) < 10:
                        break

                    payload_len = struct.unpack("<H", buffer[4:6])[0]
                    frame_total_len = 4 + 2 + payload_len + 4  # header + len + payload + end

                    if len(buffer) < frame_total_len:
                        break

                    frame = buffer[:frame_total_len]
                    del buffer[:frame_total_len]

                    result, err = parse_radar_data(bytes(frame))
                    if err:
                        error_count += 1
                        if error_count <= 3:
                            print(f"  [解析错误] {err}")
                        continue

                    frame_count += 1
                    fps_counter += 1
                    last_heartbeat = now

                    # 每秒更新 FPS
                    if now - last_fps_time >= 1.0:
                        current_fps = fps_counter / (now - last_fps_time)
                        fps_counter = 0
                        last_fps_time = now

                    if not result["targets"]:
                        print(f"  无目标 | 帧率 {current_fps:.1f}Hz | 已收 {frame_count:>4d}帧 错误{error_count:>2d}")
                    else:
                        for t in result["targets"]:
                            print(
                                f"  {result['target_count']:>4d} |"
                                f"  {t['angle']:>+7d}° |"
                                f"  {t['distance']:>7d}m |"
                                f"  {t['speed']:>7d}km/h |"
                                f"  {t['direction']:>4s} |"
                                f"  SNR={t['snr']:>2d}"
                            )
            else:
                # 无数据超时提示（3 秒无新帧时显示）
                if now - last_heartbeat >= 3.0:
                    print(f"  [等待数据] 已接收 {frame_count} 帧, 共 {error_count} 错误 | {time.strftime('%H:%M:%S')}")
                    last_heartbeat = now
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    print(f"\n{'='*60}")
    print(f"  监控结束: 共接收 {frame_count} 帧, {error_count} 帧解析错误")
    print(f"{'='*60}")


# ============================================================
#  主测试流程
# ============================================================
def optimize_for_detection(ser: serial.Serial):
    """将雷达配置为最佳探测模式：双向检测 + 最低速度阈值 + 最高灵敏度"""
    print(f"\n  [优化] 配置雷达为最佳探测模式...")

    # 先排空可能的数据帧
    if ser.in_waiting:
        _ = ser.read(ser.in_waiting)

    # 使能配置
    cmd = build_command(CMD_ENABLE_CONFIG, b"\x01\x00")
    ser.write(cmd)
    time.sleep(0.3)
    ack = ser.read(1024)
    if FRAME_HEADER not in ack:
        print(f"  [优化] 使能配置失败，重试...")
        ser.write(cmd)
        time.sleep(0.3)
        ack = ser.read(1024)

    # 设置双向检测: 最远100m(0x64), 均检测(0x02), 最小速度0(0x00), 延迟1s(0x01)
    cmd_det = build_command(CMD_SET_DETECTION_PARAM, b"\x64\x02\x00\x01")
    ser.write(cmd_det)
    time.sleep(0.3)
    _ = ser.read(1024)

    # 设置高灵敏度: 触发1次, 信噪比阈值3(更灵敏), 0, 0
    cmd_sens = build_command(CMD_SET_SENSITIVITY, b"\x01\x03\x00\x00")
    ser.write(cmd_sens)
    time.sleep(0.3)
    _ = ser.read(1024)

    # 结束配置
    ser.write(build_command(CMD_END_CONFIG))
    time.sleep(0.3)
    _ = ser.read(1024)

    print(f"  [优化] 已完成: 双向检测 | 最小速度 0km/h | 最高灵敏度")
    time.sleep(0.2)


def run_all_tests(port="/dev/ttyUSB1", baudrate=256000):
    """运行所有雷达测试"""
    print(f"\n{'='*60}")
    print(f"  HLK-LD2451 雷达模块测试")
    print(f"{'='*60}")
    print(f"  端口: {port}")
    print(f"  波特率: {baudrate}")
    print(f"{'='*60}")

    # 连接
    ser = test_connection(port, baudrate)
    if not ser:
        return

    try:
        # 0. 先优化配置为最佳探测模式
        optimize_for_detection(ser)

        # 1. 基本通信 - 使能/结束配置
        print(f"\n--- 基本通信测试 ---")
        test_enable_config(ser)
        time.sleep(0.1)
        test_end_config(ser)

        # 2. 读取固件版本
        print(f"\n--- 固件信息 ---")
        test_read_firmware(ser)

        # 3. 读取检测参数
        print(f"\n--- 检测参数 ---")
        test_read_detection_params(ser)

        # 4. 读取灵敏度参数
        print(f"\n--- 灵敏度参数 ---")
        test_read_sensitivity(ser)

        # 5. 实时数据监控（在探测模式下）
        print(f"\n--- 实时数据监控 ---")
        read_radar_data_loop(ser)

    except KeyboardInterrupt:
        print(f"\n  用户中断")
    except Exception as e:
        print(f"\n  [错误] {type(e).__name__}: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print(f"\n  串口已关闭")


def quick_data_view(port="/dev/ttyUSB1", baudrate=256000):
    """快速模式：只查看实时雷达数据（自动优化配置）"""
    ser = test_connection(port, baudrate)
    if not ser:
        return
    try:
        optimize_for_detection(ser)
        read_radar_data_loop(ser)
    except Exception as e:
        print(f"\n  [错误] {type(e).__name__}: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HLK-LD2451 雷达模块测试工具")
    parser.add_argument("port", nargs="?", default="/dev/ttyUSB1", help="串口号 (默认: /dev/ttyUSB1)")
    parser.add_argument("-b", "--baud", type=int, default=256000, help="波特率 (当前真机: 256000)")
    parser.add_argument("--scan", action="store_true", help="仅扫描串口")
    parser.add_argument("--quick", action="store_true", help="快速模式: 仅查看实时数据")
    parser.add_argument("--firmware", action="store_true", help="仅读取固件版本")
    parser.add_argument("--params", action="store_true", help="仅读取检测参数")
    args = parser.parse_args()

    if args.scan:
        scan_ports()
        sys.exit(0)

    # 扫描并显示端口信息
    scan_ports()

    if args.quick:
        quick_data_view(args.port, args.baud)
    elif args.firmware:
        ser = test_connection(args.port, args.baud)
        if ser:
            test_read_firmware(ser)
            ser.close()
    elif args.params:
        ser = test_connection(args.port, args.baud)
        if ser:
            test_read_detection_params(ser)
            ser.close()
    else:
        run_all_tests(args.port, args.baud)
