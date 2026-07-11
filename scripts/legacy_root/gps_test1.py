import serial
import pynmea2

# 替换为你最终测试成功的串口号，比如 '/dev/ttyS1' 或 '/dev/ttyS2'
SERIAL_PORT = '/dev/ttyS5' 
BAUD_RATE = 9600

def read_gps_data():
    try:
        # 打开串口，设置 1 秒超时
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"成功连接至 GPS 模块 ({SERIAL_PORT})，等待有效定位数据...\n")

        while True:
            # 读取一行原始串口数据，并进行 ASCII 解码（忽略由于信号干扰产生的乱码）
            raw_line = ser.readline().decode('ascii', errors='ignore').strip()

            # 只处理以 $ 开头的标准 NMEA 报文
            if raw_line.startswith('$GPGGA') or raw_line.startswith('$GNGGA'):
                try:
                    # 解析 GGA 报文（包含经纬度和搜星状态）
                    msg = pynmea2.parse(raw_line)
                    # num_sats 搜星数大于 0 时，数据才相对可靠
                    if int(msg.num_sats) > 0:
                        print(f"[位置] 时间: {msg.timestamp} | 纬度: {msg.latitude:.6f} | 经度: {msg.longitude:.6f} | 搜星数: {msg.num_sats}")
                    else:
                        print("[等待] 模块已连接，正在露天环境中搜星...")
                except pynmea2.ParseError:
                    continue  # 忽略解析错误的残缺报文

            elif raw_line.startswith('$GPRMC') or raw_line.startswith('$GNRMC'):
                try:
                    # 解析 RMC 报文（包含速度和航向信息）
                    msg = pynmea2.parse(raw_line)
                    if msg.status == 'A': # 'A' 表示数据有效
                        # 速度原始单位是节(knots)，转换为公里/小时(km/h)
                        speed_kmh = float(msg.spd_over_grnd) * 1.852
                        print(f"[运动] 速度: {speed_kmh:.2f} km/h | 真北航向: {msg.true_course}°")
                except pynmea2.ParseError:
                    continue

    except serial.SerialException as e:
        print(f"串口错误: 无法打开 {SERIAL_PORT}。请检查连线、端口号或权限。详细信息: {e}")
    except KeyboardInterrupt:
        print("\n数据读取已手动终止。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    read_gps_data()
