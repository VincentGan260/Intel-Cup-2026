import serial
import time

# 端口号和波特率
PORT = '/dev/ttyUSB0'  
BAUD_RATE = 115200

def parse_data(data):
    """解析维特协议中的角度数据包 (0x55 0x53)"""
    for i in range(len(data) - 11):
        # 寻找数据包头 0x55 以及 角度标识 0x53
        if data[i] == 0x55 and data[i+1] == 0x53:
            rollL = data[i+2]
            rollH = data[i+3]
            pitchL = data[i+4]
            pitchH = data[i+5]
            yawL = data[i+6]
            yawH = data[i+7]
            
            # 组合字节并换算为角度
            roll = ((rollH << 8) | rollL) / 32768.0 * 180
            pitch = ((pitchH << 8) | pitchL) / 32768.0 * 180
            yaw = ((yawH << 8) | yawL) / 32768.0 * 180
            
            # 处理超过180度的负数情况
            if roll > 180: roll -= 360
            if pitch > 180: pitch -= 360
            if yaw > 180: yaw -= 360
                
            print(f"实时姿态 -> 滚转(Roll): {roll:6.2f}°, 俯仰(Pitch): {pitch:6.2f}°, 偏航(Yaw): {yaw:6.2f}°")
            break # 找到并解析一个包后就跳出循环，避免重复打印

def main():
    try:
        # 打开串口
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.1)
        print(f"====== 成功连接传感器 {PORT} ======")
        print("正在读取六轴姿态数据，按 Ctrl+C 停止...\n")
        
        while True:
            # 只要缓冲区有数据就读取
            if ser.in_waiting > 0:
                raw_data = ser.read(ser.in_waiting)
                parse_data(raw_data)
            time.sleep(0.02) # 稍微延时，避免CPU占用过高
            
    except serial.SerialException as e:
        print(f"\n[错误] 串口通信失败，请检查连线或权限: {e}")
    except KeyboardInterrupt:
        print("\n[退出] 程序已手动终止")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()
