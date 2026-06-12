"""检查 OpenVINO 可用设备并测试性能"""
import openvino as ov

def main():
    core = ov.Core()
    
    print("=== OpenVINO 可用设备 ===")
    devices = core.available_devices
    for device in devices:
        print(f"📱 {device}")
    
    print("\n=== 设备详情 ===")
    for device in devices:
        try:
            props = core.get_property(device, ov.properties.all())
            print(f"\n{device}:")
            for key, value in props.items():
                if "PERFORMANCE" in str(key) or "SUPPORTED" in str(key) or "FULL_DEVICE_NAME" in str(key):
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  获取属性失败: {e}")


if __name__ == "__main__":
    main()
