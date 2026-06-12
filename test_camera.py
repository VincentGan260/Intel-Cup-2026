"""测试摄像头是否正常工作"""
import cv2

def test_camera(index=0):
    """测试指定索引的摄像头"""
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"❌ 摄像头索引 {index} 无法打开")
        return False
    
    print(f"✅ 摄像头索引 {index} 已成功打开")
    
    # 尝试读取一帧
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"📷 图像尺寸: {width} x {height}")
        print(f"🖼️  图像通道: {frame.shape[2] if len(frame.shape) == 3 else '灰度'}")
        print("✅ 摄像头工作正常！")
        result = True
    else:
        print("⚠️ 无法读取图像帧")
        result = False
    
    cap.release()
    return result

if __name__ == "__main__":
    print("🔍 正在检测可用摄像头...")
    
    # 测试常见的摄像头索引
    found = False
    for i in range(5):
        if test_camera(i):
            found = True
            break
    
    if not found:
        print("\n❌ 未找到可用摄像头")
        print("\n💡 可能的解决方案:")
        print("1. 检查摄像头是否正确连接")
        print("2. 检查摄像头权限: sudo chmod 666 /dev/video*")
        print("3. 检查是否被其他程序占用")
        print("4. 尝试不同的摄像头索引")
