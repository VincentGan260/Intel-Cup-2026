"""传感器模块：封装 GPS / IMU / 毫米波雷达的读取逻辑。

每个 Reader 均支持 mode="real"（真实串口）和 mode="mock"（模拟数据）两种模式，
输出使用 src/fusion/data_types.py 中定义的数据类。
"""
