# IMU横向失稳风险理论与标定方法

## 1. 文档目的

本文区分三类内容：

1. 可以由两轮车运动学或动力学直接推导的量。
2. 需要本车几何、质量、速度或执行延迟才能计算的量。
3. 只能通过本车受控实测标定的阈值。

任何没有完成对应测量的数值只能称为“搜索候选”，不得称为理论安全阈值。

## 2. 理论依据

### 2.1 正常转弯不能以零侧倾为基准

两轮车稳定转弯时需要用侧倾平衡向心加速度。平路、准稳态转弯的简化关系为：

```text
phi_eq = k_turn * atan(v^2 / (gR))
       = k_turn * atan(v * yaw_rate / g)
```

其中：

- `phi_eq`：正常转弯平衡侧倾角。
- `v`：车体纵向速度，单位m/s。
- `R`：转弯半径，单位m。
- `yaw_rate`：横摆角速度，单位rad/s。
- `g`：重力加速度，取`9.80665 m/s²`。
- `k_turn`：IMU横摆正方向与roll正方向的符号映射，只能取`+1`或`-1`，通过左右稳定转弯实测确定。

该关系说明固定`|body_roll|`阈值会把正常转弯误认为失稳。两轮车roll估计研究也使用纵向速度、横摆角速度和IMU信号共同估计转弯姿态，而不是只看一个roll值。

参考：

- [Estimating the Roll Angle for a Two-Wheeled Single-Track Vehicle Using a Kalman Filter](https://doi.org/10.3390/s22228991)
- [A bicycle can be balanced by stochastic optimal feedback control but only with accurate speed estimates](https://doi.org/10.1371/journal.pone.0278961)

### 2.2 应判断相对平衡侧倾残差

先扣除安装中性偏置：

```text
body_roll = roll_raw - roll0
```

再计算相对正常转弯平衡姿态的残差：

```text
roll_error = wrap(body_roll - phi_eq)
```

短时间内的残差扩大速度为：

```text
roll_error_rate = gyro_x - derivative(phi_eq)
outward_rate = sign(roll_error) * roll_error_rate
```

只有`outward_rate > 0`才表示车辆正在偏离当前正常转弯平衡姿态。稳定转弯和正在回正不应因此升级。

### 2.3 时间到边界只是短时预测量

在很短预测窗内近似角速度不变，可定义：

```text
tau_lean =
    (roll_error_critical - |roll_error|) / outward_rate
```

适用条件：

- `outward_rate > 0`。
- 采样新鲜且连续。
- 预测窗足够短，不能外推数秒。
- `roll_error_critical`已经通过本车标定。

`tau_lean`是“一阶恒角速度近似下到达标定边界的剩余时间”，不是完整摔倒时间，也不是保证骑手能够恢复的安全时间。

### 2.4 完整自行车动力学不能用一个通用角度代替

Whipple基准自行车模型的线性横向动力学同时包含roll和steer状态，并依赖车速、轴距、拖曳距、转向轴角度、各部件质量、质心和转动惯量。不同车辆和速度的稳定特性不同。

当前项目没有转向角传感器，也没有完整车辆质量/惯量参数，因此不能声称`35°`由完整自行车动力学模型推导得到。

参考：

- [Linearized dynamics equations for the balance and steer of a bicycle: a benchmark and review](https://bicycle.tudelft.nl/schwab/Publications/meijaard2007linearized.pdf)
- [Auto-Correction of 3D-Orientation of IMUs on Electric Bicycles](https://doi.org/10.3390/s20030589)

### 2.5 预摔研究支持多条件组合，但不提供可直接照搬阈值

公开的电动滑板车预冲击跌倒研究同时使用加速度合量、角速度合量和roll角，并通过参与者数据网格搜索角度与角速度阈值。其结论支持“角度、角速度、时间顺序联合判断”，但传感器安装、车辆结构和动作与本项目不同，数值不能直接复制。

参考：

- [Pre-Impact Fall Detection for E-Scooter Riding Using an IMU](https://doi.org/10.3390/app142210443)

## 3. 当前候选值的正确身份

| 候选 | 当前用途 | 能否称为理论值 |
|---|---|---|
| `roll_error_critical: 30°/35°/40°` | 实车搜索边界 | 不能 |
| `roll_attention: 15°/20°/25°` | 早期关注搜索点 | 不能 |
| `outward_rate: 10/15/20°/s` | 排除缓慢姿态变化的搜索点 | 不能 |
| `attention_persistence: 100/200/300 ms` | 抑制瞬时尖峰的搜索点 | 不能 |
| 高风险3个连续样本 | 约100 Hz下的初始一致性候选 | 不能 |
| `tau_lean: 0.4/0.6/0.8 s` | 尚未确认的搜索范围 | 不能 |

这些候选只用于构建受控实验矩阵，不能写入答辩材料作为安全依据。

## 4. 正式规则的推导和标定顺序

### 4.1 测量安装与信号参数

1. 固定最终安装位置并测量`roll0`。
2. 测量采样周期、时间戳排队延迟和stale分布P95/P99。
3. 对roll和gyro信号确定滤波器及其相位延迟。
4. 更换安装位置后全部重测。

### 4.2 建立正常转弯基线

同步记录：

```text
GPS速度
gyro_z横摆角速度
body_roll
gyro_x
时间戳
```

用`phi_eq=k_turn*atan(v*yaw_rate/g)`计算正常平衡侧倾，再统计：

- `roll_error`的P50/P95/P99。
- `outward_rate`的P50/P95/P99。
- 入弯、稳定转弯、出弯回正的持续时间。
- 不同速度和转弯半径下的分布差异。
- 左转、右转时`gyro_z`和`body_roll`的符号关系，据此冻结`k_turn`。

中风险角度和角速度阈值必须高于正常骑行分布，并通过留出的正常场景验证集检查误报。

### 4.3 建立安全可控的失稳趋势样本

车辆必须有人扶持或使用防倒保护，记录快速侧倾但不落地的片段。不得为了采集数据要求骑手故意摔车。

为每段数据人工标记：

- 正常转弯开始、稳定、回正。
- 异常侧倾开始。
- 扶车人员必须干预的时刻。
- 是否发生颠簸或轮胎打滑。

### 4.4 推导预警时间要求

正式高风险时间边界应满足：

```text
T_warning_required =
    T_sensor_age_P99
  + T_compute_to_motor_P99
  + T_motor_to_perceptible_P99
  + T_rider_corrective_response_quantile
  + T_margin
```

其中前三项可以通过主机和马达实测；骑手纠偏响应必须在安全受控实验中测量。一般摩托车危险感知研究只能证明响应时间受危险出现方式、骑手经验和警告形式影响，不能把其中某个数字直接移植为本项目侧倾纠偏时间。

参考：

- [Young male motorcycle rider perception response times to abrupt- and gradual-onset hazards](https://doi.org/10.1016/j.aap.2021.106519)
- [Connected Motorcycle Consortium Rider Reaction Time studies](https://www.cmc-info.net/rider-reaction-time.html)

### 4.5 以提前量而不是“触发角度”验收

每个异常侧倾事件计算：

```text
warning_lead_time = intervention_required_time - first_high_warning_time
```

正式高风险规则必须同时满足：

- 在干预时刻之前首次报警。
- `warning_lead_time`不低于已测`T_warning_required`。
- 正常转弯验证集不产生不可接受的高风险误报。
- 颠簸和单帧尖峰不触发高风险。
- IMU失效时不影响雷达和视觉预警。

如果候选阈值无法同时满足提前量和误报要求，应判定当前传感器或规则不足，而不是选择一个看起来顺眼的数字。

## 5. GPS或横摆数据失效

没有可用速度或横摆角速度时，无法可靠计算正常转弯平衡倾角`phi_eq`。系统必须标记IMU转弯补偿为degraded，不能用`phi_eq=0`伪装成正常计算结果。

降级策略需要通过实车数据另行选择：

- 只允许非常明确的快速异常侧倾触发IMU通道。
- 或暂时限制IMU最高输出等级，同时保持雷达、视觉正常工作。

两种方案必须比较漏报与误报后再确定。

## 6. 当前结论

当前可以确认的是计算结构，而不是最终数字：

```text
安装偏置修正
→ 正常转弯平衡侧倾补偿
→ roll残差及其扩大速度
→ 短时tau预测
→ 多样本一致性
→ 独立IMU风险事件
→ 与雷达、视觉做max仲裁
```

为支持室外整体验证，IMU已按理论初值以`provisional_pending_vehicle_test`状态进入低/中/高风险仲裁：中风险采用`10°`残差、`5°/s`外倾速度和`150 ms`持续时间；高风险采用`25°`临界残差、`10°/s`外倾速度、`0.8 s`前向预测和连续3样本。GPS或车速不足以可靠执行转弯补偿时，IMU最高限制为中风险。上述数值是可解释的工程起点，不是最终实车标定结论。
