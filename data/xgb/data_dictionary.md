# XGBoost风险数据字段字典

| 字段 | 类型 | 单位/取值 | 含义 | 训练用途 |
|---|---|---|---|---|
| sample_id | 元数据 | 字符串 | 样本唯一编号 | 不输入模型 |
| split | 元数据 | 字符串 | train/validation/test | 不输入模型 |
| scenario_instance_id | 元数据 | 字符串 | 合成场景实例编号 | 不输入模型 |
| scenario_type | 元数据 | 字符串 | 场景类型 | 不输入模型 |
| timestamp_s | 元数据 | s | 场景内时间戳 | 不输入模型 |
| boundary_case | 元数据 | 0/1 | 是否为阈值边界样本 | 不输入模型 |
| gps_speed_kmh | 特征 | km/h | GPS速度，用于IMU转弯补偿 | 输入模型 |
| roll_error_deg | 特征 | deg | 转弯补偿后的横滚误差 | 输入模型 |
| outward_rate_deg_s | 特征 | deg/s | 横滚误差向外增大速度 | 输入模型 |
| acc_delta_signed_mps2 | 特征 | m/s² | 加速度突变量；正负方向均保留 | 输入模型 |
| acc_change_abs_mps2 | 特征 | m/s² | 加速度突变量绝对值 | 输入模型 |
| jerk_abs_mps3 | 特征 | m/s³ | 加速度变化率绝对值 | 输入模型 |
| radar_relative_speed_mps | 特征 | m/s | 项目原始约定：靠近为负 | 输入模型 |
| radar_closing_speed_mps | 特征 | m/s | 接近速度，靠近为正 | 输入模型 |
| radar_ttc_s | 特征 | s | 路径目标TTC；无法计算时缺失 | 输入模型 |
| radar_person_matched | 特征 | 0/1 | 雷达目标是否与视觉行人匹配 | 输入模型 |
| path_object_count | 特征 | 个 | 可行驶路径中的视觉目标数量 | 输入模型 |
| max_path_bottom_ratio | 特征 | 0-1 | 路径目标框底部相对图像高度 | 输入模型 |
| box_growth_rate_per_s | 特征 | 1/s | 目标框尺度增长率 | 输入模型 |
| growth_duration_s | 特征 | s | 目标持续增大的时间 | 输入模型 |
| visual_tau_s | 特征 | s | 视觉目标尺度增长推算的时间尺度 | 输入模型 |
| rule_score | 标签辅助 | 0-1 | 各模态规则分数最大值 | 禁止输入模型 |
| risk_label | 标签 | 0/1/2 | 0低、1中、2高 | 训练目标 |
| trigger_reason | 标签辅助 | 字符串 | 产生最大规则分数的模态 | 禁止输入模型 |

完整训练字段以 `feature_config.json` 中的 `feature_columns` 为准。
空白的 TTC/距离表示当前窗口无法形成有效估计，不得替换为 0。
