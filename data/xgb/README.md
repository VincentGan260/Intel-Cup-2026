# XGBoost合成风险数据集

这是一个无真实样本阶段的规则监督数据集，用于跑通 XGBoost 三分类流程。

文件：

- `train.csv`：5250条
- `validation.csv`：1125条
- `test.csv`：1125条
- `feature_config.json`：训练目标、特征白名单和泄漏字段
- `risk_rules.md`：标签规则
- `data_dictionary.md`：字段说明
- `qa_report.json`：生成后的自动校验结果

风险标签：`0=低，1=中，2=高`。所有数据均为合成数据，不能用于证明真实道路泛化能力。
验证集和测试集提高了阈值边界样本比例，用于检查模型在 0.35/0.70、
TTC、视觉增长和IMU阈值附近的稳定性。
