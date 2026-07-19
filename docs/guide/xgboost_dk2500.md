# DK-2500 独立 XGBoost 风险服务

该服务用于在一台 DK-2500 上停用旧规则服务后，独立测试三分类
XGBoost 模型。它直接打开摄像头、雷达、GPS 和 IMU，不读取旧规则结果，
不导入规则模块，也不创建电机控制器。

## 隔离边界

- 旧服务：`rider-dashboard.service`，原文件保持不变。
- 新服务：`rider-xgb.service`，端口 `8001`。
- 新入口：`run_xgb_dashboard.py`。
- 模型日志：`data/xgb_live/risk_predictions.jsonl`。
- 新服务声明 `Conflicts=rider-dashboard.service`，避免串口和摄像头被重复打开。
- 当前模型由合成规则标签训练，只允许观察输出，不允许控制真实执行器。

## 首次部署

```bash
cd /home/intelcup/Intel-Cup-2026
git fetch origin
git switch codex/standalone-xgboost-risk
git pull --ff-only origin codex/standalone-xgboost-risk

/home/intelcup/miniconda3/envs/intel/bin/python \
  -m pip install -r requirements/risk_ml_runtime.txt

/home/intelcup/miniconda3/envs/intel/bin/python \
  scripts/risk_ml/smoke_test_runtime.py

sudo systemctl disable --now rider-dashboard.service
sudo cp deploy/edge/rider-xgb.service /etc/systemd/system/rider-xgb.service
sudo systemctl daemon-reload
sudo systemctl enable --now rider-xgb.service
```

## 验证

```bash
systemctl is-enabled rider-dashboard.service
systemctl is-active rider-dashboard.service
systemctl status rider-xgb.service --no-pager
curl --fail http://127.0.0.1:8001/api/health
journalctl -u rider-xgb.service -n 100 --no-pager
```

网页地址：

```text
http://<DK2500-IP>:8001
```

健康接口必须显示：

```json
{
  "decision_engine": "xgboost-only",
  "motor_control": false
}
```

## 回退旧服务

```bash
sudo systemctl disable --now rider-xgb.service
sudo systemctl enable --now rider-dashboard.service
```

回退只切换服务，不需要修改规则配置或规则代码。

如需让 DK-2500 上的 Codex 自动执行完整部署与验证，使用
[`docs/prompts/dk2500_xgb_deploy_codex.md`](../prompts/dk2500_xgb_deploy_codex.md)。
