# 交给 DK-2500 Codex 的自动部署指令

将下面整个代码块复制到 DK-2500 上的 Codex。该任务会停用旧服务，但不会删除、
修改或重新安装旧规则。

```text
请在这台 DK-2500 上完成独立 XGBoost 风险服务部署，并一直执行到验证完成。

安全边界：
1. 工作目录固定为 /home/intelcup/Intel-Cup-2026。
2. 不得修改 src/fusion/physical_risk_rule.py、src/fusion/imu_warning_rule.py、
   src/fusion/vision_warning_rule.py、configs/warning_rules.yaml 或
   deploy/edge/rider-dashboard.service。
3. 不得启动、导入或测试任何真实电机控制。新服务必须保持 motor_control=false。
4. 如果仓库存在未提交修改，先停止并向我报告，不得 reset、stash 或覆盖。
5. 如果新服务部署失败，保持 rider-xgb 和 rider-dashboard 两个服务都停止，
   不要自动恢复会驱动真实电机的旧服务。

按顺序执行：

1. 进入仓库并检查：
   cd /home/intelcup/Intel-Cup-2026
   git status --short
   只有工作树干净才继续。

2. 获取并切换发布分支：
   git fetch origin codex/standalone-xgboost-risk
   如果本地已有 codex/standalone-xgboost-risk，则 git switch 到该分支；
   否则基于 origin/codex/standalone-xgboost-risk 创建同名跟踪分支。
   然后执行 git pull --ff-only origin codex/standalone-xgboost-risk。

3. 验证隔离性：
   - 确认 run_xgb_dashboard.py 存在。
   - 确认 deploy/edge/rider-xgb.service 存在。
   - 搜索 run_xgb_dashboard.py 和 src/risk_ml，确认没有导入
     physical_risk_rule、imu_warning_rule、vision_warning_rule、
     warning_system 或 src.actuator。
   - 确认 rider-xgb.service 中没有 --enable-risk-rule、--motor-mode
     或 --confirm-motor-real。

4. 使用现有 Conda 环境安装仅运行时依赖：
   /home/intelcup/miniconda3/envs/intel/bin/python \
     -m pip install -r requirements/risk_ml_runtime.txt

5. 先执行不打开硬件的模型冒烟测试：
   /home/intelcup/miniconda3/envs/intel/bin/python \
     scripts/risk_ml/smoke_test_runtime.py
   必须看到 status=ok、feature_count=31 且 accuracy>=0.95。

6. 停用旧服务，避免摄像头和串口冲突：
   sudo systemctl disable --now rider-dashboard.service
   确认 systemctl is-active rider-dashboard.service 返回 inactive。

7. 在新服务启动前执行硬件预检：
   /home/intelcup/miniconda3/envs/intel/bin/python \
     scripts/preflight_dk2500.py --profile dk2500 --loops 30 --vision
   保存完整输出。如果预检明确失败，不启动新服务，保持两个服务停止并报告。

8. 安装并启动新服务：
   sudo cp deploy/edge/rider-xgb.service /etc/systemd/system/rider-xgb.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now rider-xgb.service

9. 最多等待 45 秒，每 3 秒检查一次：
   - systemctl is-active rider-xgb.service
   - curl --fail http://127.0.0.1:8001/api/health
   健康接口必须同时包含：
   decision_engine="xgboost-only"
   motor_control=false

10. 读取一次 http://127.0.0.1:8001/api/state，确认：
    - old_rules_loaded=false
    - motor_control=false
    - runtime.feature_count=31
    - features 中有 31 个字段
    - prediction 最迟在窗口预热后出现

11. 最终向我报告：
    - 当前 Git commit
    - 冒烟测试结果
    - 硬件预检结果
    - 两个 systemd 服务的 enabled/active 状态
    - /api/health 与 /api/state 的关键字段
    - journalctl -u rider-xgb.service -n 80 --no-pager 中是否有错误
    - DK-2500 的局域网 IP 和访问地址 http://<IP>:8001

不要只给命令说明；请实际执行、检查输出并完成验证。需要 sudo 或联网授权时，
使用 Codex 的审批流程请求我授权。
```
