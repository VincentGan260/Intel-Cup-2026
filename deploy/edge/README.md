# DK2500 deployment

This directory contains the systemd service used after the final hardware test.
The service assumes:

- project: `/home/intelcup/Intel-Cup-2026`
- Conda environment: `/home/intelcup/miniconda3/envs/intel`
- device id: `bike-001`
- cloud: `http://124.70.108.34`

## Install after hardware validation

```bash
cd /home/intelcup/Intel-Cup-2026
git pull
sudo cp deploy/edge/rider-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rider-dashboard.service
```

## Verify

```bash
systemctl status rider-dashboard.service --no-pager
journalctl -u rider-dashboard.service -n 100 --no-pager
curl http://127.0.0.1:8000/api/health
```

The dashboard publishes `_riderguardian._tcp` over Bonjour/mDNS so the iOS app
can discover its current hotspot address. Install `requirements/dashboard.txt`
on the board and ensure UDP port 5353 is not blocked. Use `--no-bonjour` only
when local app discovery is intentionally disabled.

The service runs real GPS/radar input, vision inference, cloud ride-sample upload,
and 60-second raw video segmentation. Do not enable it before checking the
camera and serial-device configuration on the board.

## Standalone XGBoost experiment

The independent, no-motor XGBoost service is documented in
[`docs/guide/xgboost_dk2500.md`](../../docs/guide/xgboost_dk2500.md).
It uses `rider-xgb.service` on port 8001 and conflicts with this rule-based
service so both processes cannot claim the same camera and serial devices.
