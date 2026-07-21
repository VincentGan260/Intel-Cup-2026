# White-tile segmentation deployment

The production pipeline reads `segmentation_model` from
`configs/vision/vision_pipeline.yaml` when the vision process starts.

- `1`: competition white-tile DeepLabV3-MobileNetV3 FP16 model
- `2`: original OpenVINO road-segmentation-adas FP16 model

Show the active profile:

```powershell
python scripts/vision/switch_segmentation_model.py
```

Switch to the white-tile model:

```powershell
python scripts/vision/switch_segmentation_model.py 1
```

Switch to the original ADAS model:

```powershell
python scripts/vision/switch_segmentation_model.py 2
```

Restart the vision process after switching. Both production profiles use
OpenVINO `FP16@GPU`; the competition host must expose an Intel GPU through the
OpenVINO GPU plugin. Confirm it before the demonstration:

```powershell
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

The list should contain `GPU`. The white-tile deployment files are:

```text
models/openvino/white-tile-road-fp16/white-tile-road.xml
models/openvino/white-tile-road-fp16/white-tile-road.bin
```
