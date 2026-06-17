"""视觉管线适配器：将现有 `src/vision/` 输出转换为融合模块 `VisionData`。

职责：
  1. 加载视觉管线配置，通过现有工厂函数构建检测器和分割器
  2. 调用 `VisionPipeline.process()` 得到 `VisionResult`
  3. 转换为 `src/fusion/data_types.VisionData`
  4. 所有失败路径返回无效 VisionData，不抛异常

使用方只需调用一次 `process(frame)` 即可获得对齐后的数据。

注意：视觉模块依赖的 openvino 等库只在 start() 时按需导入，
      避免模块级 import 导致无 openvino 环境崩溃。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.fusion.data_types import VisionData, VisionObject, now


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class VisionAdapter:
    """视觉适配器，将现有视觉管线输出转为融合模块 VisionData。

    Args:
        pipeline_config_path: vision_pipeline.yaml 路径
        vision_enabled: 是否启用视觉。设为 False 时 process() 直接返回空数据
        use_camera: 是否从摄像头读取帧（process 时忽略传入的 frame）
        camera_id: 摄像头设备编号
    """

    def __init__(
        self,
        pipeline_config_path: str = "configs/vision/vision_pipeline.yaml",
        vision_enabled: bool = True,
        use_camera: bool = False,
        camera_id: int = 0,
    ) -> None:
        self.pipeline_config_path = pipeline_config_path
        self.vision_enabled = vision_enabled
        self.use_camera = use_camera
        self.camera_id = camera_id
        self._cap = None
        self._pipeline = None
        self._config: Optional[dict] = None
        self._latest: Optional[VisionData] = None

    # ---- 生命周期 ----

    def start(self) -> None:
        """初始化视觉管线（加载模型 + 可选打开摄像头）。

        视觉库（openvino, cv2）在此方法内按需导入，
        避免模块级 import 导致无依赖环境崩溃。
        """
        if not self.vision_enabled:
            print("[VisionAdapter] 视觉未启用")
            return

        try:
            # 按需导入视觉模块
            import cv2

            from src.vision.common.preprocess import read_image_bgr, load_image_bgr_from_source
            from src.vision.perception.vision_pipeline import VisionPipeline
            from src.vision.detection.detector import build_detector_from_config
            from src.vision.segmentation.segmenter import build_segmenter_from_config

            # 1. 加载配置
            config_path = Path(self.pipeline_config_path)
            if not config_path.is_absolute():
                config_path = PROJECT_ROOT / config_path
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)

            # 2. 构建检测器
            det_cfg_path = self._config.get("detection_config", "configs/vision/detection.yaml")
            det_path = Path(det_cfg_path)
            if not det_path.is_absolute():
                det_path = PROJECT_ROOT / det_path
            with open(det_path, "r", encoding="utf-8") as f:
                det_config = yaml.safe_load(f)
            detector = build_detector_from_config(det_config, project_root=PROJECT_ROOT)

            # 3. 构建分割器（可选）
            segmenter = None
            enable_seg = self._config.get("enable_segmentation", True)
            if enable_seg:
                seg_cfg_path = self._config.get(
                    "segmentation_config", "configs/vision/segmentation_openvino.yaml"
                )
                seg_path = Path(seg_cfg_path)
                if not seg_path.is_absolute():
                    seg_path = PROJECT_ROOT / seg_path
                if seg_path.exists():
                    with open(seg_path, "r", encoding="utf-8") as f:
                        seg_config = yaml.safe_load(f)
                    segmenter = build_segmenter_from_config(seg_config, project_root=PROJECT_ROOT)

            # 4. 构建管线
            self._pipeline = VisionPipeline(
                detector=detector,
                segmenter=segmenter,
                enable_segmentation=enable_seg,
            )

            # 5. 打开摄像头（可选）
            if self.use_camera:
                self._cap = cv2.VideoCapture(self.camera_id)
                if not self._cap.isOpened():
                    print(f"[VisionAdapter] 无法打开摄像头 {self.camera_id}，降级为图片模式")
                    self.use_camera = False
                else:
                    print(f"[VisionAdapter] 已打开摄像头 {self.camera_id}")

            print("[VisionAdapter] 视觉管线初始化成功")
            if not enable_seg:
                print("[VisionAdapter]   → 分割已禁用（仅检测）")

        except ImportError as e:
            print(f"[VisionAdapter] 依赖库未安装: {e}")
            print("[VisionAdapter] 降级至 vision_enabled=False")
            self.vision_enabled = False
            self._pipeline = None

        except Exception as e:
            print(f"[VisionAdapter] 初始化失败: {e}")
            print("[VisionAdapter] 降级至 vision_enabled=False")
            self.vision_enabled = False
            self._pipeline = None

    def stop(self) -> None:
        """释放视觉资源。"""
        if self._cap is not None:
            try:
                import cv2
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            print("[VisionAdapter] 摄像头已释放")
        self._pipeline = None
        print("[VisionAdapter] 已停止")

    # ---- 核心接口 ----

    def process(self, frame: Optional[np.ndarray] = None) -> VisionData:
        """处理一帧图像并返回对齐后的 VisionData。

        Args:
            frame: BGR 图像 (H, W, 3)。为 None 时从摄像头读取（需 use_camera=True）。

        Returns:
            始终返回 VisionData 实例，不抛异常。
        """
        ts = now()
        result = VisionData(timestamp=ts, valid=False)

        if not self.vision_enabled or self._pipeline is None:
            self._latest = result
            return result

        try:
            import cv2

            # 获取帧
            if frame is None:
                if self._cap is not None:
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        raise RuntimeError("摄像头读取失败")
                else:
                    raise ValueError("未提供 frame，也未启用摄像头")

            # 视觉管线推理
            vision_result = self._pipeline.process(frame)
            h, w = frame.shape[:2]

            # 转换为 VisionData
            objects = []
            person_count = 0
            vehicle_count = 0
            max_conf = 0.0

            for det in vision_result.detections:
                obj = VisionObject(
                    class_name=det.class_name,
                    risk_class=det.risk_class,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    in_drivable_area=det.in_drivable_area,
                    visual_risk=det.visual_risk or 0.0,
                )
                objects.append(obj)

                max_conf = max(max_conf, det.confidence)
                if det.risk_class == "pedestrian":
                    person_count += 1
                elif det.risk_class in ("motor_vehicle", "non_motor_vehicle"):
                    vehicle_count += 1

            # 可行驶区域比例
            drivable_ratio = 0.0
            if vision_result.segmentation is not None and vision_result.segmentation.drivable_ratio is not None:
                drivable_ratio = vision_result.segmentation.drivable_ratio
            elif vision_result.drivable_mask is not None:
                drivable_ratio = float(
                    np.count_nonzero(vision_result.drivable_mask)
                ) / float(max(1, vision_result.drivable_mask.size))

            result = VisionData(
                timestamp=ts,
                valid=True,
                objects=objects,
                person_count=person_count,
                vehicle_count=vehicle_count,
                max_confidence=max_conf,
                drivable_area_ratio=round(drivable_ratio, 4),
                max_visual_risk=vision_result.max_visual_risk,
            )

        except Exception as e:
            print(f"[VisionAdapter] 处理异常: {e}")

        self._latest = result
        return result

    def get_latest(self) -> Optional[VisionData]:
        """返回最近一次 process() 的结果。"""
        return self._latest
