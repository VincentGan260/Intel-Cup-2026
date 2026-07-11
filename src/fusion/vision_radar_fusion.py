"""视觉-雷达目标级融合（综合风险）。

设计见 docs/视觉雷达融合设计.md。核心思想：
  · 并集保 recall + 几何门控保 precision；类别无关，理论主线=证据栅格 Free/Occupied/Unknown。
  · 三类目标：视觉∩雷达（用雷达真实距离/TTC）、仅视觉（视觉估距）、仅雷达（未知障碍物，几何主导）。

上游模型（本项目选型）：检测=微调 yolo26n v2，分割=road-adas。
本模块**模型无关**：只消费 VisionResult（检测框 + 可行驶掩码）与 RadarData，不做任何模型推理，
全程纯算术 + 在已算好的掩码上查表 → 端侧零额外推理开销。

输入坐标约定：
  · 检测框 bbox = (x1,y1,x2,y2) 像素，原图坐标。
  · 雷达 angle_deg ∈ [-90,90]，0=正前，正=右侧（与 RadarTarget 一致）。
  · relative_speed_mps < 0 表示接近（与 RadarTarget 一致）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.fusion.data_types import RadarData, RadarTarget
from src.vision.common.types import BBox, DetectionResult, VisionResult
from src.vision.risk.visual_risk import (
    DEFAULT_VISUAL_RISK_CONFIG,
    VisualRiskConfig,
    calculate_class_risk,
    calculate_confidence_factor,
    calculate_drivable_risk,
    calculate_lateral_risk,
    calculate_size_risk,
)


# ============================================================
#  配置（★ 标记项需用真实硬件/骑行数据标定，先用合理默认）
# ============================================================


@dataclass(frozen=True)
class VisionRadarFusionConfig:
    # —— 几何/关联 ——
    camera_fov_h_deg: float = 90.0       # ★ 相机水平视场角（待标定）
    radar_half_fov_deg: float = 15.0     # LD2451 水平探测角 ±15°（手册规格，硬件天然只看正前锥）
    flip_radar_angle: bool = False       # ★ 前装时若发现雷达 +角度与相机左右镜像，置 True 翻转符号
    assoc_angle_tol_deg: float = 6.0     # 雷达角(1°分辨率) ↔ 视觉框方位 的关联容差
    corridor_half_deg: float = 8.0       # 行进走廊半角，其内视为完全在途；8°→15°线性衰减

    # —— 运动学风险 ——
    ttc_safe_sec: float = 5.0            # TTC≥该值风险≈0
    dist_near_m: float = 2.0             # 距离风险：≤该值饱和为 1
    dist_far_m: float = 30.0             # ≥该值风险≈0（雷达可达 100m，但电动车近场为重，可调）

    # —— 仅雷达未知障碍物 ——
    require_on_road: bool = True         # 路面门控：未知障碍须投影到路面才计风险
    on_road_min_ratio: float = 0.15      # 投影列带内路面像素占比阈值
    persist_frames: int = 3              # 持续 N 帧达满可信度
    track_max_misses: int = 3            # 连续丢失 N 帧后删除轨迹


DEFAULT_FUSION_CONFIG = VisionRadarFusionConfig()


# ============================================================
#  输出结构
# ============================================================


@dataclass
class FusedObject:
    """融合后的单个障碍物。"""

    source: str                          # "vision_radar" | "vision" | "radar"
    risk: float                          # 综合风险 [0,1]
    risk_class: str                      # 视觉类别；仅雷达为 "unknown"
    bbox: Optional[BBox] = None          # 仅雷达为 None
    distance_m: float = -1.0             # 雷达真实距离；仅视觉为 -1
    relative_speed_mps: float = 0.0       # 雷达相对速度；仅视觉为 0
    ttc_sec: float = -1.0                # 接近时的 TTC；否则 -1
    angle_deg: Optional[float] = None    # 方位角
    on_road: Optional[bool] = None       # 是否投影在路面（仅雷达相关）
    persist: float = 1.0                 # 持续可信度 [0,1]


@dataclass
class FusionOutput:
    objects: List[FusedObject] = field(default_factory=list)
    max_risk: float = 0.0                # 场景级 R_obs（喂给 risk_model）
    n_vision_radar: int = 0
    n_vision_only: int = 0
    n_radar_only: int = 0                # 仅雷达「未知障碍物」（被路面门控保留的）


# ============================================================
#  几何工具（针孔模型，像素列 ↔ 方位角）
# ============================================================


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def bbox_bearing_deg(bbox: BBox, image_w: int, fov_h_deg: float) -> float:
    """检测框水平中心 → 方位角（度）。针孔：angle=atan(2u·tan(fov/2))，u∈[-0.5,0.5]。"""
    if image_w <= 0:
        return 0.0
    cx = (bbox[0] + bbox[2]) / 2.0
    u = cx / image_w - 0.5
    return math.degrees(math.atan(2.0 * u * math.tan(math.radians(fov_h_deg / 2.0))))


def bearing_to_col(angle_deg: float, image_w: int, fov_h_deg: float) -> int:
    """方位角 → 像素列（bbox_bearing_deg 的逆）。"""
    half = math.tan(math.radians(fov_h_deg / 2.0))
    if half <= 0:
        return image_w // 2
    u = math.tan(math.radians(angle_deg)) / (2.0 * half)
    return int(round((u + 0.5) * image_w))


def lateral_factor_from_angle(angle_deg: float, cfg: VisionRadarFusionConfig) -> float:
    """在途程度（仅雷达用）：走廊半角内=1，到雷达视场边缘(±15°)线性衰减到 0。

    用雷达自身 ±15° 视场作为外边界（而非相机 FOV）——雷达物理上只看正前锥，
    超出即无回波，故在 [corridor_half, radar_half_fov] 区间做有意义的横向区分。
    """
    a = abs(angle_deg)
    outer = cfg.radar_half_fov_deg
    if a <= cfg.corridor_half_deg:
        return 1.0
    if a >= outer:
        return 0.0
    return _clamp01((outer - a) / (outer - cfg.corridor_half_deg))


def kinematic_risk(target: RadarTarget, cfg: VisionRadarFusionConfig) -> tuple[float, float]:
    """运动学风险与 TTC。接近时用 TTC，否则用距离。返回 (risk, ttc)（ttc=-1 表示非接近）。"""
    v_rel = target.relative_speed_mps
    if v_rel < 0:  # 接近
        v = -v_rel
        if v > 0.01:
            ttc = target.distance_m / v
            return _clamp01(1.0 - min(ttc, cfg.ttc_safe_sec) / cfg.ttc_safe_sec), ttc
    # 非接近 → 距离风险
    d = target.distance_m
    if d <= cfg.dist_near_m:
        return 1.0, -1.0
    if d >= cfg.dist_far_m:
        return 0.0, -1.0
    return _clamp01((cfg.dist_far_m - d) / (cfg.dist_far_m - cfg.dist_near_m)), -1.0


def is_on_road(
    angle_deg: float,
    drivable_mask: Optional[np.ndarray],
    image_w: int,
    cfg: VisionRadarFusionConfig,
) -> Optional[bool]:
    """杠杆1：把雷达方位投到「已算好的」可行驶掩码上查表，判断该方位前方是否为路面。

    取该方位对应列附近一条竖带、在画面下半部（地面区域）的路面像素占比。
    无掩码时返回 None（未知，不门控）。
    """
    if drivable_mask is None or drivable_mask.size == 0:
        return None
    h, w = drivable_mask.shape[:2]
    col = bearing_to_col(angle_deg, image_w, cfg.camera_fov_h_deg)
    # 列带半宽取图像宽的 3%，至少 3 像素
    half_band = max(3, int(0.03 * w))
    c0 = max(0, col - half_band)
    c1 = min(w, col + half_band + 1)
    if c1 <= c0:
        return False
    region = drivable_mask[int(h * 0.5):, c0:c1]
    if region.size == 0:
        return False
    return bool((region > 0).mean() >= cfg.on_road_min_ratio)


# ============================================================
#  持续性跟踪（杠杆3：抗单帧杂波闪烁）
# ============================================================


@dataclass
class _Track:
    angle_deg: float
    distance_m: float
    hits: int = 1
    misses: int = 0


class _PersistenceTracker:
    """按方位角对雷达目标做轻量跟踪，输出持续可信度 ∈ [0,1]。"""

    def __init__(self, cfg: VisionRadarFusionConfig) -> None:
        self.cfg = cfg
        self._tracks: List[_Track] = []

    def update(self, targets: List[RadarTarget]) -> List[float]:
        """更新轨迹，返回与 targets 一一对应的持续可信度。"""
        matched_idx = set()
        persist = [0.0] * len(targets)
        for i, t in enumerate(targets):
            best, best_d = None, self.cfg.assoc_angle_tol_deg
            for tr in self._tracks:
                d = abs(tr.angle_deg - t.angle_deg)
                if d < best_d:
                    best, best_d = tr, d
            if best is not None:
                best.hits += 1
                best.misses = 0
                best.angle_deg = 0.5 * best.angle_deg + 0.5 * t.angle_deg  # 平滑
                best.distance_m = t.distance_m
                matched_idx.add(id(best))
            else:
                best = _Track(angle_deg=t.angle_deg, distance_m=t.distance_m)
                self._tracks.append(best)
                matched_idx.add(id(best))
            persist[i] = min(1.0, best.hits / max(1, self.cfg.persist_frames))
        # 未命中的轨迹老化
        survivors = []
        for tr in self._tracks:
            if id(tr) not in matched_idx:
                tr.misses += 1
                if tr.misses > self.cfg.track_max_misses:
                    continue
            survivors.append(tr)
        self._tracks = survivors
        return persist


# ============================================================
#  融合主体
# ============================================================


class VisionRadarFusion:
    """视觉-雷达目标级融合器（有状态：持续性跟踪跨帧）。"""

    def __init__(
        self,
        cfg: VisionRadarFusionConfig = DEFAULT_FUSION_CONFIG,
        visual_cfg: VisualRiskConfig = DEFAULT_VISUAL_RISK_CONFIG,
    ) -> None:
        self.cfg = cfg
        self.vcfg = visual_cfg
        self._tracker = _PersistenceTracker(cfg)

    def fuse_vision_result(
        self, vision: Optional[VisionResult], radar: RadarData
    ) -> Optional[FusionOutput]:
        """便捷入口：从 VisionResult 的分割掩码自动取图像尺寸后融合。

        - 正常：有分割掩码 → 全融合（视觉+雷达）。
        - 视觉失效兜底（B 方案）：vision 为 None / 无检测框且无掩码 → **纯雷达**模式，
          仅输出"仅雷达未知障碍"的 R_obs（无图像也能算：风险=运动学×横向角度×持续性，
          无掩码时跳过路面门控、保守保留），保证摄像头/模型挂掉时雷达仍报警。
        - 有检测框但无掩码（分割关闭，拿不到图像尺寸无法关联）→ 返回 None，调用方退回纯视觉。
        供管线一行接入：`out = fuser.fuse_vision_result(vres, radar)`。
        """
        if vision is None:
            vision = VisionResult()
        mask = vision.drivable_mask if vision.drivable_mask is not None else (
            vision.segmentation.drivable_mask if vision.segmentation else None)
        if mask is not None:
            h, w = mask.shape[:2]
            return self.update(vision, radar, w, h)
        if not vision.detections:
            return self.update(vision, radar, 0, 0)   # 纯雷达兜底
        return None

    def update(
        self,
        vision: VisionResult,
        radar: RadarData,
        image_w: int,
        image_h: int,
    ) -> FusionOutput:
        cfg = self.cfg
        dets = list(vision.detections)
        mask = vision.drivable_mask if vision.drivable_mask is not None else (
            vision.segmentation.drivable_mask if vision.segmentation else None)

        radar_ok = bool(radar and radar.valid)
        targets = list(radar.targets) if radar_ok else []
        persist = self._tracker.update(targets) if targets else []
        # 前装安装可能导致雷达左右与相机镜像 → 统一到「相机方位」坐标
        eff_angle = [(-t.angle_deg if cfg.flip_radar_angle else t.angle_deg) for t in targets]

        # 1) 关联：每个雷达目标找方位最近的检测框（容差内）
        det_bearing = [bbox_bearing_deg(d.bbox, image_w, cfg.camera_fov_h_deg) for d in dets]
        det_matched_target: List[Optional[int]] = [None] * len(dets)
        target_matched = [False] * len(targets)
        for ti in range(len(targets)):
            best_di, best_d = None, cfg.assoc_angle_tol_deg
            for di in range(len(dets)):
                if det_matched_target[di] is not None:
                    continue
                d = abs(det_bearing[di] - eff_angle[ti])
                if d < best_d:
                    best_di, best_d = di, d
            if best_di is not None:
                det_matched_target[best_di] = ti
                target_matched[ti] = True

        out = FusionOutput()

        # 2) 逐检测框：视觉∩雷达 或 仅视觉
        for di, det in enumerate(dets):
            ti = det_matched_target[di]
            if ti is not None:  # 视觉∩雷达：用雷达运动学替换接近度
                k_kin, ttc = kinematic_risk(targets[ti], cfg)
                risk = self._matched_risk(det, k_kin, image_w, image_h)
                out.objects.append(FusedObject(
                    source="vision_radar", risk=risk, risk_class=det.risk_class,
                    bbox=det.bbox, distance_m=targets[ti].distance_m,
                    relative_speed_mps=targets[ti].relative_speed_mps, ttc_sec=ttc,
                    angle_deg=eff_angle[ti], persist=persist[ti]))
                out.n_vision_radar += 1
            else:  # 仅视觉：沿用已算好的 visual_risk
                vr = det.visual_risk if det.visual_risk is not None else 0.0
                out.objects.append(FusedObject(
                    source="vision", risk=float(vr), risk_class=det.risk_class,
                    bbox=det.bbox, angle_deg=det_bearing[di]))
                out.n_vision_only += 1

        # 3) 仅雷达「未知障碍物」（A3：先验≈0，几何主导）+ 路面门控 + 持续性
        for ti, t in enumerate(targets):
            if target_matched[ti]:
                continue
            on_road = is_on_road(eff_angle[ti], mask, image_w, cfg)
            if cfg.require_on_road and on_road is False:
                continue  # 投影到路面外 → 视为路边杂波，丢弃
            k_kin, ttc = kinematic_risk(t, cfg)
            lateral = lateral_factor_from_angle(eff_angle[ti], cfg)
            risk = _clamp01(k_kin * lateral * persist[ti])  # 无类别先验
            out.objects.append(FusedObject(
                source="radar", risk=risk, risk_class="unknown", bbox=None,
                distance_m=t.distance_m, relative_speed_mps=t.relative_speed_mps,
                ttc_sec=ttc, angle_deg=eff_angle[ti],
                on_road=on_road, persist=persist[ti]))
            out.n_radar_only += 1

        out.max_risk = max((o.risk for o in out.objects), default=0.0)
        return out

    def _matched_risk(self, det: DetectionResult, k_kin: float,
                      image_w: int, image_h: int) -> float:
        """视觉∩雷达：复用视觉风险因子，但用雷达运动学 k_kin 替换接近度。"""
        c = self.vcfg
        weights = (c.w_class, c.w_proximity, c.w_lateral, c.w_drivable, c.w_size)
        wsum = sum(weights)
        if wsum <= 0:
            return 0.0
        factors = (
            calculate_class_risk(det.risk_class, c),
            k_kin,  # ← 雷达真实 TTC/距离，替换 bbox 估距
            calculate_lateral_risk(det.bbox, image_w, c),
            calculate_drivable_risk(det.in_drivable_area, c),
            calculate_size_risk(det.bbox, image_w, image_h, c),
        )
        base = sum(w * f for w, f in zip(weights, factors)) / wsum
        return _clamp01(base * calculate_confidence_factor(det.confidence, c))
