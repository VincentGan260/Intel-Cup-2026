from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn

@dataclass(frozen=True)
class RiskPrediction:
    probabilities: tuple[float, float, float]
    model_score: float
    model_level: int

class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x))) + x

class GTMRFN(nn.Module):
    """Per-modality MLP + two-layer TCN + gated fusion + three-class head."""
    def __init__(self, modality_dims: dict[str, int], hidden_dim: int = 24, num_classes: int = 3) -> None:
        super().__init__()
        self.modality_names = tuple(modality_dims)
        self.modality_dims = dict(modality_dims)
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for name, dim in modality_dims.items()})
        self.tcn = nn.Sequential(TemporalBlock(hidden_dim, 1), TemporalBlock(hidden_dim, 2))
        self.gates = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, num_classes))
    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        temporal, valid = [], []
        for name in self.modality_names:
            x = inputs[name]
            temporal.append(self.tcn(self.encoders[name](x).transpose(1, 2)).transpose(1, 2)[:, -1])
            valid.append(x[:, -1, -1] > 0.5)
        encoded, mask = torch.stack(temporal, 1), torch.stack(valid, 1)
        gate_logits = self.gates(encoded).squeeze(-1).masked_fill(~mask, -1e4)
        gate_logits = torch.where((~mask.any(1)).unsqueeze(1), torch.zeros_like(gate_logits), gate_logits)
        fused = (encoded * torch.softmax(gate_logits, 1).unsqueeze(-1)).sum(1)
        return self.head(fused)
