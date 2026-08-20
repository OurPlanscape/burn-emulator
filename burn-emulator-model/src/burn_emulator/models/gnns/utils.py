import math
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing


class ScalarGate(nn.Module):
    def __init__(self, hidden_ch: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        v = value.unsqueeze(-1)
        return (v * torch.sigmoid(self.mlp(v))).squeeze(-1)


class RingGNNLayer(MessagePassing):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_rings: int,
        use_slope_gate: bool = True,
        use_wind_gate: bool = True,
    ):
        super().__init__(aggr="max")
        self.num_rings = num_rings
        self.use_slope_gate = use_slope_gate
        self.use_wind_gate = use_wind_gate

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_ch + 5, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.norm = nn.LayerNorm(out_ch)
        self.skip = nn.Linear(in_ch, out_ch, bias=False)
        if use_slope_gate:
            self.slope_gate = ScalarGate()
        if use_wind_gate:
            self.wind_gate = ScalarGate()

    def forward(self, x, edge_index, ring_id, t_align_src, t_align_dst, w_align):
        src, dst = edge_index
        ring_delta = ((ring_id[dst] - ring_id[src]).abs() / self.num_rings).unsqueeze(1)
        is_outward = (ring_id[dst].round() != ring_id[src].round()).to(x.dtype).unsqueeze(1)
        if self.use_slope_gate:
            sw = self.slope_gate(t_align_src).unsqueeze(1)
            dw = self.slope_gate(t_align_dst).unsqueeze(1)
        else:
            sw = t_align_src.unsqueeze(1)
            dw = t_align_dst.unsqueeze(1)

        ww = self.wind_gate(w_align).unsqueeze(1) if self.use_wind_gate else w_align.unsqueeze(1)

        edge_feat = torch.cat([ring_delta, is_outward, sw, dw, ww], dim=1)
        out = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        return self.norm(out + self.skip(x))

    def message(self, x_i, x_j, edge_feat):
        return self.mlp(torch.cat([x_i, x_j, edge_feat], dim=-1))


class ContClassProjector(nn.Module):
    def __init__(self, n_cont: int, n_class: int, hidden_channels: int):
        super().__init__()
        self.cont_proj = nn.Linear(n_cont, hidden_channels // 2)
        self.class_proj = nn.Linear(n_class, hidden_channels // 2)

    def forward(self, cont_feat: Tensor, class_feat: Tensor) -> Tensor:
        h_cont = F.relu(self.cont_proj(cont_feat))
        h_class = F.relu(self.class_proj(class_feat))
        return torch.cat([h_cont, h_class], dim=1)


def wind_deg_to_unit(wind_deg: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    wind_rad = wind_deg.to(dtype) * (math.pi / 180.0)
    return torch.sin(wind_rad), -torch.cos(wind_rad)


def edge_alignment(vx: Tensor, vy: Tensor, edge_dx_norm: Tensor, edge_dy_norm: Tensor) -> Tensor:
    return -(vx * edge_dx_norm + vy * edge_dy_norm)


def apply_edge_dropout(
    edge_index: Tensor,
    is_lateral: Tensor,
    lateral_p: float,
    outward_p: float,
    training: bool,
) -> Tensor:
    if not training or (lateral_p <= 0 and outward_p <= 0):
        return edge_index

    device = edge_index.device
    keep_lat = torch.rand(int(is_lateral.sum()), device=device) > lateral_p
    keep_out = torch.rand(int((~is_lateral).sum()), device=device) > outward_p
    keep = torch.empty(edge_index.shape[1], dtype=torch.bool, device=device)
    keep[is_lateral] = keep_lat
    keep[~is_lateral] = keep_out
    return edge_index[:, keep]


def partition_same_out_edges(
    a: Tensor, b: Tensor, ra: Tensor, rb: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    same = ra == rb
    a_out = ra < rb
    b_out = rb < ra

    same_src = torch.cat([a[same], b[same]])
    same_dst = torch.cat([b[same], a[same]])
    out_src = torch.cat([a[a_out], b[b_out]])
    out_dst = torch.cat([b[a_out], a[b_out]])
    return same_src, same_dst, out_src, out_dst


def run_branches_concurrently(
    branches: Iterable[nn.Module],
    call: Callable[[nn.Module], Tensor],
    use_streams: bool,
) -> list:
    branches = list(branches)
    if len(branches) > 1 and use_streams:
        current_stream = torch.cuda.current_stream()
        streams = [torch.cuda.Stream() for _ in branches]
        outputs = []
        for stream, branch in zip(streams, branches, strict=True):
            stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                outputs.append(call(branch))
        for stream in streams:
            current_stream.wait_stream(stream)
        return outputs
    return [call(branch) for branch in branches]
