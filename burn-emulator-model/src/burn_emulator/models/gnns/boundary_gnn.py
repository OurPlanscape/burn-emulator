import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from burn_emulator.models.gnns.utils import (
    ContClassProjector,
    RingGNNLayer,
    apply_edge_dropout,
    edge_alignment,
    partition_same_out_edges,
    run_branches_concurrently,
    wind_deg_to_unit,
)


def build_grid_adjacency(H: int, W: int, device=None) -> Data:
    """Static 8-connected pixel adjacency for an H x W raster"""
    N = H * W
    ys = torch.arange(H, dtype=torch.float, device=device)
    xs = torch.arange(W, dtype=torch.float, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (N, 2)

    idx = torch.arange(N, device=device).reshape(H, W)
    # 4 "forward" offsets cover every unordered neighbor pair exactly once
    offsets_4 = [(0, 1), (1, -1), (1, 0), (1, 1)]
    src_nodes, dst_nodes = [], []
    for dy, dx in offsets_4:
        sy = slice(max(0, -dy), H + min(0, -dy))
        sx = slice(max(0, -dx), W + min(0, -dx))
        ty = slice(max(0, dy), H + min(0, dy))
        tx = slice(max(0, dx), W + min(0, dx))
        src_nodes.append(idx[sy, sx].reshape(-1))
        dst_nodes.append(idx[ty, tx].reshape(-1))

    edge_index = torch.stack([torch.cat(src_nodes), torch.cat(dst_nodes)], dim=0)
    return Data(pos=pos, edge_index=edge_index)


def compute_boundary_rings(
    mask: torch.Tensor,
    ring_width: int,
    max_distance: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    H, W = mask.shape
    device = mask.device
    mask = mask.bool()

    visited = mask.clone()
    frontier = mask.clone()
    steps = torch.zeros(H, W, dtype=torch.long, device=device)

    kernel = torch.ones(1, 1, 3, 3, device=device)
    for step in range(1, max_distance + 1):
        dilated = F.conv2d(frontier.float().view(1, 1, H, W), kernel, padding=1).view(H, W) > 0
        frontier = dilated & ~visited
        if not frontier.any():
            break
        steps[frontier] = step
        visited |= frontier

    exterior = ~mask
    reached = exterior & (steps > 0)
    keep = mask | reached

    ring_id = torch.zeros(H, W, dtype=torch.long, device=device)
    ring_id[reached] = (steps[reached] - 1) // ring_width + 1
    return ring_id, keep


def build_boundary_graph(
    image: torch.Tensor,
    boundary_mask: torch.Tensor,
    ring_width: int,
    max_distance: int,
) -> Data:
    C, H, W = image.shape
    N = H * W
    device = image.device

    adjacency = build_grid_adjacency(H, W, device=device)
    ring_id, keep = compute_boundary_rings(boundary_mask, ring_width, max_distance)
    ring_id_flat = ring_id.reshape(-1)
    keep_flat = keep.reshape(-1)

    feats_flat = image.permute(1, 2, 0).reshape(N, C)
    keep_flat = keep_flat & (feats_flat[:, 0] >= -0.8)  # drop no-data pixels too

    a, b = adjacency.edge_index
    valid = keep_flat[a] & keep_flat[b]
    a, b = a[valid], b[valid]
    ra, rb = ring_id_flat[a], ring_id_flat[b]

    same_src, same_dst, out_src, out_dst = partition_same_out_edges(a, b, ra, rb)
    is_lateral = torch.cat(
        [
            torch.ones(same_src.shape[0], dtype=torch.bool, device=device),
            torch.zeros(out_src.shape[0], dtype=torch.bool, device=device),
        ]
    )

    new_idx = torch.full((N,), -1, dtype=torch.long, device=device)
    new_idx[keep_flat] = torch.arange(int(keep_flat.sum()), device=device)
    src = new_idx[torch.cat([same_src, out_src])]
    dst = new_idx[torch.cat([same_dst, out_dst])]
    edge_index = torch.stack([src, dst], dim=0)

    max_rings = -(-max_distance // ring_width)  # ceil div
    norm_r = (ring_id_flat.float() / max_rings).clamp(max=1.0)
    norm_x = adjacency.pos[:, 0] / max(W - 1, 1)
    norm_y = adjacency.pos[:, 1] / max(H - 1, 1)
    pos_feat = torch.stack([norm_r, norm_x, norm_y], dim=1)

    return Data(
        x=feats_flat[keep_flat],
        pos=adjacency.pos[keep_flat],
        pos_feat=pos_feat[keep_flat],
        ring_id=ring_id_flat[keep_flat].float(),
        edge_index=edge_index,
        is_lateral=is_lateral,
    )


class GNNBranch(nn.Module):
    def __init__(
        self,
        ring_width: int,
        img_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        max_distance: int,
        lateral_edge_dropout: float = 0.0,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: bool = True,
        use_wind_gate: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.ring_width = ring_width
        self.max_distance = max_distance
        self.max_rings = -(-max_distance // ring_width)  # ceil div
        self.lateral_edge_dropout = lateral_edge_dropout
        self.outward_edge_dropout = outward_edge_dropout

        # separate projections for continuous and one-hot channels from burn_emulator:
        #   ch0-1 : flow_x, flow_y  (continuous)
        #   ch2-5 : 4 continuous variables
        #   ch6-18 : 13 one-hot class channels
        # +3 more continuous dims for boundary-ring position features (norm_r, norm_x, norm_y)
        n_cont = (img_channels - 13) + 3
        n_class = 13
        self.projector = ContClassProjector(n_cont, n_class, hidden_channels)
        self.layers = nn.ModuleList(
            [
                RingGNNLayer(
                    hidden_channels,
                    hidden_channels,
                    self.max_rings,
                    use_slope_gate=use_slope_gate,
                    use_wind_gate=use_wind_gate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        images: list[torch.Tensor],
        boundary_masks: list[torch.Tensor],
        wind_dx: torch.Tensor,
        wind_dy: torch.Tensor,
    ) -> Batch:
        graphs = [
            build_boundary_graph(img, mask, self.ring_width, self.max_distance)
            for img, mask in zip(images, boundary_masks, strict=True)
        ]
        data = Batch.from_data_list(graphs)
        dtype = data.x.dtype

        ei = data.edge_index
        is_lateral = data.is_lateral

        # edge dropout (training only) — separate rates for lateral and outward edges
        ei = apply_edge_dropout(
            ei, is_lateral, self.lateral_edge_dropout, self.outward_edge_dropout, self.training
        )

        # direction vectors: computed per batch since which side is src/dst
        # is data-dependent (flips per sample)
        src, dst = ei
        pos = data.pos
        edge_dx = pos[dst, 0] - pos[src, 0]
        edge_dy = pos[dst, 1] - pos[src, 1]
        edge_len = (edge_dx**2 + edge_dy**2).sqrt().clamp(min=1e-6)
        edge_dx_norm = (edge_dx / edge_len).to(dtype)
        edge_dy_norm = (edge_dy / edge_len).to(dtype)

        # terrain alignment
        fx_src, fy_src = data.x[src, 0].float(), data.x[src, 1].float()
        fx_dst, fy_dst = data.x[dst, 0].float(), data.x[dst, 1].float()
        t_align_src = edge_alignment(fx_src, fy_src, edge_dx_norm, edge_dy_norm).to(dtype)
        t_align_dst = edge_alignment(fx_dst, fy_dst, edge_dx_norm, edge_dy_norm).to(dtype)

        # wind alignment: how much this edge points along the (per-sample) wind direction
        edge_sample = data.batch[src]
        w_align = edge_alignment(
            wind_dx[edge_sample], wind_dy[edge_sample], edge_dx_norm, edge_dy_norm
        ).to(dtype)

        ring_id = data.ring_id
        h = self.projector(torch.cat([data.x[:, :-13], data.pos_feat], dim=1), data.x[:, -13:])

        for layer in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, ei, ring_id, t_align_src, t_align_dst, w_align)
            h = F.relu(h)

        return h, data.pos, data.batch


class PixelDecoder(nn.Module):
    """Reconstructs each sample at its own resolution. Since the refine stack
    is all 1x1 convs (equivalently, a per-pixel MLP), it's applied directly
    on flat node features and only reshaped into an image per sample at the
    end — no shared canvas needed.
    """

    def __init__(self, hidden_ch: int, out_ch: int = 3, refine_ch: int = 32):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Linear(hidden_ch, refine_ch),
            nn.ReLU(inplace=True),
            nn.Linear(refine_ch, refine_ch),
            nn.ReLU(inplace=True),
            nn.Linear(refine_ch, out_ch),
        )

    def forward(
        self, node_feats: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, sizes: list
    ) -> list:
        out = self.refine(node_feats)  # (N_total, out_ch)
        images = []
        for i, (H, W) in enumerate(sizes):
            sel = batch == i
            xs, ys = pos[sel, 0].long(), pos[sel, 1].long()
            canvas = out.new_zeros(out.shape[1], H, W)
            canvas[:, ys, xs] = out[sel].t()
            images.append(canvas)
        return images


class BoundaryGNN(nn.Module):
    def __init__(
        self,
        img_channels: int = 19,
        hidden_channels: int = 64,
        out_channels: int = 3,
        num_layers: tuple = (64,),
        dropout: float = 0.1,
        refine_ch: int = 64,
        ring_widths: tuple = (2**0.5,),
        max_distance: int = 64,
        lateral_edge_dropout: float = 0.3,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: tuple = (True,),
        use_wind_gate: tuple = (True,),
    ):
        super().__init__()

        assert len(ring_widths) == len(num_layers) == len(use_slope_gate) == len(use_wind_gate), (
            "all branch tuples must have the same length"
        )
        self.branches = nn.ModuleList(
            [
                GNNBranch(
                    ring_width=rw,
                    img_channels=img_channels,
                    hidden_channels=hidden_channels,
                    num_layers=nl,
                    dropout=dropout,
                    max_distance=max_distance,
                    lateral_edge_dropout=lateral_edge_dropout,
                    outward_edge_dropout=outward_edge_dropout,
                    use_slope_gate=usg,
                    use_wind_gate=uwg,
                )
                for rw, nl, usg, uwg in zip(
                    ring_widths,
                    num_layers,
                    use_slope_gate,
                    use_wind_gate,
                    strict=True,
                )
            ]
        )

        self.scale_proj = nn.Linear(hidden_channels * len(ring_widths), hidden_channels)
        self.decoder = PixelDecoder(hidden_channels, out_channels, refine_ch)

    def forward(
        self,
        images: list,
        wind_deg: torch.Tensor,
        boundary_masks: list,
    ) -> list:
        wind_dx, wind_dy = wind_deg_to_unit(wind_deg, images[0].dtype)

        # branches are independent — run concurrently on separate CUDA streams
        branch_outputs = run_branches_concurrently(
            self.branches,
            lambda branch: branch(images, boundary_masks, wind_dx, wind_dy),
            use_streams=images[0].is_cuda,
        )

        # branches share node layout per sample (same max_distance cutoff),
        # so pos/batch from any branch describe all of them
        pos, batch = branch_outputs[0][1], branch_outputs[0][2]
        h = torch.cat([out for out, _, _ in branch_outputs], dim=1)
        h = F.relu(self.scale_proj(h))

        sizes = [tuple(img.shape[1:]) for img in images]
        return self.decoder(h, pos, batch, sizes)
