import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from burn_emulator.models.gnns.utils import (
    ContClassProjector,
    RingGNNLayer,
    apply_edge_dropout,
    edge_alignment,
    partition_same_out_edges,
    run_branches_concurrently,
    wind_deg_to_unit,
)


def build_radial_graph(
    grid_size: int = 129,
    ring_width: int = 2**0.5,
    filter_corners: bool = True,
) -> Data:
    H = W = grid_size
    N = H * W

    # get coordinates
    ys = torch.arange(H, dtype=torch.float)
    xs = torch.arange(W, dtype=torch.float)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (N, 2)

    # assign rings to coords
    cx = cy = (grid_size - 1) / 2.0
    max_dist = (cx**2 + cy**2) ** 0.5
    dist = ((pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2).sqrt()
    ring_id = torch.ceil(dist / ring_width)
    num_rings = ring_id.max().long().item() + 1

    keep_mask = dist < min(cx, cy) + 1 if filter_corners else torch.ones(N, dtype=torch.bool)

    # map old flat index -> new compacted index (-1 for dropped nodes)
    new_idx = torch.full((N,), -1, dtype=torch.long)
    new_idx[keep_mask] = torch.arange(int(keep_mask.sum()))

    # get normalized ring dist from center
    norm_r = ring_id.float() / num_rings
    norm_dx = (pos[:, 0] - cx) / max_dist
    norm_dy = (pos[:, 1] - cy) / max_dist
    x_pos = torch.stack([norm_r, norm_dx, norm_dy], dim=1)  # (N, 3)

    # gather 8-neighbor edges across the grid, filtered to kept nodes
    idx = torch.arange(N).reshape(H, W)
    a_nodes, b_nodes = [], []

    offsets_8 = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)]
    for dy, dx in offsets_8:
        sy = slice(max(0, -dy), H + min(0, -dy))
        sx = slice(max(0, -dx), W + min(0, -dx))
        ty = slice(max(0, dy), H + min(0, dy))
        tx = slice(max(0, dx), W + min(0, dx))
        a = idx[sy, sx].reshape(-1)
        b = idx[ty, tx].reshape(-1)

        valid = keep_mask[a] & keep_mask[b]
        a_nodes.append(a[valid])
        b_nodes.append(b[valid])

    a, b = torch.cat(a_nodes), torch.cat(b_nodes)
    same_src, same_dst, out_src, out_dst = partition_same_out_edges(a, b, ring_id[a], ring_id[b])

    # remap to compacted node indices
    same_src, same_dst = new_idx[same_src], new_idx[same_dst]
    out_src, out_dst = new_idx[out_src], new_idx[out_dst]

    same_edge_index = (
        torch.unique(torch.stack([same_src, same_dst], dim=0), dim=1)
        if same_src.numel() > 0
        else torch.empty(2, 0, dtype=torch.long)
    )
    out_edge_index = (
        torch.unique(torch.stack([out_src, out_dst], dim=0), dim=1)
        if out_src.numel() > 0
        else torch.empty(2, 0, dtype=torch.long)
    )

    x_pos = x_pos[keep_mask]
    pos = pos[keep_mask]
    ring_id = ring_id[keep_mask]

    return Data(
        x=x_pos,
        edge_index=torch.cat([same_edge_index, out_edge_index], dim=1),
        same_edge_index=same_edge_index,
        out_edge_index=out_edge_index,
        pos=pos,
        ring_id=ring_id,
    ), num_rings


def batch_edge_index(edge_index: torch.Tensor, N: int, B: int) -> torch.Tensor:
    offsets = torch.arange(B, device=edge_index.device) * N
    ei = edge_index.unsqueeze(0) + offsets.reshape(B, 1, 1)  # (B, 2, E)
    return ei.permute(1, 0, 2).reshape(2, -1)  # (2, B*E)


def sample_image(image: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    xs, ys = pos[:, 0].long(), pos[:, 1].long()
    return image[:, :, ys, xs].permute(0, 2, 1)  # (B, N, C)


class GNNBranch(nn.Module):
    def __init__(
        self,
        ring_width: int,
        img_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        grid_size: int,
        train_batch_size: int,
        lateral_edge_dropout: float = 0.0,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: bool = True,
        use_wind_gate: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.dropout = dropout
        self.ring_width = ring_width
        self.train_batch_size = train_batch_size
        self.lateral_edge_dropout = lateral_edge_dropout
        self.outward_edge_dropout = outward_edge_dropout

        # building graph with edge filter rules
        graph, num_rings = build_radial_graph(
            grid_size=grid_size,
            ring_width=ring_width,
        )
        self.num_rings = num_rings

        # is this even necessary?
        self.register_buffer("same_edge_index", graph.same_edge_index)
        self.register_buffer("out_edge_index", graph.out_edge_index)
        self.register_buffer("ring_id", graph.ring_id)
        self.register_buffer("pos", graph.pos)
        self.register_buffer("pos_feat", graph.x)

        # pre-tile edge index and node features for train_batch_size
        N = graph.pos.shape[0]
        ei_single = torch.cat([graph.same_edge_index, graph.out_edge_index], dim=1)
        ei_batched = batch_edge_index(ei_single, N, train_batch_size)
        self.register_buffer("ei_batched", ei_batched)

        pos_feat_batched = (
            graph.x.unsqueeze(0).expand(train_batch_size, -1, -1).reshape(train_batch_size * N, -1)
        )
        self.register_buffer("pos_feat_batched", pos_feat_batched)

        # pre-compute normalised edge direction vectors (fixed per graph)
        src, dst = ei_single
        edge_dx = graph.pos[dst, 0] - graph.pos[src, 0]
        edge_dy = graph.pos[dst, 1] - graph.pos[src, 1]
        edge_len = (edge_dx**2 + edge_dy**2).sqrt().clamp(min=1e-6)
        self.register_buffer("edge_dx_norm", edge_dx / edge_len)  # src->dst unit vector x
        self.register_buffer("edge_dy_norm", edge_dy / edge_len)  # src->dst unit vector y

        # separate projections for continuous and one-hot channels from burn_emulator:
        #   ch0-1 : flow_x, flow_y  (continuous)
        #   ch2-5 : 4 continuous variables
        #   ch6-18 : 13 one-hot class channels
        # pos_feat adds 3 more continuous dims (norm_r, norm_dx, norm_dy)
        n_cont = (img_channels - 13) + 3  # flow + continuous + pos features
        n_class = 13
        self.projector = ContClassProjector(n_cont, n_class, hidden_channels)
        self.layers = nn.ModuleList(
            [
                RingGNNLayer(
                    hidden_channels,
                    hidden_channels,
                    num_rings,
                    use_slope_gate=use_slope_gate,
                    use_wind_gate=use_wind_gate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        feats_flat: torch.Tensor,
        B: int,
        missing_flat: torch.Tensor,
        wind_dx: torch.Tensor,
        wind_dy: torch.Tensor,
    ) -> torch.Tensor:
        N = self.pos.shape[0]
        dtype = feats_flat.dtype

        # use pre-tiled buffers if B matches train_batch_size, else compute on the fly
        if self.train_batch_size == B:
            ei_batch = self.ei_batched
            pos_feat_t = self.pos_feat_batched.to(dtype)
        else:
            ei_single = torch.cat([self.same_edge_index, self.out_edge_index], dim=1)
            ei_batch = batch_edge_index(ei_single, N, B)
            pos_feat_t = self.pos_feat.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1).to(dtype)

        # filter edges touching missing pixels
        src, dst = ei_batch
        valid = ~missing_flat[src] & ~missing_flat[dst]
        ei_batch = ei_batch[:, valid]

        E_single = self.edge_dx_norm.shape[0]

        # edge dropout (training only) — separate rates for lateral and outward edges
        if self.training and (self.lateral_edge_dropout > 0 or self.outward_edge_dropout > 0):
            E_same = self.same_edge_index.shape[1]
            is_lateral = (ei_batch[0] % E_single) < E_same
            ei_batch = apply_edge_dropout(
                ei_batch,
                is_lateral,
                self.lateral_edge_dropout,
                self.outward_edge_dropout,
                self.training,
            )

        # terrain/wind alignment using pre-computed unit direction vectors
        src_v = ei_batch[0]
        dst_v = ei_batch[1]
        edge_local = src_v % E_single
        edge_dx_norm = self.edge_dx_norm[edge_local]
        edge_dy_norm = self.edge_dy_norm[edge_local]
        fx_src = feats_flat[src_v, 0].float()
        fy_src = feats_flat[src_v, 1].float()
        fx_dst = feats_flat[dst_v, 0].float()
        fy_dst = feats_flat[dst_v, 1].float()

        # t_align_src/dst: how much the edge is going/arriving uphill from
        # the src/dst node's perspective
        t_align_src = edge_alignment(fx_src, fy_src, edge_dx_norm, edge_dy_norm).to(dtype)
        t_align_dst = edge_alignment(fx_dst, fy_dst, edge_dx_norm, edge_dy_norm).to(dtype)

        # wind alignment: how much this edge points along
        # the (per-sample, global) wind direction
        batch_idx = src_v // N
        w_align = edge_alignment(
            wind_dx[batch_idx], wind_dy[batch_idx], edge_dx_norm, edge_dy_norm
        ).to(dtype)

        # ring_id tiled to (B*N,) for use in layer forward
        ring_id_long = self.ring_id.unsqueeze(0).expand(B, -1).reshape(-1).to(dtype)

        # projecting continuous variables and one-hot variables separately
        h = self.projector(torch.cat([feats_flat[:, :-13], pos_feat_t], dim=1), feats_flat[:, -13:])

        for layer in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, ei_batch, ring_id_long, t_align_src, t_align_dst, w_align)
            h = F.relu(h)

        return h


class PixelDecoder(nn.Module):
    def __init__(self, hidden_ch: int, out_ch: int = 3, grid_size: int = 128, refine_ch: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_ch = hidden_ch
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_ch, refine_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_ch, refine_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_ch, out_ch, 1),
        )

    def forward(self, node_feats: torch.Tensor, pos: torch.Tensor, B: int) -> torch.Tensor:
        H = W = self.grid_size
        C = self.hidden_ch
        N = pos.shape[0]

        xs, ys = pos[:, 0].long(), pos[:, 1].long()
        canvas = node_feats.new_zeros(B, C, H, W)
        canvas[:, :, ys, xs] = node_feats.reshape(B, N, C).permute(0, 2, 1)

        return self.refine(canvas)


class RadialGNN(nn.Module):
    def __init__(
        self,
        img_channels: int = 19,
        hidden_channels: int = 64,
        out_channels: int = 3,
        num_layers: tuple = (64,),
        dropout: float = 0.1,
        grid_size: int = 129,
        refine_ch: int = 64,
        ring_widths: tuple = (2**0.5,),
        lateral_edge_dropout: float = 0.3,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: tuple = (True,),
        use_wind_gate: tuple = (True,),
        train_batch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = grid_size

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
                    grid_size=grid_size,
                    train_batch_size=train_batch_size,
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
        self.decoder = PixelDecoder(hidden_channels, out_channels, grid_size, refine_ch)

    def forward(self, image: torch.Tensor, wind_deg: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        N = self.branches[0].pos.shape[0]

        wind_dx, wind_dy = wind_deg_to_unit(wind_deg, image.dtype)

        img_feats = sample_image(image, self.branches[0].pos)
        feats_flat = img_feats.reshape(B * N, -1)
        missing_flat = feats_flat[:, 0] < -0.8

        # branches are independent — run concurrently on separate CUDA streams
        branch_outputs = run_branches_concurrently(
            self.branches,
            lambda branch: branch(feats_flat, B, missing_flat, wind_dx, wind_dy),
            use_streams=feats_flat.is_cuda,
        )

        h = torch.cat(branch_outputs, dim=1)
        h = F.relu(self.scale_proj(h))

        return self.decoder(h, self.branches[0].pos, B)
