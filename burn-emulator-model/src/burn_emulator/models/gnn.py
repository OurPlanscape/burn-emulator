import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


def build_radial_graph(
    grid_size: int = 128,
    num_rings: int = 64,
    grid_outward: bool = True,
    src_ratio: float = 0.1,
    n_dst: int = 4,
    n_neighbors: int = 4,
    seed: int = 42,
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
    ring_id = (dist / max_dist * num_rings).long().clamp(0, num_rings - 1)

    # get ring dist from center
    norm_r = ring_id.float() / (num_rings - 1)
    norm_dx = (pos[:, 0] - cx) / max_dist
    norm_dy = (pos[:, 1] - cy) / max_dist
    x_pos = torch.stack([norm_r, norm_dx, norm_dy], dim=1)  # (N, 3)

    # get center index for explicit edge creation later
    idx = torch.arange(N).reshape(H, W)
    centre_idx = dist.argmin().item()

    same_src_t, same_dst_t = [], []
    out_src_t, out_dst_t = [], []

    if grid_outward:
        # get 8 neighbor edges filter to only outward directed edges
        offsets_8 = [
            (dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1] if not (dy == 0 and dx == 0)
        ]
        for dy, dx in offsets_8:
            sy = slice(max(0, -dy), H + min(0, -dy))
            sx = slice(max(0, -dx), W + min(0, -dx))
            ty = slice(max(0, dy), H + min(0, dy))
            tx = slice(max(0, dx), W + min(0, dx))
            a = idx[sy, sx].reshape(-1)
            b = idx[ty, tx].reshape(-1)
            ra, rb = ring_id[a], ring_id[b]
            same = ra == rb
            a_out = ra < rb
            b_out = rb < ra
            same_src_t += [a[same], b[same]]
            same_dst_t += [b[same], a[same]]
            out_src_t += [a[a_out], b[b_out]]
            out_dst_t += [b[a_out], a[b_out]]
        # Explicitly connect centre pixel to all ring-1 pixels
        ring1 = torch.where(ring_id == 1)[0]
        if ring1.numel() > 0:
            out_src_t.append(torch.full((ring1.numel(),), centre_idx, dtype=torch.long))
            out_dst_t.append(ring1)
        same_src = torch.cat(same_src_t)
        same_dst = torch.cat(same_dst_t)

    else:
        gen = torch.Generator()
        gen.manual_seed(seed)
        ring_width = max_dist / num_rings
        seeded_same_src, seeded_same_dst = [], []

        # centre -> n_dst random pixels in ring 0
        ring0 = torch.where(ring_id == 0)[0]
        k0 = min(n_dst, ring0.numel())
        perm0 = torch.randperm(ring0.numel(), generator=gen)[:k0]
        out_src_t.append(torch.full((k0,), centre_idx, dtype=torch.long))
        out_dst_t.append(ring0[perm0])

        # O( N ᪲ ) lol
        for r in range(num_rings - 1):
            # get potential outward edges
            src_nodes = torch.where(ring_id == r)[0]
            dst_nodes = torch.where(ring_id == r + 1)[0]
            if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
                continue

            # filter to density of src nodes
            k_src = max(1, int(src_nodes.numel() * src_ratio))
            src_perm = torch.randperm(src_nodes.numel(), generator=gen)[:k_src]
            srcs = src_nodes[src_perm]
            dst_pos_all = pos[dst_nodes]

            selected_dsts = set()
            for s in srcs.tolist():
                # find potential dst within ring_width*2
                src_p = pos[s]
                d = ((dst_pos_all - src_p) ** 2).sum(dim=1).sqrt()
                nearby = dst_nodes[d <= ring_width * 2]
                if nearby.numel() == 0:
                    continue

                # append edge and add to selection for distal edge
                k_dst = min(n_dst, nearby.numel())
                dst_perm = torch.randperm(nearby.numel(), generator=gen)[:k_dst]
                dsts = nearby[dst_perm]
                out_src_t.append(torch.full((dsts.numel(),), s, dtype=torch.long))
                out_dst_t.append(dsts)
                selected_dsts.update(dsts.tolist())

            # each selected dst seeds lateral edges to its n_neighbors nearest same-ring neighbours
            for d_idx in selected_dsts:
                d_pos = pos[d_idx]
                d2 = ((dst_pos_all - d_pos) ** 2).sum(dim=1).sqrt()
                nearby_same = dst_nodes[(d2 > 0) & (d2 <= ring_width * 2)]
                if nearby_same.numel() == 0:
                    continue
                # take n_neighbors nearest
                k_nb = min(n_neighbors, nearby_same.numel())
                nb_d = d2[(d2 > 0) & (d2 <= ring_width * 2)]
                neighbours = nearby_same[nb_d.argsort()[:k_nb]]
                n = neighbours.numel()
                seeded_same_src += [torch.full((n,), d_idx, dtype=torch.long), neighbours]
                seeded_same_dst += [neighbours, torch.full((n,), d_idx, dtype=torch.long)]

        if seeded_same_src:
            same_src = torch.cat(seeded_same_src)
            same_dst = torch.cat(seeded_same_dst)
        else:
            same_src = torch.empty(0, dtype=torch.long)
            same_dst = torch.empty(0, dtype=torch.long)

    out_src = torch.cat(out_src_t) if out_src_t else torch.empty(0, dtype=torch.long)
    out_dst = torch.cat(out_dst_t) if out_dst_t else torch.empty(0, dtype=torch.long)

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

    return Data(
        x=x_pos,
        edge_index=torch.cat([same_edge_index, out_edge_index], dim=1),
        same_edge_index=same_edge_index,
        out_edge_index=out_edge_index,
        pos=pos,
        ring_id=ring_id,
    )


def batch_edge_index(edge_index: torch.Tensor, N: int, B: int) -> torch.Tensor:
    offsets = torch.arange(B, device=edge_index.device) * N
    ei = edge_index.unsqueeze(0) + offsets.reshape(B, 1, 1)  # (B, 2, E)
    return ei.permute(1, 0, 2).reshape(2, -1)  # (2, B*E)


def sample_image(image: torch.Tensor, pos: torch.Tensor, grid_size: int = 128):
    B, C, H, W = image.shape
    norm = (pos / (grid_size - 1)) * 2.0 - 1.0
    grid = norm.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, 2)
    out = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2).permute(0, 2, 1)  # (B, N, C)


class RadialGNNLayer(MessagePassing):
    def __init__(self, in_ch: int, out_ch: int, num_rings: int, use_slope_gate: bool = True):
        super().__init__(aggr="max")
        self.num_rings = num_rings
        self.use_slope_gate = use_slope_gate

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_ch + 4, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.norm = nn.LayerNorm(out_ch)
        self.skip = nn.Linear(in_ch, out_ch, bias=False)
        if use_slope_gate:
            self.slope_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, edge_index, ring_id, t_align_src, t_align_dst):
        src, dst = edge_index
        ring_delta = ((ring_id[dst] - ring_id[src]).abs() / self.num_rings).unsqueeze(1)
        is_outward = (ring_id[dst].round() != ring_id[src].round()).to(x.dtype).unsqueeze(1)
        if self.use_slope_gate:
            sw = (t_align_src * torch.sigmoid(self.slope_gate)).unsqueeze(1)
            dw = (t_align_dst * torch.sigmoid(self.slope_gate)).unsqueeze(1)
        else:
            sw = t_align_src.unsqueeze(1)
            dw = t_align_dst.unsqueeze(1)
        edge_feat = torch.cat([ring_delta, is_outward, sw, dw], dim=1)
        out = self.propagate(edge_index, x=x, edge_feat=edge_feat)
        return self.norm(out + self.skip(x))

    def message(self, x_i, x_j, edge_feat):
        return self.mlp(torch.cat([x_i, x_j, edge_feat], dim=-1))


class GNNBranch(nn.Module):
    def __init__(
        self,
        num_rings: int,
        img_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        grid_size: int,
        train_batch_size: int,
        grid_outward: bool = True,
        src_ratio: float = 0.1,
        n_dst: int = 4,
        n_neighbors: int = 2,
        lateral_edge_dropout: float = 0.0,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.dropout = dropout
        self.num_rings = num_rings
        self.train_batch_size = train_batch_size
        self.lateral_edge_dropout = lateral_edge_dropout
        self.outward_edge_dropout = outward_edge_dropout

        # building graph with edge filter rules
        graph = build_radial_graph(
            grid_size=grid_size,
            num_rings=num_rings,
            grid_outward=grid_outward,
            src_ratio=src_ratio if src_ratio is not None else 0.1,
            n_dst=n_dst if n_dst is not None else 4,
            n_neighbors=n_neighbors if n_neighbors is not None else 2,
        )

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
        self.cont_proj = nn.Linear(n_cont, hidden_channels // 2)
        self.class_proj = nn.Linear(n_class, hidden_channels // 2)
        self.layers = nn.ModuleList(
            [
                RadialGNNLayer(
                    hidden_channels, hidden_channels, num_rings, use_slope_gate=use_slope_gate
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, feats_flat: torch.Tensor, B: int, missing_flat: torch.Tensor) -> torch.Tensor:
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
            edge_local = ei_batch[0] % E_single
            is_lateral = edge_local < E_same
            keep_lat = (
                torch.rand(is_lateral.sum(), device=ei_batch.device) > self.lateral_edge_dropout
            )
            keep_out = (
                torch.rand((~is_lateral).sum(), device=ei_batch.device) > self.outward_edge_dropout
            )
            keep = torch.empty(ei_batch.shape[1], dtype=torch.bool, device=ei_batch.device)
            keep[is_lateral] = keep_lat
            keep[~is_lateral] = keep_out
            ei_batch = ei_batch[:, keep]

        # terrain alignment using pre-computed unit direction vectors
        src_v = ei_batch[0]
        dst_v = ei_batch[1]
        edge_local = src_v % E_single
        fx_src = feats_flat[src_v, 0].float()
        fy_src = feats_flat[src_v, 1].float()
        fx_dst = feats_flat[dst_v, 0].float()
        fy_dst = feats_flat[dst_v, 1].float()

        # t_align_src: how much is this edge going uphill from src's perspective
        t_align_src = (
            -(fx_src * self.edge_dx_norm[edge_local] + fy_src * self.edge_dy_norm[edge_local])
        ).to(dtype)
        # t_align_dst: how much is the edge arriving uphill at dst (same direction, dst flow)
        t_align_dst = (
            -(fx_dst * self.edge_dx_norm[edge_local] + fy_dst * self.edge_dy_norm[edge_local])
        ).to(dtype)

        # ring_id tiled to (B*N,) for use in layer forward
        ring_id_long = self.ring_id.unsqueeze(0).expand(B, -1).reshape(-1).to(dtype)

        # projecting continuous variables and one-hot variables separately
        h_cont = F.relu(self.cont_proj(torch.cat([feats_flat[:, :-13], pos_feat_t], dim=1)))
        h_class = F.relu(self.class_proj(feats_flat[:, -13:]))
        h = torch.cat([h_cont, h_class], dim=1)

        for layer in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, ei_batch, ring_id_long, t_align_src, t_align_dst)
            h = F.relu(h)

        return h


class PixelDecoder(nn.Module):
    def __init__(self, hidden_ch: int, out_ch: int = 3, grid_size: int = 128, refine_ch: int = 32):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_ch = hidden_ch
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_ch, refine_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_ch, refine_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(refine_ch, out_ch, 1),
        )

    def forward(self, node_feats: torch.Tensor, B: int) -> torch.Tensor:
        H = W = self.grid_size
        C = self.hidden_ch
        canvas = node_feats.T.reshape(C, B, H, W).permute(1, 0, 2, 3)
        return self.refine(canvas)


class RadialGNN(nn.Module):
    def __init__(
        self,
        img_channels: int = 19,
        hidden_channels: int = 64,
        out_channels: int = 3,
        num_layers: tuple = (64, 32, 16),
        dropout: float = 0.1,
        grid_size: int = 128,
        refine_ch: int = 64,
        ring_scales: tuple = (64, 32, 16),
        grid_outward: tuple = (True, False, False),
        src_ratio: tuple = (None, 0.5, 0.25),
        n_dst: tuple = (None, 4, 4),
        n_neighbors: tuple = (None, 2, 2),
        lateral_edge_dropout: float = 0.3,
        outward_edge_dropout: float = 0.0,
        use_slope_gate: tuple = (True, True, False),
        train_batch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = grid_size

        assert (
            len(ring_scales)
            == len(num_layers)
            == len(grid_outward)
            == len(src_ratio)
            == len(n_dst)
            == len(n_neighbors)
            == len(use_slope_gate)
        ), "all branch tuples must have the same length"

        self.branches = nn.ModuleList(
            [
                GNNBranch(
                    num_rings=nr,
                    img_channels=img_channels,
                    hidden_channels=hidden_channels,
                    num_layers=nl,
                    dropout=dropout,
                    grid_size=grid_size,
                    train_batch_size=train_batch_size,
                    grid_outward=gout,
                    src_ratio=sr,
                    n_dst=nd,
                    n_neighbors=nn,
                    lateral_edge_dropout=lateral_edge_dropout,
                    outward_edge_dropout=outward_edge_dropout,
                    use_slope_gate=usg,
                )
                for nr, nl, gout, sr, nd, nn, usg in zip(
                    ring_scales,
                    num_layers,
                    grid_outward,
                    src_ratio,
                    n_dst,
                    n_neighbors,
                    use_slope_gate,
                )
            ]
        )

        self.scale_proj = nn.Linear(hidden_channels * len(ring_scales), hidden_channels)
        self.decoder = PixelDecoder(hidden_channels, out_channels, grid_size, refine_ch)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        N = self.branches[0].pos.shape[0]

        missing = image[:, 0] < -0.8
        missing_flat = missing.reshape(B * N)

        img_feats = sample_image(image, self.branches[0].pos.to(image.dtype), self.grid_size)
        feats_flat = img_feats.reshape(B * N, -1)

        # branches are independent — run in parallel with jit.fork
        futures = [torch.jit.fork(branch, feats_flat, B, missing_flat) for branch in self.branches]
        branch_outputs = [torch.jit.wait(f) for f in futures]

        h = torch.cat(branch_outputs, dim=1)
        h = F.relu(self.scale_proj(h))

        return self.decoder(h, B)
