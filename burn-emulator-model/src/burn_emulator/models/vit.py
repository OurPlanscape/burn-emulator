import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over non-batch dims
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)  # rescale so expected value is unchanged
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )  # (B, heads, N, head_dim)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, layer_scale_init) if layer_scale_init else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=drop)
        self.ls2 = LayerScale(dim, layer_scale_init) if layer_scale_init else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def circular_mask_indices(H: int, W: int, radius: float = None):
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if radius is None:
        radius = min(H, W) / 2.0
    mask = dist <= radius
    ys, xs = torch.where(mask)
    return ys.long(), xs.long(), mask


class PixelViT(nn.Module):
    def __init__(
        self,
        img_size: int = 129,
        in_chans: int = 19,
        radius: float = 64,
        embed_dim: int = 64,
        depth: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_classes: int = 3,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-5,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        # ys, xs, mask = circular_mask_indices(img_size, img_size, radius)
        # self.register_buffer("ys", ys)
        # self.register_buffer("xs", xs)
        # self.register_buffer("mask", mask)  # (H, W)
        # self.register_buffer("flat_idx", ys * img_size + xs)

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_pixels = self.img_size**2
        # self.num_pixels = ys.numel()

        self.pixel_embed = nn.Linear(in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_pixels + 1, embed_dim))
        nn.init.xavier_normal_(self.cls_token)
        nn.init.xavier_normal_(self.pos_embed)

        self.dropout = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=dropout,
                    attn_drop=attn_dropout,
                    drop_path=dpr[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def extract_pixels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert self.img_size == H and self.img_size == W, (
            f"expected {self.img_size}x{self.img_size} images, got {H}x{W}"
        )
        x_flat = x.reshape(B, C, H * W)  # view, no copy (x is contiguous)
        pix = x_flat.index_select(2, self.flat_idx)  # (B, C, N)
        pix = pix.permute(0, 2, 1)  # (B, N, C)
        return pix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # pix = self.extract_pixels(x)              # (B, N, C)
        # tokens = self.pixel_embed(pix)             # (B, N, D)
        tokens = self.pixel_embed(x.reshape(B, C, H * W).permute(0, 2, 1))

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)
        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        pixel_tokens = tokens[:, 1:]  # (B, N, D) drop CLS
        logits = self.head(pixel_tokens)  # (B, N, num_classes)

        # out = logits.new_zeros(B, self.num_classes, self.img_size * self.img_size)
        # out.index_copy_(2, self.flat_idx, logits.permute(0, 2, 1))
        # (B, num_classes, N) -> flat scatter
        out = logits.view(B, self.num_classes, self.img_size, self.img_size)
        return out
