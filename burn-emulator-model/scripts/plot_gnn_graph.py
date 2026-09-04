import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from burn_emulator.models.gnns.radial_gnn import build_radial_graph


def _ring_parity_colors(ring_id, num_rings, shade_range=(0.35, 0.95)):
    """
    Map each node to a color based on its ring's parity:
      - even rings -> shades of purple
      - odd rings  -> shades of green
    Each successive ring (within a parity group) gets a different shade,
    so e.g. ring 0 and ring 2 are both purple but visually distinct.
    """
    even_cmap = plt.get_cmap("Purples")
    odd_cmap = plt.get_cmap("Greens")

    # normalize ring index *within its own parity group* so shades progress
    # smoothly from light to dark as you move outward, rather than jumping
    # around based on the raw (global) ring number.
    even_rings = np.arange(0, num_rings, 2)
    odd_rings = np.arange(1, num_rings, 2)

    def _shade_lookup(rings, cmap):
        n = max(len(rings), 1)
        shades = np.linspace(shade_range[0], shade_range[1], n)
        lookup = {r: cmap(s) for r, s in zip(rings, shades, strict=True)}
        return lookup

    even_lookup = _shade_lookup(even_rings, even_cmap)
    odd_lookup = _shade_lookup(odd_rings, odd_cmap)

    colors = np.empty((ring_id.shape[0], 4), dtype=float)
    for r in np.unique(ring_id):
        mask = ring_id == r
        colors[mask] = even_lookup[r] if r % 2 == 0 else odd_lookup[r]

    return colors


def plot_radial_graph(
    data,
    figsize=(20, 20),
    node_size=3.0,
    same_edge_color="#001942",
    out_edge_color="#F75A00",
    same_edge_alpha=0.75,
    out_edge_alpha=0.75,
    same_linewidth=0.3,
    out_linewidth=0.4,
    show_nodes=True,
    save_path="radial_graph.png",
    dpi=200,
):
    pos = data.pos.numpy()
    ring_id = data.ring_id.numpy()
    num_rings = int(ring_id.max()) + 1

    node_colors = _ring_parity_colors(ring_id, num_rings)

    fig, ax = plt.subplots(figsize=figsize)

    # --- same-ring edges ---
    if data.same_edge_index.numel() > 0:
        same_idx = data.same_edge_index.numpy()
        same_segs = np.stack([pos[same_idx[0]], pos[same_idx[1]]], axis=1)  # (E, 2, 2)
        same_lc = LineCollection(
            same_segs,
            colors=same_edge_color,
            linewidths=same_linewidth,
            alpha=same_edge_alpha,
            zorder=1,
        )
        ax.add_collection(same_lc)

    # --- outward edges ---
    if data.out_edge_index.numel() > 0:
        out_idx = data.out_edge_index.numpy()
        out_segs = np.stack([pos[out_idx[0]], pos[out_idx[1]]], axis=1)  # (E, 2, 2)
        out_lc = LineCollection(
            out_segs,
            colors=out_edge_color,
            linewidths=out_linewidth,
            alpha=out_edge_alpha,
            zorder=2,
        )
        ax.add_collection(out_lc)

    # --- nodes ---
    if show_nodes:
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=node_colors,
            s=node_size,
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(pos[:, 0].min() - 1, pos[:, 0].max() + 1)
    ax.set_ylim(pos[:, 1].min() - 1, pos[:, 1].max() + 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # match image-style (row0 at top) grid coordinates
    ax.axis("off")

    n_nodes = pos.shape[0]
    n_same = data.same_edge_index.shape[1]
    n_out = data.out_edge_index.shape[1]
    ax.set_title(
        f"Radial graph: {n_nodes} nodes, {n_same} same-ring edges, {n_out} outward edges",
        fontsize=14,
    )

    legend_handles = [
        plt.Line2D([0], [0], color=same_edge_color, lw=2, label="same-ring edges"),
        plt.Line2D([0], [0], color=out_edge_color, lw=2, label="outward edges"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Example usage: swap grid_size/num_rings down if this is too slow/dense.
    data = build_radial_graph(
        grid_size=129,
    )
    plot_radial_graph(
        data,
        figsize=(24, 24),
        node_size=1.5,
        save_path="radial_graph.png",
        dpi=220,
    )
