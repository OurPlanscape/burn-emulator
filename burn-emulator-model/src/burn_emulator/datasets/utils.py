def compute_bounds(
    y: int, x: int, h: int, w: int, window_size: int
) -> tuple[int, int, int, int, slice, slice]:
    """Window bounds + slices centered on (y, x), clamped to an (h, w) raster."""
    s = window_size // 2
    off = window_size % 2
    ymin, ymax = max(0, y - s), min(y + s + off, h)
    xmin, xmax = max(0, x - s), min(x + s + off, w)
    return ymin, ymax, xmin, xmax, slice(ymin, ymax), slice(xmin, xmax)


def compute_padding(
    ymin: int, ymax: int, xmin: int, xmax: int, window_size: int
) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Pad amounts and F.pad specs to restore a border-clamped window to window_size."""
    ydiff = window_size - (ymax - ymin)
    xdiff = window_size - (xmax - xmin)
    ypad = (0, 0, ydiff, 0) if ymin == 0 else (0, 0, 0, ydiff)
    xpad = (xdiff, 0, 0, 0) if xmin == 0 else (0, xdiff, 0, 0)
    return ydiff, xdiff, ypad, xpad


def compute_crop_region(
    bounds: tuple[int, int, int, int], diffs: tuple[int, int]
) -> tuple[int, int, int, int, int, int]:
    """Inverse of compute_bounds/compute_padding: raster box + the source offset within a
    padded window that maps onto it. Returns (y0, y1, x0, x1, ysrc, xsrc)."""
    y0, y1, x0, x1 = (int(v) for v in bounds)
    yd, xd = (int(v) for v in diffs)
    ysrc = yd if (yd > 0 and y0 == 0) else 0
    xsrc = xd if (xd > 0 and x0 == 0) else 0
    return y0, y1, x0, x1, ysrc, xsrc
