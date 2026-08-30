"""Connected components and the small morphology the region stage needs.

Written against numpy alone rather than `scipy.ndimage.label` / `cv2` on
purpose: this is the stage between the map and every downstream decision, it is
about fifty lines, and keeping it dependency-free means the region and routing
logic can be unit-tested in an environment with no scipy, no cv2 and no torch --
which is where the fast test loop lives (see tests/).

Two-pass union-find with path compression: pass one assigns provisional labels
scanning north/west (plus the diagonals under 8-connectivity) and records
equivalences, pass two resolves them. 8-connectivity is the default because a
tampered region traced by a thresholded heatmap frequently pinches to a
diagonal pixel bridge, and 4-connectivity would split it into two regions that
then get routed separately and double-counted at fusion.
"""
from __future__ import annotations

import numpy as np


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = [0]  # index 0 is the background sentinel

    def make(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def label_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label the True pixels of `mask`. Returns (labels, count); 0 is background.

    Uses `scipy.ndimage.label` when scipy is installed and the pure-numpy
    implementation otherwise. The two agree exactly (asserted in
    tests/test_regions.py); the difference is speed, and it is not marginal --
    the Python version walks every foreground pixel, so on a whole-image
    synthesis verdict (where the mask covers most of a 1024x768 frame) it costs
    ~2s against scipy's ~0.02s. The fallback exists so the region and routing
    logic stays testable in a bare environment, not as a serious runtime path.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    try:
        from scipy import ndimage
    except ImportError:
        return _label_components_numpy(mask, connectivity)

    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    )
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return labels.astype(np.int32), int(count)


def _label_components_numpy(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Two-pass union-find labelling, dependency-free. See `label_components`."""
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=np.int32)
    uf = _UnionFind()

    neighbours = [(-1, 0), (0, -1)] if connectivity == 4 else [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for y in range(h):
        row = m[y]
        if not row.any():
            continue
        for x in np.flatnonzero(row):
            found = [
                labels[y + dy, x + dx]
                for dy, dx in neighbours
                if 0 <= y + dy < h and 0 <= x + dx < w and labels[y + dy, x + dx]
            ]
            if not found:
                labels[y, x] = uf.make()
            else:
                labels[y, x] = min(found)
                for other in found:
                    uf.union(labels[y, x], other)

    if labels.max() == 0:
        return labels, 0

    # Resolve equivalences and renumber 1..n so the ids are dense.
    roots = np.array([uf.find(i) for i in range(len(uf.parent))], dtype=np.int32)
    resolved = roots[labels]
    unique = np.unique(resolved[resolved > 0])
    remap = np.zeros(int(resolved.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return remap[resolved], int(unique.size)


def _shift_or(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys_dst = slice(max(0, dy), h + min(0, dy))
    xs_dst = slice(max(0, dx), w + min(0, dx))
    ys_src = slice(max(0, -dy), h + min(0, -dy))
    xs_src = slice(max(0, -dx), w + min(0, -dx))
    out[ys_dst, xs_dst] = mask[ys_src, xs_src]
    return out


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Square-structuring-element dilation, `radius` iterations of a 3x3."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(radius))):
        acc = out.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= _shift_or(out, dy, dx)
        out = acc
    return out


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dual of `dilate` -- erosion of a mask is dilation of its complement."""
    return ~dilate(~np.asarray(mask, dtype=bool), radius)


def close_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate then erode: bridges the pinholes threshold noise leaves inside a region."""
    return erode(dilate(mask, radius), radius)


def boundary(mask: np.ndarray) -> np.ndarray:
    """The one-pixel inner rim of `mask` -- where blending seams would be."""
    m = np.asarray(mask, dtype=bool)
    return m & ~erode(m, 1)


def ring(mask: np.ndarray, radius: int = 4) -> np.ndarray:
    """The band of `radius` pixels just *outside* `mask`.

    The specialists compare statistics inside a region against this ring rather
    than against the whole image: an inpainted patch differs from its immediate
    surroundings even when the image as a whole is heterogeneous, and it is that
    local discontinuity -- not a global one -- that a paste actually creates.
    """
    m = np.asarray(mask, dtype=bool)
    return dilate(m, radius) & ~m
