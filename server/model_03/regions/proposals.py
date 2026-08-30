"""Layer 2a: turn the map into region proposals with descriptors.

A region is a connected run of confident "likely AI" pixels, grown slightly into
the surrounding uncertain band. The growth is deliberate: a real tampered area
does not end at a threshold crossing, its evidence fades out, and clipping the
region at the hi-threshold contour would systematically cut off exactly the
blending seam the specialists want to measure.

Each region carries the descriptors the router decides on. They are cheap
statistics of the map and the geometry, not learned features -- the router is a
handful of rules, and its inputs are chosen so a person can read a routing
decision and check it:

    area_frac              how much of the image this region covers
    mean_score, p90_score  how strong the evidence is
    scale_profile          per-scale mean score -- fine-only vs. all-scale
    scale_disagreement     spread across scales inside the region
    boundary_sharpness     how fast the map falls off at the region rim
    compactness            4*pi*area / perimeter^2; 1.0 is a disc
    fill_ratio             area / bbox area; low means a stringy, diffuse blob
    uncertain_halo_frac    how much of the surrounding ring is uncertain
    touches_border         a region running off-frame reads as outpainting

Regions are ranked by evidence mass (area x strength) rather than by score
alone, so a large moderately-suspicious area outranks three speckles that
happened to peak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mapper.heatmap import LABEL_AI, LABEL_UNCERTAIN, AILikelihoodMap
from regions.components import boundary, close_mask, dilate, label_components, ring


@dataclass
class Region:
    """One suspicious area, in working-resolution coordinates."""

    region_id: int
    mask: np.ndarray                    # (H, W) bool, full working-resolution frame
    bbox: tuple[int, int, int, int]     # x0, y0, x1, y1 (x1/y1 exclusive)
    area_px: int
    area_frac: float
    centroid: tuple[float, float]
    mean_score: float
    max_score: float
    p90_score: float
    scale_profile: dict[int, float]
    scale_disagreement: float
    boundary_sharpness: float
    compactness: float
    fill_ratio: float
    uncertain_halo_frac: float
    touches_border: bool
    support: float
    meta: dict = field(default_factory=dict)

    def crop_box(self, pad: int = 0, bounds: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
        """The bbox padded by `pad`, clipped to (width, height) if given."""
        x0, y0, x1, y1 = self.bbox
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        if bounds is not None:
            w, h = bounds
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
        return (x0, y0, x1, y1)

    def bbox_in_original(self, scale_factor: float) -> list[int]:
        return [int(round(v * scale_factor)) for v in self.bbox]

    def descriptors(self) -> dict:
        """The router's inputs, as a plain dict (also what lands in the report)."""
        return {
            "area_frac": self.area_frac,
            "mean_score": self.mean_score,
            "max_score": self.max_score,
            "p90_score": self.p90_score,
            "scale_profile": {str(k): v for k, v in sorted(self.scale_profile.items())},
            "scale_disagreement": self.scale_disagreement,
            "boundary_sharpness": self.boundary_sharpness,
            "compactness": self.compactness,
            "fill_ratio": self.fill_ratio,
            "uncertain_halo_frac": self.uncertain_halo_frac,
            "touches_border": self.touches_border,
            "support": self.support,
        }

    @property
    def evidence_mass(self) -> float:
        """Area-weighted strength -- the ranking key."""
        return self.area_frac * self.mean_score


def _boundary_sharpness(heat: np.ndarray, mask: np.ndarray) -> float:
    """Mean |gradient| of the map along the region's rim.

    A pasted object with a hard blending seam produces a fast fall-off; a
    diffuse global artefact produces a slow one. Computed on the rim only, so
    it describes the transition rather than the region's interior texture.
    """
    rim = boundary(mask)
    if not rim.any():
        return 0.0
    filled = np.nan_to_num(heat, nan=float(np.nanmean(heat)) if np.isfinite(np.nanmean(heat)) else 0.5)
    gy, gx = np.gradient(filled.astype(np.float64))
    return float(np.sqrt(gy ** 2 + gx ** 2)[rim].mean())


def _perimeter(mask: np.ndarray) -> float:
    """4-connected perimeter: boundary pixels weighted by exposed edges."""
    m = np.asarray(mask, dtype=bool)
    padded = np.pad(m, 1, mode="constant")
    exposed = (
        (~padded[:-2, 1:-1]).astype(np.int32)
        + (~padded[2:, 1:-1]).astype(np.int32)
        + (~padded[1:-1, :-2]).astype(np.int32)
        + (~padded[1:-1, 2:]).astype(np.int32)
    )
    return float(exposed[m].sum())


def extract_regions(
    amap: AILikelihoodMap,
    min_area_frac: float = 0.004,
    grow_into_uncertain: int = 3,
    close_radius: int = 2,
    halo_radius: int = 6,
    max_regions: int = 8,
) -> list[Region]:
    """Extract, filter and rank suspicious regions from a map.

    `min_area_frac` is the noise floor: below it a component is speckle from
    patch jitter, not an object. It is a *fraction* rather than a pixel count so
    the same setting behaves the same way on a thumbnail and on a 4K upload.
    """
    heat = amap.heat
    labels = amap.labels
    height, width = labels.shape
    total = float(height * width)

    core = labels == LABEL_AI
    if not core.any():
        return []

    # Grow the confident core into adjacent uncertain territory, but never into
    # confident non-AI -- an unrestricted dilation would let one region annex
    # the whole frame and take the fusion stage with it. The same restriction is
    # re-applied after closing, since closing dilates too.
    allowed = core | (labels == LABEL_UNCERTAIN)
    grown = (dilate(core, grow_into_uncertain) & allowed) if grow_into_uncertain > 0 else core.copy()
    if close_radius > 0:
        grown = close_mask(grown, close_radius) & allowed

    comp_labels, count = label_components(grown, connectivity=8)
    disagreement = amap.scale_disagreement()

    regions: list[Region] = []
    for cid in range(1, count + 1):
        mask = comp_labels == cid
        area_px = int(mask.sum())
        if area_px / total < min_area_frac:
            continue

        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1

        vals = heat[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue

        halo = ring(mask, halo_radius)
        halo_uncertain = (
            float((labels[halo] == LABEL_UNCERTAIN).mean()) if halo.any() else 0.0
        )

        perim = _perimeter(mask)
        bbox_area = float((x1 - x0) * (y1 - y0))

        regions.append(
            Region(
                region_id=len(regions) + 1,
                mask=mask,
                bbox=(x0, y0, x1, y1),
                area_px=area_px,
                area_frac=area_px / total,
                centroid=(float(xs.mean()), float(ys.mean())),
                mean_score=float(vals.mean()),
                max_score=float(vals.max()),
                p90_score=float(np.percentile(vals, 90)),
                scale_profile={
                    scale: float(np.nanmean(np.where(mask, smap, np.nan)))
                    for scale, smap in amap.per_scale.items()
                },
                scale_disagreement=float(disagreement[mask].mean()),
                boundary_sharpness=_boundary_sharpness(heat, mask),
                compactness=float(4.0 * np.pi * area_px / (perim ** 2)) if perim > 0 else 0.0,
                fill_ratio=area_px / bbox_area if bbox_area > 0 else 0.0,
                uncertain_halo_frac=halo_uncertain,
                touches_border=bool(x0 == 0 or y0 == 0 or x1 == width or y1 == height),
                support=float(amap.support[mask].mean()),
            )
        )

    regions.sort(key=lambda r: r.evidence_mass, reverse=True)
    regions = regions[:max_regions]
    for new_id, region in enumerate(regions, start=1):
        region.region_id = new_id
    return regions
