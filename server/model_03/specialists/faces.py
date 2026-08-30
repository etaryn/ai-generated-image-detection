"""Face location -- the router's one piece of semantic input.

Routing a region to a face specialist requires knowing where the faces are.
This uses OpenCV's Haar cascade when cv2 is installed and reports *nothing*
when it is not.

Reporting nothing is the important behaviour. The alternative -- guessing at
faces from skin-tone heuristics -- would route ordinary regions to the face
specialist and attach a "face manipulation" label to them, which is the single
most damaging thing this pipeline could get wrong: it is the finding a user is
most likely to act on and least able to check. A missing detector means the
face route is simply never taken, every such region falls through to the
inpainting or fallback specialist, and the report says the face route was
unavailable. Under-claiming is the correct failure here.

Haar cascades are also frontal-only and date from 2001. They are adequate for
"is there a face in this bounding box" and not adequate for anything else; the
`detector` field in the output records which one ran so a reader can weigh it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FaceBox:
    x0: int
    y0: int
    x1: int
    y1: int
    source: str

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, dtype=bool)
        m[max(0, self.y0) : self.y1, max(0, self.x0) : self.x1] = True
        return m

    def to_dict(self) -> dict:
        return {"bbox": [self.x0, self.y0, self.x1, self.y1], "source": self.source}


_CASCADE = None
_CASCADE_TRIED = False


def _load_cascade():
    """Load the frontal-face cascade once; return None if cv2 is absent."""
    global _CASCADE, _CASCADE_TRIED
    if _CASCADE_TRIED:
        return _CASCADE
    _CASCADE_TRIED = True
    try:
        import cv2  # noqa: F401

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        _CASCADE = None if cascade.empty() else cascade
    except Exception:
        _CASCADE = None
    return _CASCADE


def detect_faces(image: Image.Image, min_size_frac: float = 0.03) -> list[FaceBox]:
    """Detect frontal faces. Returns [] when no detector is installed."""
    cascade = _load_cascade()
    if cascade is None:
        return []

    import cv2

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    min_side = max(20, int(min(image.size) * min_size_frac))
    found = cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_side, min_side)
    )
    return [
        FaceBox(int(x), int(y), int(x + w), int(y + h), source="haar_frontalface")
        for (x, y, w, h) in found
    ]


def faces_available() -> bool:
    """Whether a face detector is installed at all -- reported in the output."""
    return _load_cascade() is not None


def overlap_fraction(mask: np.ndarray, face: FaceBox) -> float:
    """Fraction of `mask` that falls inside `face`'s box."""
    m = np.asarray(mask, dtype=bool)
    total = int(m.sum())
    if total == 0:
        return 0.0
    return float((m & face.mask(m.shape)).sum() / total)
