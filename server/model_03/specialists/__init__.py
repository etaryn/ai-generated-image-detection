"""Layer 2b -- the region specialists, and the registry the router names them from."""
from specialists.base import Specialist, SpecialistContext, SpecialistResult
from specialists.compositing import CompositingSpecialist
from specialists.face_edit import FaceEditSpecialist
from specialists.fallback import FallbackSpecialist
from specialists.faces import FaceBox, detect_faces, faces_available
from specialists.inpainting import InpaintingSpecialist
from specialists.synthesis import SynthesisSpecialist

SPECIALISTS = {
    "inpainting": InpaintingSpecialist,
    "synthesis": SynthesisSpecialist,
    "face_edit": FaceEditSpecialist,
    "compositing": CompositingSpecialist,
    "fallback": FallbackSpecialist,
}


def build_specialists(names=None) -> dict:
    """Instantiate the named specialists (all of them by default)."""
    names = list(names or SPECIALISTS)
    unknown = [n for n in names if n not in SPECIALISTS]
    if unknown:
        raise ValueError(f"unknown specialists {unknown}; expected from {sorted(SPECIALISTS)}")
    return {name: SPECIALISTS[name]() for name in names}


__all__ = [
    "CompositingSpecialist",
    "FaceBox",
    "FaceEditSpecialist",
    "FallbackSpecialist",
    "InpaintingSpecialist",
    "SPECIALISTS",
    "Specialist",
    "SpecialistContext",
    "SpecialistResult",
    "SynthesisSpecialist",
    "build_specialists",
    "detect_faces",
    "faces_available",
]
