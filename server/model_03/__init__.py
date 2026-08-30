"""model_03 -- region-aware AI image tamper forensics.

Find first, analyse second: build a calibrated AI-likelihood map over the image,
then route each suspicious region to a forensic specialist, then fuse the
regional findings into one image-level verdict.

See README.md for the pipeline, and for an honest account of which stages are
learned and which are hand-derived heuristics.
"""
