"""Tests for data/download_cifake.py's REORGANIZATION logic (find_class_dirs,
layout_from_source) against a synthetic mimicked CIFAKE-style folder tree --
NOT the actual kagglehub download, which needs network/credentials and isn't
exercised here. This is deliberately the part most likely to break in practice
(depends on the exact folder names/casing/nesting Kaggle serves), so it's the
part worth testing without needing a real download.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.download_cifake import find_class_dirs, layout_from_source  # noqa: E402


def _make_fake_cifake_tree(root: Path, n_per_split_class: int = 3) -> None:
    """Mimics CIFAKE's actual on-disk shape: <root>/{train,test}/{REAL,FAKE}/*.jpg"""
    for split in ("train", "test"):
        for label in ("REAL", "FAKE"):
            d = root / split / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_split_class):
                (d / f"{label.lower()}_{split}_{i}.jpg").write_bytes(b"fake-jpeg-bytes")


def test_find_class_dirs_locates_both_splits():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_cifake_tree(root)
        found = find_class_dirs(root)
        assert len(found["real"]) == 2, "expected train/REAL and test/REAL"
        assert len(found["fake"]) == 2, "expected train/FAKE and test/FAKE"


def test_find_class_dirs_is_case_insensitive():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train" / "Real").mkdir(parents=True)
        (root / "train" / "fake").mkdir(parents=True)
        found = find_class_dirs(root)
        assert len(found["real"]) == 1
        assert len(found["fake"]) == 1


def test_layout_from_source_merges_splits_without_collisions():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        out = Path(tmp) / "out"
        _make_fake_cifake_tree(source, n_per_split_class=3)

        counts = layout_from_source(source, out, symlink=False)  # copy mode: works on any filesystem
        assert counts == {"real": 6, "fake": 6}, f"expected 6+6 (train+test merged), got {counts}"

        real_files = sorted((out / "real").iterdir())
        fake_files = sorted((out / "fake").iterdir())
        assert len(real_files) == 6
        assert len(fake_files) == 6
        # Both splits should be represented (prefix-based naming keeps them distinct).
        assert any(p.name.startswith("train_") for p in real_files)
        assert any(p.name.startswith("test_") for p in real_files)


def test_layout_from_source_raises_when_no_class_dirs_found():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "empty_source"
        source.mkdir()
        (source / "some_other_folder").mkdir()
        try:
            layout_from_source(source, Path(tmp) / "out")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError when no REAL/FAKE dirs are found")


def test_layout_from_source_is_idempotent_on_rerun():
    """Running it twice (e.g. after adding --copy, or a partial failure) should
    not error or duplicate files."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        out = Path(tmp) / "out"
        _make_fake_cifake_tree(source, n_per_split_class=2)

        first = layout_from_source(source, out, symlink=False)
        second = layout_from_source(source, out, symlink=False)
        assert first == {"real": 4, "fake": 4}
        # Second run should find everything already exists -> 0 newly written.
        assert second == {"real": 0, "fake": 0}
        assert len(list((out / "real").iterdir())) == 4
        assert len(list((out / "fake").iterdir())) == 4


if __name__ == "__main__":
    import traceback

    tests = {name: obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)}
    passed, failed = 0, 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
