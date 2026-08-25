from dataclasses import dataclass

import pytest

from parastell.geometry_overlap import audit_native_cad_overlaps


@dataclass(frozen=True)
class _Box:
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


class _Solid:
    def __init__(self, bounds):
        self._bounds = _Box(*bounds)

    def BoundingBox(self):
        return self._bounds

    def Volume(self):
        box = self._bounds
        return (
            (box.xmax - box.xmin)
            * (box.ymax - box.ymin)
            * (box.zmax - box.zmin)
        )

    def intersect(self, other):
        left = self._bounds
        right = other._bounds
        lower = (
            max(left.xmin, right.xmin),
            max(left.ymin, right.ymin),
            max(left.zmin, right.zmin),
        )
        upper = (
            min(left.xmax, right.xmax),
            min(left.ymax, right.ymax),
            min(left.zmax, right.zmax),
        )
        if any(high <= low for low, high in zip(lower, upper)):
            return _Solid((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return _Solid((*lower, *upper))


def test_native_overlap_audit_allows_shared_faces_and_separated_solids():
    result = audit_native_cad_overlaps(
        [
            _Solid((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
            _Solid((1.0, 0.0, 0.0, 2.0, 1.0, 1.0)),
            _Solid((3.0, 0.0, 0.0, 4.0, 1.0, 1.0)),
        ],
        ["first", "touching", "separated"],
    )
    assert result["status"] == "PASS"
    assert result["pair_count"] == 3
    assert result["aabb_candidate_pair_count"] == 1
    assert result["positive_volume_overlap_count"] == 0


def test_native_overlap_audit_rejects_positive_shared_volume():
    with pytest.raises(ValueError, match="positive-volume overlaps"):
        audit_native_cad_overlaps(
            [
                _Solid((0.0, 0.0, 0.0, 2.0, 1.0, 1.0)),
                _Solid((1.0, 0.0, 0.0, 3.0, 1.0, 1.0)),
            ],
            ["first", "overlapping"],
        )
