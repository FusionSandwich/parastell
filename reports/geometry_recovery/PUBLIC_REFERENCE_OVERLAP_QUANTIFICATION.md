# Public-reference overlap quantification

The current public ParaStell example is a negative physical reference. Exact source-CAD Boolean intersections independently confirm that four native 40 × 50 cm homogenized magnets penetrate the 10 cm vacuum-vessel solid. This is not a faceting-only diagnosis.

| Magnet | CAD intersection volume (cm³) | Boolean-result boundary area (cm²) | Intersection AABB spans (cm) | DAGMC witness (cm) |
|---|---:|---:|---|---|
| magnet-0005 | 5092.197638 | 3649.566180 | 15.686 × 85.775 × 58.713 | 1160.39, 380.476, 128.459 |
| magnet-0006 | 21.589483 | 168.216703 | 2.646 × 50.185 × 14.693 | 1095.15, 498.581, 92.5898 |
| magnet-0011 | 21.591609 | 167.838750 | 50.185 × 2.646 × 14.693 | 476.445, 1095.28, -86.2664 |
| magnet-0012 | 5113.576116 | 3673.516439 | 85.775 × 15.686 × 58.713 | 364.23, 1158.37, -121.329 |

Each Boolean result is one connected solid. The remaining 14 magnet/vessel pairs have zero Boolean intersection volume.

The AABB spans are not penetration depths. The reported area is the entire Boolean-result boundary area, not a material-interface contact area. A signed-normal penetration metric, local clearance sampling, local radial-build attribution, `overlap_check -p 1/-p 4`, and a refined matched-faceting build remain required before Phase 1 is complete.

Disposition: `REJECTED_PHYSICAL_OVERLAP_REFERENCE_ONLY`.
