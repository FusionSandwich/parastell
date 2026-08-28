# Candidate A1 source-CAD decision

`A1_breeder_only_clearance_target_5_cm` is rejected and may not advance to DAGMC or OpenMC qualification.

The hardened exact-solid audit completed all 276 component/magnet pair checks. It found no unintended Boolean intersection, no duplicate magnet, exact canonical magnet identity with the independent vanilla build, and exact semantic source-mesh identity. However, the measured global vacuum-vessel-to-magnet clearance is **4.493516669986348 cm**, below the preregistered **5.0 cm** minimum. The limiting witnesses are the symmetry-related `magnet-0008` and `magnet-0009` locations.

The audit also exposed a separate diagnostic defect: OpenCascade commonly leaves `TopoDS_Solid.Closed()` false even when its bounding shell is closed and the solid is valid. The source-CAD auditor now requires a valid `Solid` with closed bounding shells. This correction removes a false inventory failure but cannot rescue the true clearance failure.

The next candidate remains in the preferred local breeder-only repair class. It uses the same frozen VMEC, coils, magnet solids/transforms, layer order, thickness matrices for the first wall/back wall/shield/vessel, source mesh definition, and 90-degree extent, with a slightly tighter local clearance constraint derived from the exact A1 shortfall.
