# Failed feature root cause

The seven failed-branch witnesses are real nonadjacent physical-volume findings, but the casing/winding split is not their sole origin. The untouched public R1 build independently reports four vacuum-vessel/original-magnet overlaps at the same semantic coils: magnet-0005, magnet-0006, magnet-0011, and magnet-0012.

R1 uses one homogenized solid for each coil. The failed branch retains the same intended 40 × 50 cm outer envelope, then subtracts a 5 cm casing and retains the 30 × 40 cm inner winding pack. This split fragments the R1 magnet-0005 and magnet-0012 intersections into separate casing and winding-pack witnesses. It also reports one shield/magnet-0005 witness that the coarser R1 native check did not report. Exact penetration depths and intersection volumes were not computed and are classified `NOT_QUANTIFIED`.

The failed artifact did **not** use the historical 39 cm shield workaround; its exact config uses the public 50 cm shield. No evidence for a 38.5 cm input was found. The four persistent R1 overlaps mean shrinking the shield would change the reference physics and is not an authorized parity repair.

The strongest software-path defect is an incomplete validation claim: the failed branch’s “native combined CAD overlap PASS” used 36 solids and 630 pairs, exactly `C(36,2)`. Those were the split magnet solids only. The PyDAGMC in-vessel volumes were absent from that CadQuery collection, so reactor/magnet intersections were never audited before export.

Post-export addition of interstitial volume 43 and graveyard volume 44 changed the H5M and exterior topology, but none of the seven pairs involves those volumes. It is not the cause of the physical intersections. Complete volume senses and watertightness pass, and no duplicated magnets were found.

## Casing topology

Classification: `E — combination of B, C, and D`.

- B: 16 of 18 clipped casing-external subsets are valid open sleeves with exposed winding ends; only magnets 0008 and 0009 close as casing-only external envelopes.
- C: the representative outer-casing declaration appended winding-only surfaces 102 and 103 and mislabeled the union.
- D: the global split is unnecessary for exact R1 parity.
- Not A: each complete casing volume is watertight. The open object is the selected external-only subset, not the casing solid itself.

The clean producer path therefore uses the complete outer boundary of each original homogenized R1 magnet and never forces casing closure by appending winding-only caps.

The full pair table, locations, R1 correspondences, and hypothesis audit are in `FAILED_FEATURE_ROOT_CAUSE.json`.

