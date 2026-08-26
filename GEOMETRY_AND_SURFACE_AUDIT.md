# Geometry and surface audit

Raw H5M SHA-256: `7c710fe3dd261ce7f46e5d08b4f9d2924513994ccb6229468933d0b91b2cd7cb`

Canonical fingerprint: `4f3588b5bdbc80977ed69cdb32188e2bd6ea4e2a414a8338801a0824fbf2c709`

## Independent reload and native checks

- PyMOAB reload: PASS (169104 vertices; 339496 triangles).
- PyDAGMC reload: PASS (44 volumes; 269 surfaces).
- `check_watertight`: PASS; zero unmatched edges, unsealed surfaces, and unsealed volumes.
- Native `overlap_check`: FAIL; 20 reported locations, including 7 nonadjacent-volume findings.
- OpenMC geometry debug: BLOCKED_LOCAL_NUCLEAR_DATA_ABSENT.

The 500k transport receipt independently records zero lost particles and zero DAGMC navigation errors, but this does not erase the native overlap findings or replace the requested geometry-debug mode.

## Magnet topology

- 18 explicit casing/winding pairs: PASS.
- All winding-pack shells closed: True.
- All external-only casing shells closed: False.
- Internal casing/winding interfaces excluded: True.
- Seeded ray first-hit identities: PASS (344 rays).
- Direct winding-pack-to-vacuum faces: 28 (gate FAIL).

## Native overlap findings

| Volumes | Shared DAGMC surface | Classification | Location (cm) |
|---|---|---|---|
| 1 / 43 | 1,8,9 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 379.6, 1199, 3.246 |
| 2 / 43 | 10,11 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 376.5, 1189, 7.371 |
| 3 / 43 | 12,13 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 1266, 367.1, 203.7 |
| 4 / 43 | 14,15 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 343.6, 1185, -19.74 |
| 5 / 17 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 1166, 341.1, 104.7 |
| 5 / 43 | 16,17 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 1274, 420.8, 242.7 |
| 6 / 17 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 1146, 362.9, 100.3 |
| 6 / 18 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 1151, 333.7, 92.71 |
| 6 / 19 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 1089, 498.5, 92.62 |
| 6 / 29 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 438, 1096, -75.37 |
| 6 / 31 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 305, 1157, -84.02 |
| 6 / 32 | none | UNINTENDED_NONADJACENT_VOLUME_OVERLAP | 333.7, 1151, -92.71 |
| 6 / 43 | 7,18,19 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 336, 1159, -114.1 |
| 17 / 43 | 84,85,86,87,88,89,90,91,92,97 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 1153, 318.3, 83.49 |
| 18 / 43 | 102,103 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 1132, 342.2, 88.61 |
| 19 / 43 | 104,105,106,107,108,109,110,111,112,117 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 1096, 477.5, 92.46 |
| 29 / 43 | 164,165,166,167,168,169,170,171,172,177 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 455.8, 1091, -94.25 |
| 30 / 43 | 182,183 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 443.8, 1077, -121.9 |
| 31 / 43 | 184,185,186,187,188,189,190,191,192,197 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 293.4, 1153, -71.47 |
| 32 / 43 | 202,203 | COINCIDENT_SHARED_BOUNDARY_CONTACT | 313.5, 1153, -92.12 |

## Classification

`GEOMETRY_FULL_ASSEMBLY_PASS = FAIL_NATIVE_OVERLAPS_AND_GEOMETRY_DEBUG_NOT_COMPLETED`

`WINDING_PACK_ENVELOPES_PASS = PASS`

`OUTER_CASING_ENVELOPES_PASS = FAIL_16_OF_18_OPEN_AFTER_INTERNAL_FACE_EXCLUSION`
