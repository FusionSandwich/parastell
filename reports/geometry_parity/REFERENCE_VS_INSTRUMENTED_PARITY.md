# Reference versus instrumented parity

## Result

`BYTE_IDENTICAL_REUSE` — pass.

The clean instrumented mode opens the accepted R1 H5M as an immutable input and constructs OpenMC configuration outside that file. It does not regenerate, split, translate, rotate, scale, or post-process any physical solid.

| Check | R1 | Instrumented | Result |
|---|---:|---:|---|
| Raw H5M SHA-256 | `8741dd48fded42e8411816e56e3e5e10a29db26ddb785b4a389f8a38b09707a0` | same file and digest | PASS |
| Canonical fingerprint | `1c4a6c1fdb37f7bb9d7ef59ab99913884bde6df7b55977a53a68f2a037552bd1` | identical | PASS |
| Physical volumes | 24 | 24 | PASS |
| DAGMC surfaces | 142 | 142 | PASS |
| Homogenized magnets | 18 | 18 | PASS |
| Extra H5M volumes | 0 | 0 | PASS |
| H5M mutation | none | none | PASS |
| Source mesh | `300922f116cb8463a84f2f1b8c80eb359e89d70d3427e536a9bd4e500fb2bb3d` | unchanged external input | PASS |

The optional OpenMC CSG world sphere is recorded in the OpenMC model only. It is not written into the H5M and does not affect this parity result.

## Important limitation

Parity proves that the instrumented path is not the source of a physical change. It does not make the accepted input physically valid. Native `overlap_check` finds four true nonadjacent vacuum-vessel/magnet overlaps in the untouched R1 H5M, so the overall Prompt-7A geometry gate remains blocked.
