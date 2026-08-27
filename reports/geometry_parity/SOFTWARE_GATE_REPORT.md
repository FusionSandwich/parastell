# Software gate report

## Passing checks

- Focused parity suite after final formatting: 15 passed, 0 failed, 6 dependency warnings; 221.28 s.
- Project CI-equivalent suite (`cd tests; python -m pytest -q .`): 58 passed, 7 skipped, 0 failed; 2,256.75 s. The skips are the existing Cubit/license-only tests.
- Changed-file Black 25.12.0 check: 7 files unchanged; pass.
- `python -m compileall -q parastell examples tests`: pass.
- Build 1.5.0 with the existing backend and `--no-isolation`: source distribution and wheel built successfully.
- `parastell -h`: pass.
- `python -m parastell -h`: pass.
- `python -c "import parastell; import parastell.reference_geometry"`: pass in the qualified container.
- `git diff --check`: pass.
- All committed compact JSON reports parse successfully.

## Separately classified baseline limitations

The literal root invocation `python -m pytest -q tests` cannot collect two untouched mainline modules because they open `files_for_tests/...` relative to the current directory. The fixture files exist. The repository's own CI explicitly changes into `tests` before running pytest; that supported invocation passes as reported above.

Repository-wide `black --check .` reports one untouched mainline file, `parastell/nwl_utils.py`, that would be reformatted. All Prompt-7A files pass Black. The unrelated baseline file was preserved rather than changed opportunistically.

The build reports upstream setuptools deprecation warnings for the TOML license table and license classifier. They are not build failures. No dependency was acquired and no environment was modified.

## Classification

`PASS_CHANGED_SCOPE_WITH_RECORDED_BASELINE_LIMITATIONS`

This software result does not override the physical geometry failures in `GEOMETRY_DEBUG_REPORT.json`.
