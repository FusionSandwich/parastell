#!/usr/bin/env bash
set -euo pipefail
umask 077

PRE=/home/apollon/josma/data/parastell_prompt7r_geometry/runtime_stack_preflight_v3_20260827T133901Z
SCI=/home/apollon/josma/data/parastell_prompt7r_geometry/refacet_v5_remote_20260827T133901Z
CODE="$PRE/code"
SOURCE=/home/apollon/josma/data/parastell_prompt7r_geometry/sourcecad_v16_remote_20260827T112400Z/inputs/geometry-recovery-20260827/candidate_A1_target575_coarse_full_20260827T051300
AUDIT_ROOT=/home/apollon/josma/data/parastell_prompt7r_geometry/sourcecad_v16_remote_20260827T112400Z
PHYSICAL="$AUDIT_ROOT/inputs/geometry-recovery-20260827/candidate_A1_target575_physical_v1_20260827T094210/candidate_physical_change.json"
CRITERIA="$AUDIT_ROOT/inputs/reports/geometry_recovery/CANDIDATE_ACCEPTANCE_CRITERIA.json"
REFERENCE="$AUDIT_ROOT/inputs/geometry-recovery-20260827/vanilla_main_reference/20260827T021126/VANILLA_BUILD_MANIFEST.json"
PROTOCOL="$PRE/controls/FACETING_COMPARISON_PROTOCOL.json"
PY=/home/apollon/josma/data/transport/bin/python
RECEIPT="$PRE/runtime_receipt.json"
VALIDATION="$PRE/runtime_validation.json"
CONTROL="$PRE/controls/REFACET_V5_COARSE_CONTROL.json"
PID_FILE="$PRE/refacet_v5_coarse.pid"

test "$(hostname)" = bateman
test ! -e "$SCI"
test ! -L "$SCI"
test ! -e "$SCI/coarse"
test ! -e "$PID_FILE"
test ! -L "$PID_FILE"
test "$(pgrep -fc '[r]efacet_source_cad_candidate.py.*refacet_v5_remote_20260827T133901Z' || true)" = 0
test "$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)" -ge 134217728
test "$(df -PB1 "$PRE" | awk 'NR==2 {print $4}')" -ge 21474836480

test "$(sha256sum "$PY" | awk '{print $1}')" = 63770468d7041b46aa7fc01ad9a17b4e616dbbb7d613f5470e1cdc5359c83a86
test "$(sha256sum "$PRE/controls/runtime_probe_v3_code.tar.gz" | awk '{print $1}')" = 961cb634de80c9a404377b5d2a71946677151ede22f9a10dc2d8eb2be442ad4d
test "$(sha256sum "$CODE/parastell/source_cad_refaceting.py" | awk '{print $1}')" = 3e91d876713f4554c1f6c4f8e80b16ff4ad72ad1d0ab8eddf2bf8c49f489f26f
test "$(sha256sum "$CODE/scripts/refacet_source_cad_candidate.py" | awk '{print $1}')" = e8e0ba4e5932f0a4718089d8abbca4a60d17dae9265c139be03d2ee89238289f
test "$(sha256sum "$RECEIPT" | awk '{print $1}')" = 2f6260b1fd45b1b7e8a3d9d079a4faf97158e03106352c9699990d508f8c85c7
test "$(sha256sum "$VALIDATION" | awk '{print $1}')" = 946ba2126615f13bbc14a8ed86cba461811d8d61bc9701ad5eb861ab40881da6
test "$(sha256sum "$CONTROL" | awk '{print $1}')" = 1d599ea9cb0357f5fb721d8bee5f805ea369cde7d7df6e82c26bde8c30d7184d
test "$(sha256sum "$SOURCE/dagmc.h5m" | awk '{print $1}')" = 8a2d1930cc03a82269feeedef60267197581559c99aa69a021235a7ac7fafa90
test "$(sha256sum "$SOURCE/candidate_build_manifest.json" | awk '{print $1}')" = 36fed787de68ba1ed963b8d0bf212e9f35615ad14abefad39b8ef3e16a755f91
test "$(sha256sum "$AUDIT_ROOT/audit_output/source_cad_audit.json" | awk '{print $1}')" = e47f34884fc04b4a353b96a4e6a18928bdb4d501e26f3b6793b213c80ae2a9aa
test "$(sha256sum "$AUDIT_ROOT/audit_output/source_cad_audit_seal.json" | awk '{print $1}')" = 397968a853f7c5a5d75d95c3143cebaa73b2280e8b6be30475a3a9db3d5ddd3c
test "$(sha256sum "$PHYSICAL" | awk '{print $1}')" = 7411d2564b9210cf2d0c7d99e0691a346083fd5ca202c4529c2b609d12465141
test "$(sha256sum "$REFERENCE" | awk '{print $1}')" = 48e38a68edee72839e57f90c884f4c48a04da4b9b9ab22160411f22d2e0a4935
test "$(sha256sum "$CRITERIA" | awk '{print $1}')" = 092315725cfd06e64fa403cc8f484a8135f4920a5af93bcdccc797dfb41134ca
test "$(sha256sum "$PROTOCOL" | awk '{print $1}')" = d1f4216a279ffe6608e3b334d85828bb126036833e1d17aa241880c73ec8cacf
test "$(sha256sum "$CODE/parastell/__init__.py" | awk '{print $1}')" = 77564a6ccef7e8d0c545c92a79a25b218d4bc8e58769c43b115eacf19b4690e0
test "$(sha256sum "$CODE/parastell/reference_geometry.py" | awk '{print $1}')" = cb6d172e99c5699514936fc8dd390a3c1427b13473567982ae3eeaab679f157b
test "$(sha256sum "$CODE/parastell/native_dagmc_topology.py" | awk '{print $1}')" = a4ee8a607fb58fb58c95eb540b7e76bc729c75dc59b00182457eadb45f64b9c7

mkdir -- "$SCI"

set -o noclobber
printf '%s\n' "$$" > "$PID_FILE"
set +o noclobber

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32

exec timeout --signal=TERM --kill-after=120s 10800s \
  /usr/bin/time -v \
  "$PY" "$CODE/scripts/refacet_source_cad_candidate.py" \
  "$SOURCE" \
  "$AUDIT_ROOT/audit_output/source_cad_audit.json" \
  "$AUDIT_ROOT/audit_output/source_cad_audit_seal.json" \
  "$PHYSICAL" \
  "$SCI/coarse" \
  --acceptance-criteria "$CRITERIA" \
  --faceting-protocol "$PROTOCOL" \
  --runtime-receipt "$RECEIPT" \
  --level coarse \
  --threads 32 \
  --expected-source-manifest-sha256 36fed787de68ba1ed963b8d0bf212e9f35615ad14abefad39b8ef3e16a755f91 \
  --expected-source-audit-sha256 e47f34884fc04b4a353b96a4e6a18928bdb4d501e26f3b6793b213c80ae2a9aa \
  --expected-source-audit-seal-sha256 397968a853f7c5a5d75d95c3143cebaa73b2280e8b6be30475a3a9db3d5ddd3c \
  --expected-physical-change-report-sha256 7411d2564b9210cf2d0c7d99e0691a346083fd5ca202c4529c2b609d12465141 \
  --expected-reference-manifest-sha256 48e38a68edee72839e57f90c884f4c48a04da4b9b9ab22160411f22d2e0a4935 \
  --expected-acceptance-criteria-sha256 092315725cfd06e64fa403cc8f484a8135f4920a5af93bcdccc797dfb41134ca \
  --expected-faceting-protocol-sha256 d1f4216a279ffe6608e3b334d85828bb126036833e1d17aa241880c73ec8cacf \
  --expected-runtime-receipt-sha256 2f6260b1fd45b1b7e8a3d9d079a4faf97158e03106352c9699990d508f8c85c7 \
  --expected-attempt-id refacet_v5_remote_20260827T133901Z \
  --expected-launch-nonce 1e48fbda07a40027d31107cbec7b6573fbf7d1583e943caa5a4b292a67c73ccd \
  --expected-host-alias poly-bateman
