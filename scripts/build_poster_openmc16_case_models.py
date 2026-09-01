"""Build hash-bound OpenMC 0.16 models for staged poster material cases."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from parastell.parametric_openmc16_model import build_model
from parastell.reference_geometry import sha256_file


SCHEMA = "parastell.poster_openmc16_case_model_builds/v1.0.0"
DEFAULT_CASE_IDS = ("P00", "P01", "P02", "P03", "P05", "P06")


def _write_json(path: Path, payload: Any) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template_control", type=Path)
    parser.add_argument("case_directory", type=Path)
    parser.add_argument("cross_sections_xml", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--expected-case-plan-sha256", required=True)
    parser.add_argument("--expected-cross-sections-sha256", required=True)
    parser.add_argument("--particles", type=int, default=200)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=9_011_000)
    parser.add_argument("--case-ids", nargs="+", default=DEFAULT_CASE_IDS)
    args = parser.parse_args()

    template_path = args.template_control.resolve(strict=True)
    case_directory = args.case_directory.resolve(strict=True)
    catalog = args.cross_sections_xml.resolve(strict=True)
    output = args.output_directory.resolve()
    plan_path = case_directory / "POSTER_PARAMETRIC_CASE_PLAN.json"
    if sha256_file(template_path) != args.expected_template_sha256:
        raise ValueError("template-control hash mismatch")
    if sha256_file(plan_path) != args.expected_case_plan_sha256:
        raise ValueError("case-plan hash mismatch")
    if sha256_file(catalog) != args.expected_cross_sections_sha256:
        raise ValueError("cross-section catalog hash mismatch")
    if output.exists():
        raise FileExistsError(f"create-only output exists: {output}")
    if args.particles <= 0 or args.batches <= 0 or args.seed_base <= 0:
        raise ValueError("run controls must be positive")
    case_ids = list(args.case_ids)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be nonempty and unique")

    template = json.loads(template_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    declared = {row["case_id"]: row for row in plan.get("cases", [])}
    if any(case_id not in declared for case_id in case_ids):
        raise ValueError("requested case is absent from the staged plan")

    output.mkdir(parents=True, exist_ok=False)
    controls = output / "controls"
    models = output / "models"
    controls.mkdir()
    models.mkdir()
    results = []
    all_pass = True
    for offset, case_id in enumerate(case_ids, start=1):
        material_path = (case_directory / case_id / "materials.xml").resolve(
            strict=True
        )
        control = copy.deepcopy(template)
        control["inputs"]["materials_xml"] = {
            "path": str(material_path),
            "sha256": sha256_file(material_path),
        }
        control["inputs"]["cross_sections_xml"] = {
            "path": str(catalog),
            "sha256": args.expected_cross_sections_sha256,
        }
        control["run"] = {
            "particles": args.particles,
            "batches": args.batches,
            "seed": args.seed_base + offset,
            "statepoint_interval_batches": 1,
        }
        control_path = controls / f"{case_id}.json"
        control_sha = _write_json(control_path, control)
        row = {
            "case_id": case_id,
            "concept": declared[case_id].get("concept"),
            "control": {"path": str(control_path), "sha256": control_sha},
            "materials_xml": {
                "path": str(material_path),
                "sha256": sha256_file(material_path),
            },
            "cross_sections_xml": {
                "path": str(catalog),
                "sha256": args.expected_cross_sections_sha256,
            },
        }
        try:
            _, receipt = build_model(
                control_path, control_sha, models / case_id
            )
            receipt_path = (
                models / case_id / "PARAMETRIC_OPENMC16_MODEL_RECEIPT.json"
            )
            row.update(
                {
                    "status": "PASS",
                    "model_xml": receipt["model_xml"],
                    "model_receipt": {
                        "path": str(receipt_path),
                        "sha256": sha256_file(receipt_path),
                    },
                    "nuclear_data_manifest_sha256": receipt[
                        "nuclear_data_manifest"
                    ]["manifest_sha256"],
                    "physical_source_rate_n_per_s_for_modeled_90_degrees": receipt[
                        "source"
                    ][
                        "physical_rate_n_per_s_for_modeled_90_degrees"
                    ],
                    "statepoint_policy": receipt["statepoint_policy"],
                }
            )
        except Exception as error:  # preserve independent case evidence
            all_pass = False
            row.update(
                {
                    "status": "BLOCKED_MODEL_OR_NUCLEAR_DATA_COVERAGE",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        results.append(row)

    receipt = {
        "schema": SCHEMA,
        "status": "PASS" if all_pass else "BLOCKED_CASE_MODEL_BUILD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim": "BOUNDED_SMOKE_ONLY",
        "production_run_authorized": False,
        "template_control_sha256": args.expected_template_sha256,
        "case_plan_sha256": args.expected_case_plan_sha256,
        "cross_sections_xml_sha256": args.expected_cross_sections_sha256,
        "case_results": results,
    }
    _write_json(output / "POSTER_OPENMC16_CASE_MODEL_BUILDS.json", receipt)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
