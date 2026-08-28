"""Export validated ParaStell radiation results for downstream consumers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parastell.downstream_response_export import (
    build_downstream_exports,
    write_downstream_exports,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--ownership-contribution-id", required=True)
    parser.add_argument("--delayed-photon-source-id")
    arguments = parser.parse_args()
    payload = json.loads(
        arguments.handoff.resolve(strict=True).read_text(encoding="utf-8")
    )
    exports = build_downstream_exports(
        payload,
        ownership_contribution_id=arguments.ownership_contribution_id,
        delayed_photon_source_id=arguments.delayed_photon_source_id,
    )
    paths = write_downstream_exports(arguments.output_directory, exports)
    print(json.dumps([str(path.resolve()) for path in paths], indent=2))


if __name__ == "__main__":
    main()
