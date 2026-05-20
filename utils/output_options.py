from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputOptions:
    save_candidate_artifacts: bool
    save_all_results_csv: bool
    best_bitstreams_dir: Path | None


def resolve_output_options(
    *,
    output_dir: Path,
    best_bitstreams_dir: Path | None,
    best_only: bool,
    save_candidates: bool,
    save_all_results_csv: bool,
) -> OutputOptions:
    if best_only:
        return OutputOptions(
            save_candidate_artifacts=False,
            save_all_results_csv=False,
            best_bitstreams_dir=best_bitstreams_dir or output_dir,
        )

    return OutputOptions(
        save_candidate_artifacts=save_candidates,
        save_all_results_csv=save_all_results_csv,
        best_bitstreams_dir=best_bitstreams_dir,
    )
