from __future__ import annotations

import pytest

from experiments.stage1_best_training_candidate_discovery_cli import (
    CandidateDiscoveryError,
    ensure_new_output_dir,
)


def test_candidate_discovery_refuses_existing_output_dir() -> None:
    class ExistingPath:
        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return "existing"

    existing = ExistingPath()
    with pytest.raises(CandidateDiscoveryError, match="refusing_existing_output_directory"):
        ensure_new_output_dir(existing)
