from __future__ import annotations

import json
from pathlib import Path


def stage1_evidence_fixture_root() -> Path:
    root = Path(__file__).resolve().parent / "fixtures" / "stage1_evidence"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "Stage 1 evidence fixture is missing: "
            f"expected repository fixture manifest at {manifest}"
        )
    return root


def load_selected_split_fixture_manifest() -> dict[str, object]:
    root = stage1_evidence_fixture_root()
    repository_root = Path(__file__).resolve().parents[1]
    config_path = repository_root / "experiments" / "configs" / "stage1_best_training_selected_split_benchmark.json"
    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    manifest["methods_source"] = str(root / "stage1_closed_loop_baseline_formal" / "batch_manifest.json")
    for scenario in manifest["scenarios"]:
        item = dict(scenario)
        item["snapshot"] = str(root / "stage1_best_training_candidate_discovery_rewired_seed1_20260727_212932" / "scenario_snapshots" / Path(str(item["snapshot"])).name)
        scenario.clear()
        scenario.update(item)
    return manifest
