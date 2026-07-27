from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .candidate_acceptance_loop import ClosedLoopBudgets, ExactCandidateEvaluation, ExactEvaluator, HardLogicEvaluator
from .closed_loop_random_baselines import (
    run_classical_joint_marked_random,
    run_direct_logic_feasible_random,
    run_full_space_random,
    run_logic_rejection_random,
)
from .coherent_phase_value import QuantizedSparseValueModel
from .closed_loop_metadata import build_quantized_model_snapshot
from .logic_feasibility_oracle import LogicFeasibilitySpec
from .sparse_vqc_bbht import (
    BBHTConfig,
    BBHTTrialExecution,
    SparseVQCBBHTResult,
    run_sparse_vqc_bbht,
)
from .candidate_acceptance_loop import COST_ONLY_BBHT_ADMISSION_POLICY


@dataclass(frozen=True)
class ClosedLoopScenario:
    """Frozen pre-search artifacts shared by all closed-loop comparison methods."""

    scenario_id: str
    generator_pair: tuple[int, int]
    window_start: int
    horizon: int
    training_seed: int
    training_indices: tuple[int, ...]
    training_labels: tuple[tuple[int, float], ...]
    value_model: QuantizedSparseValueModel
    initial_incumbent_index: int
    initial_exact_cache: Mapping[int, ExactCandidateEvaluation]
    evaluate_candidate: ExactEvaluator
    hard_logic_is_feasible: HardLogicEvaluator
    commitments: Sequence[object]
    budgets: ClosedLoopBudgets
    bbht_config: BBHTConfig
    reproducibility_metadata: Mapping[str, object]
    feasibility_spec: LogicFeasibilitySpec | None = None

    def __post_init__(self) -> None:
        if not str(self.scenario_id).strip():
            raise ValueError("scenario_id 不能为空")
        if not self.training_indices:
            raise ValueError("training_indices 不能为空")
        if int(self.initial_incumbent_index) not in self.initial_exact_cache:
            raise ValueError("初始 incumbent 必须存在于 scenario initial_exact_cache")
        object.__setattr__(
            self,
            "initial_exact_cache",
            MappingProxyType({int(index): record for index, record in self.initial_exact_cache.items()}),
        )
        object.__setattr__(
            self,
            "training_indices",
            tuple(int(index) for index in self.training_indices),
        )
        object.__setattr__(
            self,
            "training_labels",
            tuple((int(index), float(cost)) for index, cost in self.training_labels),
        )
        object.__setattr__(self, "commitments", tuple(self.commitments))
        object.__setattr__(
            self, "reproducibility_metadata", MappingProxyType(dict(self.reproducibility_metadata))
        )

    def metadata(self) -> dict[str, object]:
        metadata = {
            "scenario_id": self.scenario_id,
            "generator_pair": list(self.generator_pair),
            "window_start": int(self.window_start),
            "horizon": int(self.horizon),
            "training_seed": int(self.training_seed),
            "training_indices": [int(index) for index in self.training_indices],
            "initial_incumbent_index": int(self.initial_incumbent_index),
            "fixed_point": {
                "fractional_bits": int(self.value_model.fixed_point_config.fractional_bits),
                "unit": float(self.value_model.fixed_point_config.unit),
            },
            "reproducibility_metadata": dict(self.reproducibility_metadata),
        }
        for key in (
            "initial_incumbent_policy",
            "initial_incumbent_bitstring",
            "initial_incumbent_true_cost",
            "best_training_candidate_indices",
            "best_training_tie_count",
            "best_training_tie_break_rule",
            "training_selection_protocol",
            "global_truth_used_online",
            "global_truth_used_for_posthoc_validation",
        ):
            if key in self.reproducibility_metadata:
                metadata[key] = self.reproducibility_metadata[key]
        return metadata


def run_closed_loop_method(
    scenario: ClosedLoopScenario,
    method: str,
    run_seed: int,
    *,
    trial_executor: Callable[..., BBHTTrialExecution] | None = None,
    persist_dynamic_oracle_metadata: bool = False,
) -> dict[str, object]:
    """Run one isolated method from a frozen scenario; no validation input exists."""

    common = {
        "initial_incumbent_index": scenario.initial_incumbent_index,
        "initial_exact_cache": dict(scenario.initial_exact_cache),
        "training_indices": scenario.training_indices,
        "evaluate_candidate": scenario.evaluate_candidate,
    }
    if method == "joint_bbht":
        result = run_sparse_vqc_bbht(
            scenario.value_model,
            config=_bbht_config_for_seed(scenario.bbht_config, run_seed),
            feasibility_spec=scenario.feasibility_spec,
            trial_executor=trial_executor,
            method="joint_bbht",
            persist_dynamic_oracle_metadata=persist_dynamic_oracle_metadata,
            hard_logic_metadata=scenario.hard_logic_is_feasible,
            initial_incumbent_policy=str(
                scenario.reproducibility_metadata.get(
                    "initial_incumbent_policy", "first_training"
                )
            ),
            **common,
        )
        return _method_envelope(scenario, method, run_seed, result, "quantum_method", persist_dynamic_oracle_metadata)
    if method == "cost_only_bbht":
        result = run_sparse_vqc_bbht(
            scenario.value_model,
            config=_bbht_config_for_seed(scenario.bbht_config, run_seed),
            feasibility_spec=None,
            trial_executor=trial_executor,
            method="cost_only_bbht",
            admission_policy=COST_ONLY_BBHT_ADMISSION_POLICY,
            persist_dynamic_oracle_metadata=persist_dynamic_oracle_metadata,
            hard_logic_metadata=scenario.hard_logic_is_feasible,
            initial_incumbent_policy=str(
                scenario.reproducibility_metadata.get(
                    "initial_incumbent_policy", "first_training"
                )
            ),
            **common,
        )
        return _method_envelope(scenario, method, run_seed, result, "quantum_method", persist_dynamic_oracle_metadata)
    random_common = {
        "initial_incumbent_index": scenario.initial_incumbent_index,
        "initial_exact_cache": dict(scenario.initial_exact_cache),
        "training_indices": scenario.training_indices,
        "evaluate_candidate": scenario.evaluate_candidate,
        "budgets": scenario.budgets,
        "seed": int(run_seed),
        "hard_logic_is_feasible": scenario.hard_logic_is_feasible,
    }
    if method == "full_space_random":
        result = run_full_space_random(scenario.value_model, **random_common)
        return _method_envelope(scenario, method, run_seed, result, "random_baseline", persist_dynamic_oracle_metadata)
    if method == "logic_rejection_random":
        result = run_logic_rejection_random(scenario.value_model, **random_common)
        return _method_envelope(scenario, method, run_seed, result, "random_baseline", persist_dynamic_oracle_metadata)
    if method == "direct_logic_feasible_random":
        result = run_direct_logic_feasible_random(scenario.value_model, **random_common)
        return _method_envelope(scenario, method, run_seed, result, "diagnostic_only", persist_dynamic_oracle_metadata)
    if method == "classical_joint_marked_random":
        result = run_classical_joint_marked_random(scenario.value_model, **random_common)
        return _method_envelope(scenario, method, run_seed, result, "diagnostic_only", persist_dynamic_oracle_metadata)
    raise ValueError(f"不支持的 closed-loop method: {method}")


def _bbht_config_for_seed(config: BBHTConfig, seed: int) -> BBHTConfig:
    return BBHTConfig(
        lambda_factor=config.lambda_factor,
        max_trials=config.max_trials,
        max_oracle_calls=config.max_oracle_calls,
        max_new_ed_lp_calls=config.max_new_ed_lp_calls,
        max_threshold_updates=config.max_threshold_updates,
        max_consecutive_nonimproving_marked=config.max_consecutive_nonimproving_marked,
        max_same_encoded_threshold_updates=config.max_same_encoded_threshold_updates,
        max_auxiliary_syndrome_rejections=config.max_auxiliary_syndrome_rejections,
        shots_per_trial=1,
        minimum_auxiliary_zero_probability=config.minimum_auxiliary_zero_probability,
        seed=int(seed),
    )


def _method_envelope(
    scenario: ClosedLoopScenario,
    method: str,
    run_seed: int,
    result: SparseVQCBBHTResult | object,
    method_role: str,
    persist_dynamic_oracle_metadata: bool,
) -> dict[str, object]:
    diagnostic_only = bool(getattr(result, "diagnostic_only", False))
    envelope = {
        "method": method,
        "method_role": method_role,
        "run_seed": int(run_seed),
        "scenario": scenario.metadata(),
        "diagnostic_only": diagnostic_only,
        "result": result,
        "result_schema": result.as_dict(),
    }
    if persist_dynamic_oracle_metadata:
        envelope["result_schema_version"] = "stage1_closed_loop_dynamic_oracle_v1"
        envelope["quantized_model_snapshot"] = build_quantized_model_snapshot(
            scenario_id=scenario.scenario_id,
            generator_pair=scenario.generator_pair,
            window_start=scenario.window_start,
            training_seed=scenario.training_seed,
            model=scenario.value_model,
            hard_logic_is_feasible=scenario.hard_logic_is_feasible,
            training_indices=scenario.training_indices,
            training_labels=scenario.training_labels,
            initial_incumbent_index=scenario.initial_incumbent_index,
            initial_incumbent_true_cost=float(
                scenario.initial_exact_cache[scenario.initial_incumbent_index].total_cost
            ),
            initial_cache_indices=tuple(scenario.initial_exact_cache),
        )
    return envelope
