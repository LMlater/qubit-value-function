from experiments.stage2_generalization_full_benchmark_cli import (
    decision_cutoff_for_model,
    is_regression_model,
)
from qubit_value_function.stage2_models import ConstantBaselineModel, ThresholdConditionedQNNModel


def test_benchmark_metadata_uses_null_cutoff_for_non_threshold_models() -> None:
    assert decision_cutoff_for_model(ConstantBaselineModel()) is None
    assert decision_cutoff_for_model(ThresholdConditionedQNNModel()) is None


def test_threshold_qnn_probabilities_are_not_placed_in_regression_or_ranking_tables() -> None:
    assert not is_regression_model("threshold_conditioned_qnn")
    assert is_regression_model("full_expectation_qnn")
