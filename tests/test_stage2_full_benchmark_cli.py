from experiments.stage2_generalization_full_benchmark_cli import decision_cutoff_for_model
from qubit_value_function.stage2_models import ConstantBaselineModel, ThresholdConditionedQNNModel


def test_benchmark_metadata_uses_null_cutoff_for_non_threshold_models() -> None:
    assert decision_cutoff_for_model(ConstantBaselineModel()) is None
    assert decision_cutoff_for_model(ThresholdConditionedQNNModel()) is None
