import math

from qubit_value_function.stage1_evidence import (
    classify_unreachable,
    confusion_metrics,
    grover_probability,
    index_bitstring,
    subspace_optimum,
)


def test_index_bitstring_uses_little_endian_search_order():
    assert index_bitstring(3, 4) == "1100"
    assert index_bitstring(8, 4) == "0001"


def test_confusion_metrics_uses_strict_truth_and_zero_denominators():
    metrics = confusion_metrics([True, True, False, False], [True, False, True, False])
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 1, 1)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert confusion_metrics([], [])["precision"] is None


def test_unreachable_classification_records_cost_false_negative_and_logic_filter():
    labels = classify_unreachable(
        true_improvement=[True, False], cost_marked=[False, True], joint_marked=[False, False], logic_feasible=[True, False]
    )
    assert "B_true_improvement_all_cost_false_negative" in labels
    assert "C_cost_marked_but_no_true_improvement" in labels


def test_subspace_optimum_allows_ties_and_reports_gap():
    result = subspace_optimum([10.0, 8.0, 8.0], discovered_index=0)
    assert result["optimum_indices"] == [1, 2]
    assert result["discovered_is_optimum"] is False
    assert result["absolute_gap"] == 2.0


def test_m1_theory_is_computed_from_formula():
    assert math.isclose(grover_probability(1, 0, 16), 0.0625)
    assert math.isclose(grover_probability(1, 2, 16), 0.908447265625)


def test_uniform_oracle_query_probabilities_do_not_assume_known_marked_indices():
    from qubit_value_function.stage1_oracle_query_diagnostic import uniform_hit_probability

    assert uniform_hit_probability(1, 16, 1, replacement=True) == 1 / 16
    assert uniform_hit_probability(1, 16, 16, replacement=False) == 1.0
