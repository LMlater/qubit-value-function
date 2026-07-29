from pathlib import Path

import numpy as np

from qubit_value_function.hard_logic_independent_audit import independent_logic_check
from qubit_value_function.uc_loader import Generator, UCInstance


def _instance(initial_status: int = 0) -> UCInstance:
    gen = Generator("g", "b", [0.0, 1.0], [0.0, 1.0], [1], [0.0], initial_status, 0.0, 2, 2, None, None, False, ())
    return UCInstance(Path("synthetic"), "test", 2, [gen], [0.0, 0.0], [], [1.0, 1.0])


def test_independent_checker_reports_post_start_min_up_violation():
    result = independent_logic_check(_instance(), np.array([[1, 0]], dtype=int))
    assert result.feasible is False
    assert {row["rule_name"] for row in result.violations} == {"post_start_min_up"}


def test_independent_checker_reports_initial_residual_min_down():
    result = independent_logic_check(_instance(initial_status=-1), np.array([[1, 0]], dtype=int))
    assert result.feasible is False
    assert any(row["rule_name"] == "initial_residual_min_down" for row in result.violations)
