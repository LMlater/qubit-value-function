from __future__ import annotations

import inspect

from qubit_value_function.logic_feasibility_oracle import (
    build_joint_feasible_better_phase_oracle,
    build_logic_feasibility_phase_oracle,
    compile_logic_feasibility_spec,
)
from qubit_value_function.sparse_vqc_grover import build_sparse_vqc_grover_circuit


def test_joint_oracle_and_grover_builders_do_not_enumerate_search_states() -> None:
    for function in (
        compile_logic_feasibility_spec,
        build_logic_feasibility_phase_oracle,
        build_joint_feasible_better_phase_oracle,
        build_sparse_vqc_grover_circuit,
    ):
        source = inspect.getsource(function)
        assert "range(2**" not in source
        assert "marked_mask" not in source
        assert "values_for_all_x" not in source
