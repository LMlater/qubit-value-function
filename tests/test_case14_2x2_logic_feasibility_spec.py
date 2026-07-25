from __future__ import annotations

from pathlib import Path

import numpy as np

from qubit_value_function.commitment import is_logic_feasible
from qubit_value_function.experiment_utils import (
    embedded_selected_commitments,
    time_window_instance,
)
from qubit_value_function.logic_feasibility_oracle import compile_logic_feasibility_spec
from qubit_value_function.uc_loader import load_uc_instance


def test_case14_three_windows_compiled_logic_matches_classical_selected_subspace() -> None:
    source = load_uc_instance(Path("data/case14.json.gz"))
    selected = (0, 5)
    for window_start in (0, 1, 2):
        instance = time_window_instance(source, start=window_start, horizon=2)
        base = np.ones((len(instance.generators), 2), dtype=int)
        commitments = embedded_selected_commitments(base, selected)
        spec = compile_logic_feasibility_spec(
            instance,
            selected_generator_indices=selected,
            base_commitment=base,
        )
        assert spec.always_infeasible is False
        for index, commitment in enumerate(commitments):
            bits = tuple((int(index) >> bit) & 1 for bit in range(4))
            assert spec.is_feasible(bits) == is_logic_feasible(instance, commitment)
