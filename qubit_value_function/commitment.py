from __future__ import annotations

import itertools

import numpy as np

from .uc_loader import UCInstance


def all_commitments(num_generators: int, horizon: int) -> np.ndarray:
    """Return all binary commitments with shape (2^(G*T), G, T)."""
    num_bits = num_generators * horizon
    rows = itertools.product((0, 1), repeat=num_bits)
    return np.array(list(rows), dtype=int).reshape((-1, num_generators, horizon))


def bits_to_commitment(bits: list[int] | np.ndarray, num_generators: int, horizon: int) -> np.ndarray:
    bits_array = np.asarray(bits, dtype=int)
    if bits_array.size != num_generators * horizon:
        raise ValueError("bit vector has the wrong length")
    return bits_array.reshape((num_generators, horizon))


def commitment_to_bitstring(commitment: np.ndarray) -> str:
    return "".join(str(int(v)) for v in commitment.reshape(-1))


def is_logic_feasible(instance: UCInstance, commitment: np.ndarray) -> bool:
    """Check Boolean UC constraints that can be encoded directly in an oracle."""
    u = np.asarray(commitment, dtype=int)
    if u.shape != (len(instance.generators), instance.time_horizon):
        raise ValueError("commitment shape does not match the instance")

    for g_idx, gen in enumerate(instance.generators):
        status = u[g_idx]
        if gen.must_run and np.any(status != 1):
            return False

        if gen.initial_status < 0:
            remaining_down = max(gen.min_downtime + gen.initial_status, 0)
            if remaining_down and np.any(status[:remaining_down] != 0):
                return False
        elif gen.initial_status > 0:
            remaining_up = max(gen.min_uptime - gen.initial_status, 0)
            if remaining_up and np.any(status[:remaining_up] != 1):
                return False

        previous = 1 if gen.initial_status > 0 else 0
        for t, current in enumerate(status):
            if current == 1 and previous == 0:
                end = min(instance.time_horizon, t + gen.min_uptime)
                if np.any(status[t:end] != 1):
                    return False
            if current == 0 and previous == 1:
                end = min(instance.time_horizon, t + gen.min_downtime)
                if np.any(status[t:end] != 0):
                    return False
            previous = int(current)

    return True
