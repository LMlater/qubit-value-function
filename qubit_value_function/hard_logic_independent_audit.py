"""Independent, explanatory reference for the implemented Boolean UC rules.

It intentionally does not import commitment.is_logic_feasible or the compiled
forbidden-pattern/oracle machinery.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .uc_loader import UCInstance

@dataclass(frozen=True)
class IndependentLogicResult:
    feasible: bool
    violations: list[dict[str, Any]]

def _row(rule: str, gen, t: int, required: int, observed: list[int], explanation: str) -> dict[str, Any]:
    return {"rule_name":rule,"generator":gen.name,"time_index":int(t),"initial_status":int(gen.initial_status),"initial_duration":abs(int(gen.initial_status)),"required_duration":int(required),"observed_pattern":"".join(map(str,observed)),"explanation":explanation}

def independent_logic_check(instance: UCInstance, commitment: np.ndarray) -> IndependentLogicResult:
    """Directly check must-run and initial/post transition minimum durations."""
    u=np.asarray(commitment,dtype=int); horizon=int(instance.time_horizon)
    if u.shape!=(len(instance.generators),horizon) or np.any((u!=0)&(u!=1)): raise ValueError("invalid commitment")
    violations=[]
    for gi,gen in enumerate(instance.generators):
        status=[int(x) for x in u[gi]]
        if gen.must_run:
            for t,x in enumerate(status):
                if x!=1: violations.append(_row("must_run",gen,t,1,status,"must-run generator is off"))
        if gen.initial_status<0:
            rem=max(int(gen.min_downtime)+int(gen.initial_status),0)
            for t in range(min(rem,horizon)):
                if status[t]!=0: violations.append(_row("initial_residual_min_down",gen,t,rem,status,"initial downtime obligation requires off"))
        elif gen.initial_status>0:
            rem=max(int(gen.min_uptime)-int(gen.initial_status),0)
            for t in range(min(rem,horizon)):
                if status[t]!=1: violations.append(_row("initial_residual_min_up",gen,t,rem,status,"initial uptime obligation requires on"))
        previous=1 if gen.initial_status>0 else 0
        for t,current in enumerate(status):
            if current==1 and previous==0:
                end=min(horizon,t+int(gen.min_uptime))
                if any(x!=1 for x in status[t:end]): violations.append(_row("post_start_min_up",gen,t,int(gen.min_uptime),status[t:end],"startup is not held for minimum uptime"))
            if current==0 and previous==1:
                end=min(horizon,t+int(gen.min_downtime))
                if any(x!=0 for x in status[t:end]): violations.append(_row("post_shutdown_min_down",gen,t,int(gen.min_downtime),status[t:end],"shutdown is not held for minimum downtime"))
            previous=current
    return IndependentLogicResult(not violations,violations)
