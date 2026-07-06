# Work Log

## 2026-05-19

- Built a fixed-load UC value-function prototype using real UnitCommitment.jl data.
- Parsed `aelmp_simple.json.gz` and solved fixed-commitment ED LPs to compute exact \(V_d(x)\).
- Verified the measurement-readout VQC only as a value-function learnability baseline, not as a reversible oracle.
- Implemented a threshold-conditioned diagonal phase VQC oracle:
  \(U_\theta(\tau)|x\rangle=e^{i\phi_\theta(x,\tau)}|x\rangle\).
- Verified on `aelmp_simple`: correct threshold marking, unitary oracle, near self-inverse behavior, and Grover amplification.
- Added threshold generalization testing. Result: training-threshold marking is exact on `aelmp_simple`, but unseen-threshold performance is mixed, so naive polynomial tau-conditioning is not yet reliable.
- Added `case14` single-period experiment with 6 commitment bits and 64 enumerated states.
- Compared phase interaction orders on `case14`. Fixed-threshold results: order 1-2 miss the optimum, order 3-4 mark the optimum but have large phase errors, order 5 gives useful Grover amplification, and full order 6 matches the exact oracle.
- Key implication: low-order diagonal phase VQC is not yet sufficient for robust SCUC oracle construction; higher-order interactions or a better threshold-conditioning design are needed.
- Implemented greedy sparse monomial phase selection for the `case14` fixed-threshold oracle. Result: 5 selected terms exactly reproduce the target marking and Grover probability, much fewer than 64 full terms, but the selected terms are high-order.
- Tested a physics-feature phase model using capacity, startup, reserve, and cost-margin features. Result: it missed the optimum at the strict best-solution threshold, so simple aggregate physical features are not sufficient.
- Current implication: sparse high-order phase terms can compress the oracle for this small case, but a scalable ansatz likely needs structured high-order features or better problem-specific threshold design.
- Tested Hamming-distance phase features around the best commitment. Result: degree-6 Hamming features exactly recover the single-best-state oracle, but they fail for broader near-optimal sets, so distance-to-best is too crude.
- Built a fixed-threshold sparse phase oracle library for several target-set sizes on `case14`. Results: first successful sparse oracles require 5 terms for top-1, 9 terms for top-2, 13 terms for top-6, and 20 terms for top-10; top-21 and top-32 are not reliable within 32 terms.
- Refined Grover success evaluation by comparing with the exact oracle probability; for 10 targets the exact one-iteration Grover probability is 0.881, so matching that value is considered successful.
- Analyzed selected sparse phase terms physically. The useful high-order monomials correspond to generator bundles with enough capacity margin, but they often need inclusion-exclusion terms to exclude extra committed units.
- Tested frequent open-bundle phase features. Result: open-only bundles cannot reliably mark target sets because they cannot distinguish states with additional online units.
- Tested signed pattern bundle features with both ON and OFF literals. Result: they exactly reproduce top-1/top-2/top-6/top-10 oracles and Grover probabilities, but they are close to a commitment-pattern library and should be treated as an upper-bound/control method.
- Tested partial signed-pattern templates based on key generator statistics. Result: all target-set sizes required controlling all 6 generators to match the exact oracle; controlling only the most discriminative generators was insufficient.
- Tested frequency-template signed features. Result: simple high/low on-frequency templates did not provide a compact reliable oracle; they either failed marking or had large phase/self-inverse errors.
- Current implication: for `case14` single-period strict cost sublevel sets, exact phase marking is highly sensitive to full commitment patterns. Future work should use hierarchical/coarse-to-fine search or learn smoother surrogate target sets instead of expecting a few key-unit templates to exactly reproduce the oracle.
- Ran a hierarchical coarse-oracle diagnostic using a simplified merit-order surrogate cost. The surrogate ranks the true optimum second and covers the exact top-5 with 12 candidates and top-10 with 24 candidates.
- The coarse candidate set can reduce the classical search region, but a simple physical-feature phase model still cannot cleanly mark the candidate set as a reliable Grover oracle.
- Current implication: hierarchical coarse screening is useful as a diagnostic/classical aid, but it should not replace the original research-content-1 goal of building a VQC implicit value-function oracle.
- Returned to the main VQC oracle line and implemented an ancilla-based reversible oracle \(O_\theta=U_\theta^\dagger Z_a U_\theta\).
- Verified that this oracle is unitary and self-inverse by construction, but it is Grover-useful only when the ancilla rotation angles are near 0 or \(\pi\), otherwise amplitude leaks into the ancilla-1 subspace.
- Results: on `aelmp_simple`, full order-3 features reproduce the exact Grover probability with negligible leakage; on `case14` period 0, order-5 features give target probability 0.956 with max leakage 0.0024, while full order-6 matches the exact oracle.
- Extended the main-line ancilla VQC oracle to `case14` with 6 generators and 2 time periods. This gives 12 commitment bits and 4096 states; 768 states are logic-feasible under the real UnitCommitment.jl data.
- Added state-vector Grover simulation for \(O_\theta=U_\theta^\dagger Z_a U_\theta\), avoiding construction of the full `8192 x 8192` oracle matrix.
- `case14` T=2 result: optimum commitment is `110011111100` with cost 20578.2152604, runner-up is `110111111100` with cost 24578.2152604.
- Low-order T=2 result: orders 1-4 do not reliably mark top target sets. Order 6 marks top-1/top-16/top-64 correctly, but leakage remains large, so it is not yet a clean Grover oracle.
- Full 12-order Boolean monomial interpolation exactly represents the T=2 oracle with zero leakage, but it uses 4096 features and is recorded only as an exponential upper-bound reference, not a scalable solution.
- Improved T=2 commitment reporting: results now include both generator-major/time-major bitstrings and readable generator-by-time tables, plus online-generator lists for each time period.
- Re-aligned the T=2 oracle with research content 1 by separating exact 0-1 feasibility constraints from the VQC value-function oracle.
- Implemented a controlled ancilla value oracle: infeasible commitments use identity blocks, while feasible commitments use \(U_\theta^\dagger Z_a U_\theta\).
- Separated T=2 result: order 6 correctly marks top-1/top-16/top-64; target probabilities improve to 0.782/0.932/0.954, with max leakage 0.111/0.135/0.222 respectively. This is better than the unseparated oracle but still not a clean low-leakage oracle.
- Implemented an explicit two-ancilla oracle simulation with registers \(|x\rangle|f\rangle|a\rangle\): \(A_f \rightarrow U_\theta \rightarrow CCZ(f,a) \rightarrow U_\theta^\dagger \rightarrow A_f^\dagger\).
- Verified that the explicit two-ancilla oracle matches the previous controlled-ancilla equivalent model to numerical precision, while making the feasibility ancilla and value ancilla marking/uncomputation process explicit.
- Explicit T=2 result: feasibility ancilla is exactly uncomputed (`one_feasibility_probability = 0`); order 6 gives target probabilities 0.782/0.932/0.954 for top-1/top-16/top-64, matching the separated-oracle result.
- Implemented leakage-reweighted training for the value-function ancilla. The method iteratively increases weights on states with large \(\sin^2\theta\) leakage while keeping exact feasibility constraints separate.
- Tested direct periodic phase/branch-lifting ideas; in this `case14` T=2 setting they did not reliably improve the order-6 leakage, so the recorded implementation uses leakage-reweighted least squares instead.
- Leakage-reweighted T=2 result: for order 6, max leakage decreases from 0.111/0.135/0.222 to 0.018/0.043/0.095 for top-1/top-16/top-64. Grover target probabilities change from 0.782/0.932/0.954 to 0.622/0.918/0.943, showing a leakage-vs-amplification tradeoff rather than a uniform improvement.
- Implemented joint-score selection over ordinary and leakage-reweighted VQC candidates, scoring Grover target probability, max/mean value-ancilla leakage, and marked-set errors together.
- Joint-score T=2 result at order 6: top-1 selects an intermediate candidate with max leakage 0.038 and target probability 0.759, avoiding the leakage-only drop to 0.622; top-16 gives leakage 0.064 and target probability 0.928; top-64 gives leakage 0.109 and target probability 0.950.
- Implemented threshold-conditioned ancilla VQC \(U_\theta(x,\tau)\), so one model represents multiple value-function sublevel comparisons instead of training one oracle per threshold.
- Default T=2 threshold-conditioned result: with training thresholds top-1/top-16/top-64, order 6 correctly marks all training thresholds with max leakage 0.222, but holdout thresholds top-4/top-32 fail with false positives. This shows naive polynomial \(\tau\)-conditioning does not generalize reliably.
- Dense-\(\tau\) result: training on top-1/top-4/top-16/top-32/top-64 makes order 6 correct on all those training thresholds; holdout top-8/top-48 still misses 2 and 14 target states respectively, so more threshold samples improve but do not solve interpolation.
- Added value-function coherence diagnostics. Order 6 has zero monotonicity violations over the evaluated thresholds, so the remaining failure is mainly threshold-boundary calibration/leakage, not a simple violation of sublevel-set monotonicity.
- Tested rank-distance boundary weighting for threshold-conditioned ancilla VQC. For order 6 it reduces top-1 max leakage from 0.111 to 0.050 in the default threshold set, but holdout top-4/top-32 still fail; in dense-\(\tau\) training, holdout top-48 becomes worse (19 misses vs. 14), so boundary weighting is recorded as a negative/diagnostic result.
- Added a piecewise-linear \(\tau\) basis as an alternative to global polynomial threshold features. It preserves exact trained-threshold marking at order 6 and avoids monotonicity violations, but does not solve held-out thresholds: dense-\(\tau\) holdout top-8 still misses 2 states and top-48 has 14 false positives plus 6 false negatives.
- Current implication: fixed-angle ancilla VQC \(U_\theta(x,\tau)\) can represent trained threshold sublevel sets, but interpolation between thresholds is still unreliable. The next structural direction should be a calibrated value-register/comparator oracle or a monotonic constrained model, not more ad hoc weighting.
- Implemented a scalar value-function surrogate \(\hat V_\theta(x,u)\) followed by a fixed-point value-register comparator. The simulated reversible structure is: exact feasibility oracle, compute value register, compare with threshold register, phase mark, then uncompute.
- Value-register T=2 result: order 6 fits feasible costs with MAE 2.91 and max error 8.99, giving only 26 pairwise rank inversions among 768 feasible states. The floating comparator exactly marks top-1/top-16/top-48 and has only small boundary errors at top-4/top-8/top-32/top-64.
- Fixed-point bit-width result: 8/10 bits are too coarse; 12 and 16 bits reduce total order-6 threshold errors to 8 across top-1/top-4/top-8/top-16/top-32/top-48/top-64. Quantization still introduces false positives around top-48, so register precision and calibration remain design variables.
- Current implication: the value-register/comparator route is closer to a reversible Grover oracle and gives stronger threshold generalization than direct threshold-angle fitting, but it now shifts the problem to accurate value approximation, fixed-point precision, and reversible arithmetic cost.
- Found that several strict threshold errors were dominated by numerical cost ties at the boundary; some adjacent feasible costs differ by only about \(3.64\times 10^{-12}\), so splitting them is not physically meaningful.
- Added tie-tolerant target-set construction with tolerance `1e-6` for value-register/comparator tests.
- Tie-tolerant high-order T=2 result: order 12 fits the feasible value function with MAE \(6.69\times 10^{-9}\) and max error \(9.42\times 10^{-8}\). The floating comparator exactly marks all evaluated target sets.
- With order 12 and a 16/20-bit fixed-point value register, all evaluated threshold target sets are exactly marked: actual target counts are 1/7/9/16/34/55/65 for requested top-1/top-4/top-8/top-16/top-32/top-48/top-64.
- The corresponding Grover target probabilities are about 0.9999/0.9983/0.9995/0.9999/0.9996/0.9963/0.9949, matching the exact phase-oracle probabilities in the state-vector simulation.
- Caveat: order 12 uses 4096 Boolean features for 12 commitment bits, so this is an exponential upper-bound reference. The next research step is to approach this behavior with lower-order or structured VQC/value-arithmetic features.
- Implemented structured value-function features for the value-register route: commitment bits, physical capacity/reserve aggregates, startup/transition terms, same-time unit interaction terms, adjacent-time interaction terms, and deterministic merit-order dispatch proxy terms.
- Structured feature result: pure physical/order-1 features are poor; adding the merit-order proxy cuts MAE to about 312, and same-time interaction order 4 cuts MAE to about 16.83 using 207 features.
- The 207-feature same-time-order-4 structured model has rank-separable target sets for all evaluated thresholds after threshold-register calibration; with a 20-bit value register it exactly marks all target sets 1/7/9/16/34/55/65.
- A 257-feature model with same-time order 6 plus adjacent-time pair terms marks 6 of 7 target sets using the raw threshold, and all 7 after calibrated threshold loading with a 20-bit value register.
- Calibration means loading a learned predicted-value threshold \(\hat\tau(\tau)\) into the comparator. It does not add a second oracle; the reversible structure remains compute value register, compare, phase mark, and uncompute.
- Current implication: a non-table structured value-register oracle can approach the order-12 upper-bound behavior with far fewer features, but register precision still matters because 16-bit quantization fails at small calibration margins near top-16/top-32.
- Extended the structured value-register experiment to `case14` T=3. The search space has 18 commitment bits and 262144 states; exact ED evaluation gives 16384 finite logic-feasible commitments.
- Added exact value caching for structured experiments. The T=3 ED values are cached at `results/value_cache_case14.json_h3.npz`, avoiding repeated LP solves in later feature sweeps.
- T=3 optimum: generator-major bitstring `111000111111111000`, time-major bitstring `101110101110101110`, total cost 27985.8598204.
- T=3 structured result: pure physical features remain poor; merit-order proxy plus local interactions transfers well to the larger time horizon.
- Best T=3 local structured model tested: 761 features with same-time order 6 and adjacent-time order 3. It achieves MAE about 16.64 and max error about 116.00 over 16384 feasible states.
- With calibrated threshold loading and a 20-bit value register, the T=3 model exactly marks top-1/top-4/top-8/top-16/top-32/top-64 target sets. It does not exactly mark top-128: the boundary is not rank-separable yet, with calibration margin about -6.94.
- Current implication: the structured value-register oracle scales from 2 to 3 time periods for near-optimal target sets, but broader target sets require better boundary-ranking features or an objective focused directly on threshold separation.

## 2026-05-21

- Implemented a max-affine value-function module `qubit_value_function/max_affine.py` with the form \(\hat V(x) = \max_r (b_r + \theta_r^\top f(x))\), plus diagnostics for selected anchor states, lower-bound violations, and reversible gate-count proxies.
- Added `experiments/stage1_case14_t2_max_affine_value_surrogate.py` to evaluate a Grover-ready value-register route: compute structured features, compute all affine piece registers, reversibly take the maximum, compare with a threshold register, phase mark, and uncompute.
- Added tests for the max-affine construction in `tests/test_stage1.py`, including a synthetic convex-function recovery test and a gate-count sanity test.
- T=2 max-affine result on `case14` with 207 structured features (`same_time_order=4`): the strict supporting-cut `floor` initialization gives zero lower-bound violations and, with 64 pieces, exactly marks all tested target sets after calibrated 16/20-bit comparison, though MAE remains about 287.61.
- T=2 practical max-affine result: the `least_squares` initialization with 32 pieces reaches MAE about 12.97, max error about 137.96, and exactly marks all tested target sets after calibrated 16/20-bit comparison. This is the first piecewise-affine surrogate in the project that matches the whole tested T=2 target-set family without using the 4096-term Boolean interpolation upper bound.
- T=3 max-affine scaling result on `case14` with 380 structured features (`same_time_order=4`, `adjacent_time_order=2`): the 32-piece `least_squares` model reaches MAE about 21.70 and max error about 139.38.
- With calibrated 20-bit comparison, the T=3 max-affine model exactly marks top-1/top-4/top-8/top-16/top-32/top-64, but not top-128. At top-128 the calibration margin is about -31.84, with 6 false positives and 1 false negative.
- Current implication: the project now has a reversible oracle-compatible value-register route whose surrogate form matches the theoretical pointwise-maximum interpretation of the ED value function more closely than a single affine fit. The remaining bottleneck is no longer reversibility, but rank separation for broader target sets as the time horizon grows.
- Added boundary-aware max-affine training on top of the same reversible value-register oracle. The implementation keeps the structured 380-feature, 32-piece max-affine form, but adds weighted least-squares initialization around chosen threshold boundaries and iterative reweighting of states that remain misordered after calibrated comparison.
- Boundary-aware T=3 result on `case14`: with focus on top-128, rank window 32, boundary weight 16, non-target-side weight 3, and 4 reweighting rounds, the calibrated top-128 margin improves from about -31.84 to about +1.85.
- The reweighting history is monotone in the successful run: top-128 calibration margin goes roughly -26.28 -> -18.01 -> -2.93 -> +1.85, and the calibrated false-positive/false-negative counts go 6/1 -> 6/1 -> 2/1 -> 0/0.
- Full T=3 boundary-aware validation result: with a 20-bit value register, the same model exactly marks top-1/top-4/top-8/top-16/top-32/top-64/top-128. This closes the previous T=3 top-128 gap without changing the oracle decomposition or adding lookup-table features.
- Quantization caveat remains: the improved T=3 boundary-aware model is still not exact for top-128 with a 16-bit register, because the final positive separation margin is small enough that fixed-point rounding merges a few extra states back across the threshold.

## 2026-05-22

- Created `RESEARCH_CONTENT_1_SUMMARY.md` as a concise research-content-1 report for advisor discussion.
- The summary records the fixed-load value-function definition, the reversible value-register/comparator oracle, the max-affine VQC value-function representation, threshold-register marking, resource estimates, and the main T=2/T=3 experimental results.
- Restored `results/stage1_case14_t2_max_affine_value_surrogate.json` after noticing it had been overwritten by a temporary T=3 run, so the filename again matches the contained T=2 experiment.
- Current communication-ready conclusion: the project has a Grover-compatible value-function oracle route validated on `case14` with 6 units and up to 3 time periods; the best T=3 boundary-aware max-affine result exactly marks top-1/top-4/top-8/top-16/top-32/top-64/top-128 with a 20-bit value register.
- Rewrote `README.md` from English explanations into Chinese explanations while preserving file names, commands, parameters, and mathematical notation. This makes the project entry document easier to use for advisor communication and later thesis/proposal writing.
- Adjusted the communication documents so key formulas use Markdown/HTML-compatible centered notation instead of raw LaTeX delimiters or plain code blocks. This improves readability in editors that do not render MathJax.

## 2026-05-23

- Implemented a state-vector Grover minimum-finding simulator in `qubit_value_function/grover_minimum.py`.
- The simulator uses the current value-register oracle abstraction: in each round it marks feasible states whose predicted value is below the incumbent predicted value, applies Grover amplitude amplification on the x register, samples a candidate, and updates the incumbent only when the exact UC/SCUC value is lower.
- Added `experiments/stage1_case14_t3_grover_minimum_finding.py` for the fixed-load `case14` T=3 experiment. The script retrains the 380-feature, 32-piece boundary-aware max-affine surrogate, then compares three oracle variants: exact true-value oracle, floating max-affine surrogate oracle, and 20-bit quantized max-affine value-register oracle.
- Added tests for the state-vector threshold Grover distribution and the threshold-iterated minimum-finding loop in `tests/test_stage1.py`.
- T=3 Grover minimum-finding result with 32 trials, 20 rounds, and seed 7: all three oracle variants found the exhaustive global optimum in all trials.
- Exact-value oracle summary: success 32/32, mean oracle calls about 639.16, median oracle calls about 657.5, mean rounds about 8.97.
- Floating max-affine surrogate oracle summary: success 32/32, mean oracle calls about 617.66, median oracle calls about 637.5, mean rounds about 8.84.
- 20-bit quantized max-affine oracle summary: success 32/32, mean oracle calls about 653.94, median oracle calls about 663.5, mean rounds about 8.94.
- The exhaustive T=3 optimum remains generator-major bitstring `111000111111111000`, time-major bitstring `101110101110101110`, total cost 27985.8598204.
- Result file written to `results/stage1_case14_t3_grover_minimum_finding.json`.
- Verification: `python -m pytest -q` passes with 28 tests.
- Current implication: research content 1 now has an end-to-end small-scale classical simulation of Grover minimum finding under a fixed load curve d. It connects the reversible max-affine value-register oracle to an actual threshold-iterated minimum-search workflow and validates the returned commitment against exhaustive enumeration.

## 2026-06-02

- Added `qubit_value_function/gate_level_oracle.py` as the first real gate-level value-register oracle prototype. It uses Qiskit's `WeightedAdder` to compute an integer affine proxy value, `IntegerComparator` to mark `value <= tau`, a `Z` phase on the comparator flag, then inverse comparator and inverse adder to uncompute all non-x registers.
- Added `experiments/stage1_case14_t2_gate_level_grover_oracle.py` for a T=2-only gate-level simulation. The experiment intentionally uses a selected case14 subregister with generators g1/g2/g3 over two periods, giving 6 x qubits, rather than attempting the full 12-bit, 207-feature, 32-piece T=2 max-affine surrogate in one statevector circuit.
- The T=2 proxy oracle marks the selected generator-major bitstring `110011`, corresponding to g1 on, g2 off, and g3 on across both periods. Non-selected generators are fixed to the exhaustive T=2 optimum only for embedded-subspace validation, not for the gate-level proxy computation itself.
- Gate-level oracle check: the phase oracle marks 1 state, returns auxiliary registers to zero with probability about 1.0, and has max phase error about 4.44e-16 on the uniform x superposition.
- Gate-level Grover result: with 6 Grover iterations on the 6-qubit selected T=2 subregister, the marked selected bitstring `110011` reaches probability about 0.9966, and the auxiliary-zero probability remains about 1.0 after compute-phase-uncompute.
- Embedded UC validation: the marked selected bitstring embeds back to the full T=2 bitstring `110011111100`, matching the exhaustive T=2 optimum with true cost 20578.2152604.
- Resource snapshot for this first gate-level prototype: the phase oracle and Grover circuit both use 18 qubits in the Qiskit construction; the six-iteration Grover circuit has high-level depth 85 before deeper basis-gate decomposition.
- Added tests for the gate-level oracle: auxiliary uncomputation, Grover amplification on a T=2 toy pattern, resource summary, and the case14 T=2 selected-subregister proxy marking.
- Result file written to `results/stage1_case14_t2_gate_level_grover_oracle.json`.
- Verification: `python -m pytest -q` passes with 32 tests. Qiskit emits deprecation warnings for its current `WeightedAdder`/`IntegerComparator` BlueprintCircuit base class, but the tests and experiment run successfully.
- Current implication: research content 1 now has a genuine T=2 quantum-circuit simulator artifact, not only a classical state-space oracle abstraction. The result is deliberately scoped as a selected-subregister gate-level prototype; the full T=2 max-affine value-register oracle still needs resource-aware synthesis or staged compression before it can be simulated at gate level.
- Extended `qubit_value_function/gate_level_oracle.py` from a single affine value-register oracle to a threshold-equivalent max-affine oracle. The new structure computes each integer affine piece `L_r(x)`, compares every piece to the same threshold `tau`, applies a multi-controlled phase on all comparison flags, then uncomputes the comparators and adders. This uses the equivalence `max_r L_r(x) <= tau` iff all `L_r(x) <= tau`.
- Added `experiments/stage1_case14_t2_gate_level_max_affine_oracle.py` as the first two-piece max-affine gate-level T=2 prototype. It uses a selected case14 subregister with generators g1/g2 over two periods, giving 4 x qubits and 2 affine pieces.
- Two-piece max-affine definitions in the prototype: `L0 = (1-g1_t0) + (1-g1_t1) + g2_t0`, `L1 = g2_t0 + g2_t1`, with `tau = 0`. The max-affine threshold marks only selected bitstring `1100`, i.e. g1 on and g2 off across both periods.
- Gate-level max-affine oracle check: the phase oracle marks 1 state, returns auxiliary registers to zero with probability about 1.0, and has max phase error about 4.44e-16.
- Gate-level max-affine Grover result: with 3 Grover iterations on the 4-qubit selected T=2 subregister, the marked selected bitstring `1100` reaches probability about 0.9613, and the auxiliary-zero probability remains about 1.0.
- Embedded UC validation: the marked selected bitstring embeds back to the full T=2 bitstring `110011111100`, matching the exhaustive T=2 optimum with true cost 20578.2152604.
- Resource snapshot for the two-piece max-affine prototype: the phase oracle and Grover circuit both use 14 qubits in the Qiskit construction; the three-iteration Grover circuit has high-level depth 56.
- Added tests for max-affine gate-level behavior: piece-intersection marking, auxiliary uncomputation, Grover amplification, resource counts for two piece blocks, and the case14 T=2 selected-subregister max-affine marking.
- Result file written to `results/stage1_case14_t2_gate_level_max_affine_oracle.json`.
- Verification: `python -m pytest -q` passes with 35 tests. Qiskit still emits deprecation warnings for `WeightedAdder`/`IntegerComparator`, but the new max-affine gate-level experiment and tests run successfully.
- Current implication: the project now has a real gate-level max-affine threshold oracle prototype. It is still a small T=2 selected-subregister demonstration, not the full 12-bit, 207-feature, 32-piece T=2 max-affine synthesis; the next step is to increase the feature/piece realism while tracking qubits and depth.

## 2026-06-28

- Added a true-cost-learned small max-affine gate-level experiment for T=2: `experiments/stage1_case14_t2_learned_small_max_affine_gate_level_oracle.py`.
- The experiment keeps the fixed-load `case14` T=2 setting and uses a selected subregister with generators g1 and g6 over two periods, giving 4 Grover search qubits. Non-selected generators are fixed to the exhaustive T=2 optimum only for embedded-subspace validation.
- Replaced the previous hand-designed L0/L1 demonstration with a data-driven integer max-affine mismatch surrogate: first find the true best selected-subspace commitment, then flip each selected bit once, use the true UC cost increase as a local sensitivity, quantize those sensitivities to small integer weights, and build generator-wise affine pieces.
- Learned model in the g1/g6 subspace: optimum selected bitstring `1100`; single-bit true-cost gaps are about 109059.87, 91227.45, 6275.34, and 6369.99; quantized integer weights are `[7, 6, 1, 1]`.
- The resulting pieces are `L0 = 7*(1-g1_t0) + 6*(1-g1_t1)` and `L1 = g6_t0 + g6_t1`, with `V_hat_int(x) = max(L0, L1)`. This is not a true-cost lookup table; it encodes learned local sensitivities as reversible integer arithmetic.
- Top-1 gate-level oracle result: with threshold `tau = 0`, the oracle exactly marks selected bitstring `1100`, which embeds to the full T=2 optimum `110011111100` with true cost 20578.2152604. Grover reaches marked probability about 0.9613 after 3 iterations, with auxiliary-zero probability about 1.0 and max phase error about 4.44e-16.
- Relaxed low-cost-set oracle result: with calibrated integer threshold `tau = 1`, the same learned pieces exactly mark the top-3 selected-subspace commitments `1100`, `1110`, and `1101`, with true costs 20578.2152604, 26853.5596684, and 26948.2088920. Grover reaches marked-set probability about 0.9492 after 1 iteration.
- Resource snapshot for both target cases: the Qiskit phase oracle and Grover circuits use 21 qubits. The top-1 three-iteration Grover circuit has high-level depth 56; the top-3 one-iteration Grover circuit has high-level depth 20.
- Extended `GateLevelAffinePieceSpec` with an optional integer `bias` field so future learned affine pieces can represent `L_r(x)=b_r+theta_r^T x` more directly. Existing experiments keep `bias=0` and remain unchanged in behavior.
- Added tests for biased max-affine piece comparison and for the learned T=2 g1/g6 small max-affine model. Verification: `python -m pytest -q` passes with 37 tests. Qiskit still emits `WeightedAdder`/`IntegerComparator` deprecation warnings, but the circuits and tests run successfully.
- Current implication: the gate-level part of research content 1 is now less like a hand-written toy oracle and more like a true UC-data-driven proof of concept. It still deliberately remains a small T=2 selected-subregister demonstration, matching the near-term research strategy of maximizing correctness and Grover success probability before scaling the full 12-bit, 207-feature, 32-piece T=2 surrogate.
- Added an experiment-reporting rule to `CODEX_CONTEXT.md`: every new experiment must explicitly describe the designed circuit and oracle, including registers, value/surrogate formula, threshold condition, phase marking, uncomputation, diffuser/readout, reversibility check, and true UC/SCUC validation. The latest learned T=2 small max-affine gate-level result now includes an `oracle_and_circuit_explanation` section in its JSON output.

## 2026-06-30

- Added `experiments/stage1_case14_t2_batch_gate_level_small_subspaces.py` for batch T=2 small-subspace validation of the learned integer max-affine gate-level Grover oracle.
- The batch experiment screens all 15 two-generator subspaces of `case14` T=2. Each subspace uses 4 Grover search qubits in selected generator-major order and learns generator-wise integer max-affine mismatch pieces from true UC cost gaps.
- Screening result: all 15 two-generator subspaces exactly separate top-1 under the learned integer max-affine oracle; 3 subspaces exactly separate both top-1 and top-3. The top-1/top-3 exact subspaces are g1/g4, g1/g5, and g1/g6.
- Resource lesson: directly running statevector gate-level Grover on the top-1/top-3 exact subspaces g1/g4 and g1/g5 was too slow for the default batch because their weighted-adder resources are heavier. The default gate-level runner now screens top-1/top-3 but runs statevector Grover by default only for top-1 targets and prioritizes lower estimated qubit count.
- Completed full Qiskit gate-level statevector simulations for three low-resource top-1 subspaces: g1/g6, g4/g6, and g5/g6. Each uses 21 qubits, Grover depth 56, and marks selected bitstring `1100` with threshold `tau = 0`.
- Gate-level batch result: for all three simulated subspaces, the marked selected bitstring embeds back to the full T=2 optimum `110011111100` with true cost 20578.2152604. Grover marked probability is about 0.961319, and auxiliary-zero probability is about 1.0.
- The batch JSON result is written to `results/stage1_case14_t2_batch_gate_level_small_subspaces.json`.
- Generated an advisor-facing Markdown report at `reports/stage1_case14_t2_batch_gate_level_small_subspaces_report.md`. The report records the experiment goal, full T=2 reference optimum, screening table for all 15 subspaces, gate-level Grover result table, and per-subspace circuit/oracle explanations including registers, threshold condition, phase marking, compute-phase-uncompute sequence, reversibility, and true-cost validation.
- The report is written with UTF-8 BOM (`utf-8-sig`) so Chinese text opens correctly in Windows tools.
- Verification: `python -m pytest -q` passes with 37 tests. Qiskit still emits `WeightedAdder`/`IntegerComparator` deprecation warnings, but the gate-level scripts and tests run successfully.

## 2026-06-30 QFT Weighted-Sum Comparison

- Added `qubit_value_function/qft_weighted_sum_oracle.py`, implementing a QFT-based weighted-sum max-affine threshold oracle for the same `GateLevelMaxAffineOracleSpec` used by the current WeightedAdder route.
- The QFT implementation computes each integer affine piece by applying QFT to the value register, applying controlled phase rotations from each selected commitment bit according to its integer coefficient, applying inverse QFT to recover the computational-basis sum register, comparing with `IntegerComparator`, phase marking, and uncomputing.
- Added `experiments/stage1_case14_t2_qft_weighted_sum_oracle_comparison.py` to compare the current WeightedAdder arithmetic path against the QFT weighted-sum arithmetic path on the fixed-load `case14` T=2 g1/g6 subspace.
- Both implementations use the same learned integer max-affine surrogate: `L0 = 7*(1-g1_t0) + 6*(1-g1_t1)`, `L1 = g6_t0 + g6_t1`, and `V_hat_int(x) = max(L0, L1)`.
- Top-1 comparison with `tau = 0`: both implementations exactly mark selected bitstring `1100`, embed back to full T=2 optimum `110011111100`, and reach Grover marked probability about 0.961319. WeightedAdder uses 21 qubits with high-level Grover depth 56 and decomposed depth 309; QFT weighted-sum uses 16 qubits with high-level Grover depth 49 and decomposed depth 319.
- Top-3 comparison with `tau = 1`: both implementations exactly mark selected bitstrings `1100`, `1110`, and `1101`, with Grover marked-set probability about 0.949219. WeightedAdder uses 21 qubits with high-level Grover depth 20 and decomposed depth 103; QFT weighted-sum uses 16 qubits with high-level Grover depth 17 and decomposed depth 109.
- Auxiliary register checks remain clean for both paths: auxiliary-zero probability is about 1.0, and phase errors are on the order of 1e-15.
- Generated advisor-facing report `reports/stage1_case14_t2_qft_weighted_sum_oracle_comparison_report.md`, comparing oracle definitions, circuit sequences, qubit/depth resources, phase errors, Grover probabilities, and true UC cost validation.
- Added tests for QFT weighted-sum marking/Grover amplification and for the small-sample qubit-count comparison against WeightedAdder.
- Verification: `python -m pytest -q` passes with 39 tests. Qiskit emits deprecation warnings for `WeightedAdder`, `IntegerComparator`, and `QFT`, but all circuits and tests run successfully.
- Current implication: the project now has a direct answer to the paper-alignment question. WeightedAdder remains the clearer current mainline implementation, while QFT weighted-sum provides a paper-inspired arithmetic alternative that reduces qubit count on the small g1/g6 example but introduces QFT/IQFT and controlled phase rotations.
