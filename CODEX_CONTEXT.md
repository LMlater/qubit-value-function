# Codex Shared Project Context

This file is the shared context for Codex conversations in this project. New
threads should read this file first before proposing research directions,
editing code, or interpreting papers.

## Project Goal

The project studies a Grover-compatible value-function oracle for fixed-load
UC/SCUC-style problems.

Current main line:

```text
fixed-load UC value function
-> structured / max-affine value surrogate
-> reversible value-register comparator oracle
-> Grover minimum finding
-> gate-level and resource analysis
```

The current research object is not a generic QUBO reformulation. The central
task is to build a reversible oracle that can be embedded in Grover-style
search.

## Current Priority

Continue the existing value-function / max-affine / Grover oracle route first.
Do not switch the project to reproducing another paper unless explicitly asked.

The Zheng et al. 2026 paper is being used for learning and methodological
comparison, not as an implementation target.

## Paper Being Studied

Reference:

```text
X. Zheng, D. Lin and H. Chen, "Grover Adaptive Search-Based Hybrid Benders
Decomposition for Mixed-Integer Linear Programs," IEEE Transactions on Quantum
Engineering, vol. 7, pp. 1-23, 2026, Art. no. 3102823.
DOI: 10.1109/TQE.2026.3681202
```

Local paper file supplied by the user:

```text
D:\鐮旂┒鐢焅闈掑勾鍩洪噾\Grover_Adaptive_Search-Based_Hybrid_Benders_Decomposition_for_Mixed-Integer_Linear_Programs.pdf
```

Project study notes:

```text
papers\zheng2026_gas_bd_study_notes.md
```

## Reading Policy For This Paper

This paper should be read to understand:

1. GAS-BD (Grover adaptive search-based Benders decomposition, 鍩轰簬 Grover 鑷€傚簲鎼滅储鐨?Benders 鍒嗚В).
2. GAS-MBO (Grover adaptive search for master Benders oracle, 鐢ㄤ簬 Benders 涓婚棶棰樼璋曠殑 Grover 鑷€傚簲鎼滅储).
3. Penalty-free oracle design (鏃犵綒椤圭璋曡璁?.
4. Direct encoding of Benders cuts (Benders 鍓茬殑鐩存帴缂栫爜).
5. Quantum resource accounting (閲忓瓙璧勬簮浼拌).

Do not treat the paper as something to reproduce directly. Its examples are
small: the UC case has two thermal units and one time interval; the FCKP case
has three binary activation variables and three continuous allocation variables;
the IBM ibm_torino experiment uses a reduced UC instance.

## Lessons Relevant To This Project

The paper is useful because it strengthens the interpretation of our current
route:

1. Our oracle is also penalty-free: it does not rely on QUBO penalty terms.
2. The max-affine surrogate has a natural connection to Benders value-function
   cuts:

```text
Q_hat(x) = max_r (alpha_r + beta_r^T x)
V_hat(x) = max_r (b_r + theta_r^T f(x))
```

3. Future work may compare learned max-affine pieces with true LP-dual Benders
   cuts, but this is not the immediate priority.
4. Resource analysis should track commitment qubits, feature/value registers,
   comparator ancilla, max-affine piece count, and Grover iterations.
5. Cut-pool management in the paper is analogous to limiting the number of
   max-affine pieces in our oracle.
6. Post-classical validation in GAS-BD is analogous to accepting Grover-sampled
   candidates only after checking the true UC value.

## Terminology Rule For Chinese Explanations

When explaining papers or methods in Chinese, keep important English terms and
add a Chinese translation the first time the term appears. Example:

```text
hybrid Benders锛堟贩鍚?Benders 鍒嗚В锛?master problem锛堜富闂锛?subproblem锛堝瓙闂锛?penalty-free oracle锛堟棤缃氶」绁炶皶锛?value-register comparator oracle锛堝€煎瘎瀛樺櫒姣旇緝鍣ㄧ璋曪級
```

After the first appearance, the English term can be used directly if the meaning
is clear.

## Instructions For Future Codex Threads

Before working on this project, read:

```text
CODEX_CONTEXT.md
RESEARCH_CONTENT_1_SUMMARY.md
WORK_LOG.md
```

For paper-specific discussion, also read:

```text
papers\zheng2026_gas_bd_study_notes.md
```

When uncertain, preserve the existing research route and ask whether the user
wants paper study, code implementation, experiment design, or writing support.


## Experiment Reporting Rule

Every new experiment must explicitly describe the designed circuit and oracle.
At minimum, record the search register, value/surrogate formula, threshold
condition, oracle phase-marking rule, auxiliary/value/comparator registers,
compute-phase-uncompute sequence, Grover diffuser/readout method, reversibility
check, and true UC/SCUC cost validation.
