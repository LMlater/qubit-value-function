# Stage B QNN convergence-qualified calibration

The calibration grid is fit/validation-only: optimizer seeds 11, 23, and 47
at iteration budgets 10, 20, 40, and 80. A budget is eligible only when every
required seed completed, produced finite computable validation metrics, did
not report `optimizer_failed`, and converged. Saturation is report-only and
cannot make an ineligible budget selectable.

Regression QNNs select eligible candidates by validation MAE, validation
regret, lower budget, then frozen candidate order. The threshold-conditioned
QNN selects by validation F1, recall, precision, lower budget, then candidate
order. Cutoffs are selected from 0.4, 0.5, 0.6, and 0.7 on validation only.
No interpolation, extrapolation, or unseen-state metric is accepted by the
selection interface.

The formal frozen budget is 20 for the simplified expectation QNN, 20 for the
full-readout expectation QNN, and 40 for the threshold-conditioned QNN. The
former metric-only ranking (10/20/10) is retained only as a historical
diagnostic because its simplified and threshold candidates were not
convergence qualified.

The optimizer is L-BFGS-B with `ftol=1e-10`, `gtol=1e-6`, and `maxls=20`.
The simplified and full heads use regularization 0.5 and theta regularization
`1e-5`; the threshold QNN uses theta regularization `1e-4`.
