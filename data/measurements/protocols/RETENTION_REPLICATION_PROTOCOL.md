# Early-task retention replication, frozen 2026-09-13

The hypothesis is tested, not assumed: restricting scaffold-directed plasticity
during later learning impairs first-task retention more than restricting random
directions. Positive, null and opposite results will all be reported.

- Scope: pretrained ResNet-18, CIFAR-10, replay, same dataset across histories.
- Fixed scaffold: the existing rank-8 scaffold fitted on six estimation histories.
- New intervention histories: orders 2131, 2137, 2141, seeds 170 and 171.
  These histories have existing observational measurements but were not part of
  the earlier direction-suppression experiment. This is not a blind new dataset.
- Paired start: saved state immediately before task five. Classifier continues
  training. All conditions receive identical image/augmentation/replay streams
  within each history; fresh AdamW state, five epochs, 145 successful updates.
- Conditions: normal training and scaffold/random1/random2 suppression at
  coefficients 1, 10 and 100. Rank and the two fixed random bases are unchanged
  from the previous intervention. No coefficient selection from test accuracy.
- Primary: final first-task accuracy at coefficient 10, scaffold minus the mean
  of the two random conditions. Negative means scaffold suppression hurts more.
  Each run first contributes one paired contrast. Aggregate equal-weight order
  means; nominal 95% hierarchical order/run bootstrap, 20,000 draws. Six runs
  nested in only three orders give limited uncertainty resolution.
- Secondary: change from the common starting accuracy, first-task epoch-average
  accuracy, final and epoch-average new-task accuracy, all five task accuracies,
  coefficient sensitivity, inside/outside drift on first-task and old examples.
- Evaluation: fixed last 50 training-split examples per class, excluded from all
  training/replay. Predictions compete among all ten classes, without task ID.
- Numerical validity: FP32 penalty, AMP classification, zero skipped steps,
  finite states and common initial metrics. Do not silently omit failed branches.
- Limit: same coefficient is not equal realized suppression. This is a causal
  test of the specified training constraint, not selective destruction of a
  memory store. It does not test whether amplifying drift improves retention.
- Existing three intervention orders are discovery/context only; do not pool
  them into the held-out primary interval.

## Control and recovery

Use `.venv/Scripts/python.exe experiment_control.py pause`, `resume`, `stop`,
`status`, or `start`. Pause is checked before every data batch, retains in-memory
state and halts training updates. It can retain GPU memory. Stop exits at a batch
boundary. Resuming after stop restarts the unfinished epoch from its last saved
epoch checkpoint; deterministic epoch seeds reproduce its input stream. Completed
branches and evaluation jobs are skipped. The queue owns only its own workers.
No experiment daemon or scheduled task modifies the manuscript automatically.
