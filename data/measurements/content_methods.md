# Definitions for the two follow-up experiments

## Extra drift coverage after removing current PCA

Let `D` be the centered feature-change matrix on held-out first-task anchors, with shape `n × d`. Its rows correspond to anchors and columns to feature coordinates. Let `U` (`d × r`) contain the fixed orthonormal scaffold directions estimated from separate early-drift histories. Let `V` (`d × k`) contain the leading orthonormal representation PCA directions fitted on the test network's training anchors **at the start of the transition**. PCA does not use its future drift or held-out evaluation anchors.

Define

\[
P=VV^\top,\qquad R=D(I-P),\qquad Q=\operatorname{orth}((I-P)U).
\]

`R` is residual drift and `Q` is the part of the original scaffold outside the removed PCA space, orthonormalized. `Q` is determined entirely by `U` and the starting representation PCA; it is not fitted to residual drift. Its rank `q` remains equal to `r` in all evaluated conditions.

The primary residual coverage and its random-control gain are

\[
C_{\rm residual}=\frac{\|RQ\|_F^2}{\|R\|_F^2},\qquad
G_{\rm residual}=C_{\rm residual}-\frac{q}{d-k}.
\]

The second term is the exact expected coverage of a uniformly oriented rank-`q` subspace in the PCA complement. It is a same-rank uniform control, not a classifier-matched control. No Monte Carlo approximation is needed for its expectation.

The added coverage relative to the **original total drift** is also saved:

\[
C_{\rm additional}=\frac{\|DQ\|_F^2}{\|D\|_F^2}
=\frac{\|R\|_F^2}{\|D\|_F^2}C_{\rm residual}.
\]

Because `Q` is orthogonal to `V`, this equals the coverage increment from augmenting the PCA span with the scaffold. Reporting both normalizations prevents a small residual denominator from being mistaken for a large fraction of total drift. The CSV also retains the unreorthonormalized fixed-scaffold result and its attenuation-matched control.

Primary comparisons remove `k=r` PCA directions. All eight conditions additionally remove 8, 16 and 32 directions. End-of-transition PCA is saved separately as a retrospective sensitivity analysis. Each condition uses twelve existing test histories (six task orders, two runs per order). Coverage intervals use 10,000 paired hierarchical order/run resamples, conditional on the fitted scaffold, with no multiplicity adjustment.

## Early-content dependence with identical later tasks

Within a model, all runs start from the same original pretrained initialization. For each seed, train task A once and copy its complete state to branches X and Y:

- A: deer/frog, CIFAR-10 labels `[4,6]`.
- X task 2: truck/ship, labels `[9,8]`.
- Y task 2: horse/airplane, labels `[7,0]`.
- Common task 3: bird/automobile, labels `[2,1]`.
- Common task 4: cat/dog, labels `[3,5]`.

Each branch's scaffold pools six source-run 1-to-2 drift subspaces, using the leading rank-`r` eigenspace of their mean projection matrix. The rank is the existing manuscript rank for that model and is the same in X and Y. Six separate paired test seeds complete both four-task histories.

Let `U_X` and `U_Y` be the two fixed scaffold bases. Let `D_{h,t}` denote centered held-out-anchor drift during common task `t` in test branch `h`. All four cells are evaluated:

\[
C_{h\leftarrow s,t}=\frac{\|D_{h,t}U_s\|_F^2}{\|D_{h,t}\|_F^2},
\qquad h,s\in\{X,Y\},\quad t\in\{3,4\}.
\]

The two own-history advantages are

\[
\Delta_{X,t}=C_{X\leftarrow X,t}-C_{X\leftarrow Y,t},\qquad
\Delta_{Y,t}=C_{Y\leftarrow Y,t}-C_{Y\leftarrow X,t}.
\]

Their paired average is secondary to inspecting both contrasts. Positive average preference alone would not imply preference in both branches. Scaffold similarity is `||U_X^T U_Y||_F^2/r`, a normalized subspace overlap between zero and one.

Task 3 replays A only; task 4 replays A and task 3. Branch-specific second-task images are excluded from later replay. Both branches use all ten output classes during common tasks, and a fresh AdamW optimizer for each task. Their common-task augmented input hashes and batch/replay IDs must match exactly. This is a controlled replay protocol and differs from replaying all previous classes.

Each task uses five epochs, learning rate 0.0003 and weight decay 0.05, with batch size 64 (32 current plus 32 replay when replay is present). The shared first task has 75 updates; subsequent tasks have 145 each. Mixer uses activation recomputation. MobileNet uses full FP32 throughout after all failed/partial AMP attempts were excluded; the other models use AMP. See the per-model numerical policy and verification records.

Five hundred first-task training anchors fit the source directions; 100 held-out anchors evaluate test drift and are excluded from training and replay. **Estimation and test histories nevertheless share training content.** Separate seeds and held-out anchors do not constitute training-content-disjoint transfer.

Coverage intervals resample six paired test seeds 10,000 times, conditional on the fitted scaffolds. The separately reported source-resampling overlap range is descriptive; with six source runs, repeated-source pooling changes effective source diversity, so it is not presented as a calibrated confidence interval. There is one class partition per model. A history preference supports persistent effects of early content on later feature drift; it does not identify the scaffold as a unique causal memory store or separate the roles of weights, classifier state and normalization buffers.
