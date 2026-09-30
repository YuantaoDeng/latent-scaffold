# All-model directional interventions, frozen before new outcomes

Scope: the four main-paper visual models (ResNet-18, DeiT-Tiny,
MobileNetV3-Small, MLP-Mixer-B/16) on CIFAR-10 and CIFAR-100. Existing
early-content manipulations are a different experiment, not substitutes.

For each condition use the last three predefined target task orders and their
two seeds in results/all_model_followups/inventory.json. Use saved pre-final-task
states (verified recovery states for ResNet/CIFAR-100), fixed condition-specific
scaffold ranks and two fixed same-rank uniform random subspaces (seeds91001/91002).
No PCA comparisons. No rank/strength selection based on these outcomes.

Stage 1: complete directional suppression in the seven missing conditions,
reusing the already completed six-history ResNet/CIFAR-10 retention replication.
Normal training plus scaffold/random1/random2 at coefficients1,10,100; 60 branches
per condition, five epochs per branch. Existing ResNet results are marked reused,
not counted as newly trained or as precision-identical replications.

Stage 2: final-task feature-gradient enhancement in all eight conditions, normal
plus scaffold/random1/random2 at gains1.25,1.5,2.0; 60 branches per condition.
This is a NEW matched final-task design: the older ResNet enhancement began after
task2 and continued tasks3-5, so it must not be silently pooled with this study.
The failed fixed-motion pilot is not extended.

Suppression uses the original centered, projected displacement penalty on old
training-memory examples relative to the pre-final-task state. The same penalty
coefficient and rank do NOT guarantee equal suppressed energy. Enhancement uses
G+(gain-1)GP_scaffold, with the same added-gradient norm matching for random
controls as scaffold_enhancement.py. Actual cumulative drift is separately measured
on excluded first-task examples and training memory; gradient gains are not
asserted to equal realized drift gains.

All paired branches share initial weights, exact current/replay augmented input
hashes, task-boundary AdamW reset, lr0.0003, weight_decay0.05, batch32+32, memory500,
five epochs. All-class competition is restricted to classes actually learned.
MobileNet uses FP32; other models AMP with initial scale128. Penalty and observation
are FP32. TF32 is disabled to make cross-batch reference errors small (MobileNet
preflight otherwise exceeded the reference consistency check). Mixer uses activation
recomputation to preserve the original batch size. Numerical skipped updates fail
explicitly, never silently counted as completed.

The evaluation split is the excluded last50 training images/class, not the official
test split; all used training examples and memory are disjoint from it. Record every
seen task's accuracy/loss at epoch0 and each epoch1-5, first/old/new accuracy, penalty,
actual drift energy and data hashes. Epoch0 evaluates the expanded final-task class
competition and therefore is not a prior-task retention score under its old mask.

Primary suppression outcome: scaffold minus average random new-task acquisition
(accuracy averaged over five epochs), coefficient10. Companion: versus baseline
and achieved suppression. First-task final accuracy is secondary; report all tasks
and strengths, no claim that retention improves from an acquisition cost.
Primary enhancement: final first-task accuracy, gain1.5, scaffold minus mean random,
plus required scaffold-minus-baseline comparison; both positive intervals required
for a retention-improvement claim. All gains/other tasks exploratory.
Use paired hierarchical bootstrap (three orders, two seeds),20000 draws, nominal95%
intervals; epochs/examples are not independent histories. Analyze each condition
only once all six histories/60 branches complete. Do not choose only positive models.

One GPU worker under experiment_control.py; pause/resume at batch boundaries.
Resume model/Adam/scaler saved each epoch. Once a branch completes, retain its
per-example predictions/features and full accuracy curves, remove only its redundant
resume state. Avoid duplicating hundreds of large final model states; original source
checkpoints remain untouched. Stop may replay an unfinished epoch. Queue resumes
accuracy reconstruction after causal jobs, retrying incomplete jobs. Native process
startup failures stop the queue instead of skipping dozens of experiments.
