# CIFAR-100: three-task early-content experiments

Completed: 4/4 models. Each condition uses six paired estimation runs and six separate paired test runs. Existing manuscript ranks are retained.

| Model | Rank | Scaffold overlap | X own gain pp [95% CI] | Y own gain pp [95% CI] | Paired gain pp [95% CI] | Positive paired runs |
|---|---|---|---|---|---|---|
| ResNet-18 | 8 | 26.62% | +3.64 [+3.04, +4.27] | +5.36 [+4.93, +5.79] | +4.50 [+4.16, +4.84] | 6/6 |
| DeiT-Tiny | 20 | 47.21% | +15.98 [+15.17, +16.75] | +15.09 [+14.26, +15.84] | +15.53 [+14.81, +16.08] | 6/6 |
| MobileNetV3-S | 8 | 14.24% | +2.98 [+2.39, +3.61] | +4.59 [+4.24, +5.03] | +3.79 [+3.45, +4.17] | 6/6 |
| MLP-Mixer-B/16 | 8 | 58.92% | +3.65 [+3.17, +4.18] | +3.89 [+3.45, +4.41] | +3.77 [+3.50, +4.03] | 6/6 |

X: A -> BX -> C; Y: A -> BY -> C. Each task has ten classes and each history has thirty classes. A, BX, BY and C are disjoint blocks from original order 1103, fixed before the new experiments. Within a paired seed A is trained once and its complete state is copied to X/Y. Task C replays A only and both branches compete over the same union of forty classes. Common augmented input hashes and image IDs match exactly. Every condition retains five epochs, 450 training images per class, AdamW reset per task, batch64 and memory500.

Source 1-to-2 drift on 500 training anchors fits the scaffold. Subsequent drift is measured on 500 held-out anchors excluded from training and replay. Estimation/test histories share training content; this is not content-disjoint transfer. Paired bootstrap intervals use 10,000 test-seed draws, conditional on fitted scaffolds, without multiplicity adjustment. Similarity resampling ranges in per-model files are descriptive. Both branch contrasts must be inspected; a positive paired average alone is insufficient for bilateral preference.

MobileNet uses FP32, consistent with the earlier CIFAR-10 follow-up. DeiT, ResNet and Mixer use AMP, with every planned update required. Mixer uses activation recomputation. Different model ranks and pretrained initializations preclude interpreting numerical differences as isolated architecture effects.

One fixed class partition and one common later task are evaluated. The experiment tests whether early-content effects persist into the next common task; it cannot establish longer-term persistence, decay, or exclusive causal storage in the scaffold.

Per-model reports contain full four-cell coverage matrices, per-seed data, task labels, initialization provenance and verification files.
