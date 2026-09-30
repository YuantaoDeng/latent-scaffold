# Experiments and saved measurements

Paths in the measurement column are relative to `data/measurements/`.
The records are provided as reference data; this release does not include the
standalone figure builders or frozen-result verification runner.

| Experiment | Worker or shared module in `scripts/` | Measurements |
| --- | --- | --- |
| Cross-history recurrence | `sequence_drift_experiment.py`, `resnet_temporal_dimension.py`, `final_rank_experiments.py` | `baseline.json` |
| Rank selection | `recompute_deit_cifar10_current_rank.py`, `deit_pca_dimension.py`, `additional_rank_models.py` | `rank_snapshot.json`, `rank_bounds_snapshot.json` |
| Paired early content | `all_model_content_worker.py`, `resnet_cifar100_content.py`, `run_remaining_cifar100_content.py` | `content_c10.csv`, `content_c100.csv`, `raw/` |
| Local responses and readouts | `resnet_temporal_dimension.py`, `deit_rank20_existing.py`, `mlp_rnn_extension.py` | `baseline.json` |
| AdamW updates | `run_all_multistep_updates.py`, `pilot_multistep_updates.py`, `mixer_update_measurement.py` | `updates.csv`, `raw/` |
| No replay | `run_noreplay_all.py`, `noreplay_all_worker.py` | `noreplay_motion.csv`, `noreplay_updates.csv` |
| Fixed suppression | `all_model_causal.py`, `resnet_cifar10_unified_main.py` | `suppression_curves.csv`, `suppression_raw/` |
| Energy matching | `relative90_worker.py`, four `*_lambda100_energy_matched_queue.py` files | `energy_matched/` |
| Task accuracy | `accuracy_backfill.py` | `baseline_accuracy_cells.json`, `task_accuracy_matrix.csv` |

The current DeiT/CIFAR-10 rank is 8. Some historical code paths retain earlier
rank choices; use `recompute_deit_cifar10_current_rank.py` for the current paper
condition. The fixed suppression coefficients are 10 and 100; the main
comparison uses 100. Energy-matched random controls use selected coefficients.

Acquisition averages epochs 1–5; retention uses the final epoch. The two random
branches are averaged within each history before paired comparison.

See [TRAINING.md](TRAINING.md) for the states and inventories required by each
worker. Some shared modules have `analyze_` or `summarize_` in their names and
remain because experiment workers import or invoke them. The dependency graph
is recorded in `CODE_DEPENDENCIES.json`.
