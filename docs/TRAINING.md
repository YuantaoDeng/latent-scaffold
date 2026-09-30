# Training sources and prerequisites

Use Python 3.12 or newer with the pinned dependencies. Run all experiment
commands from the extracted package root, using `run.py`.
Most training workers explicitly use CUDA. Install a CUDA-compatible PyTorch
and torchvision pair for the machine, then the remaining dependencies.
`requirements.txt` records the versions observed during packaging;
`docs/ENVIRONMENT.json` also records the CUDA suffix. It is not a record of every
historical training environment.

## Data and initialization

`drift_experiment.IndexedCifar` uses the torchvision CIFAR-10/CIFAR-100 training
split under `data/`. Each selected class uses its first 450 examples for training
and last 50 for held-out evaluation. The experiment constructor uses
`download=False`; prepare the data explicitly before training:

```text
python prepare_data.py --dataset cifar10
python prepare_data.py --dataset cifar100
```

These commands download the original torchvision datasets into `data/` and
require internet access. The experiment scripts use the following timm names:

| Short name | timm model |
| --- | --- |
| `resnet18` | `resnet18` |
| `deit_tiny` | `deit_tiny_patch16_224` |
| `mobilenetv3_small` | `mobilenetv3_small_100.lamb_in1k` |
| `mixer_b16` | `mixer_b16_224.goog_in21k_ft_in1k` |

Some historical runners set `HF_HUB_OFFLINE=1` and expect a populated
`data/hf_cache`. Populate pretrained weights before using those runners, or set
`HF_HUB_OFFLINE=0` where a runner respects an existing environment value.
Inspect `drift_experiment.model` for the original initialization behavior.
No dataset or pretrained-weight redistribution is included.

## Baseline trajectories

```text
python baseline_commands.py --model resnet18 --dataset cifar10
python baseline_commands.py --model deit_tiny --dataset cifar100
```

These print nine commands per condition: three estimation orders and six test
orders, with two runs per order. They invoke the existing baseline trainer with
the paper configuration and write new endpoint trajectories under
`results/retrained/`. They do not overwrite frozen reference configurations.
The other two model choices are `mobilenetv3_small` and `mixer_b16`.
For Mixer, the paper's model-specific runner enables activation recomputation;
the generic endpoint command does not enable it and can require more GPU memory.
Use `mlp_rnn_extension.py` for the original captured-state Mixer workflow.

The original geometry/response/update pipeline additionally captures states at
specific checkpoints and actual optimizer steps. Its original paths and capture
logic are preserved in `resnet_temporal_dimension.py`, `deit_cifar10_dimension.py`,
`deit_three_claims.py`, `additional_rank_models.py`, `mlp_rnn_extension.py`, and
`final_rank_experiments.py`. A baseline endpoint file alone is insufficient for
these follow-ups.

## Follow-up entry points

The commands below are entry-point examples **after their prerequisite states
have been produced/restored in the original relative layout**. They are not a
fresh-machine end-to-end command sequence.

| Experiment | Entry point | Required products |
| --- | --- | --- |
| ResNet response geometry | `python run.py resnet_temporal_dimension responses --dataset cifar10` | Estimation manifests, recovered checkpoint states and fitting anchors |
| Current DeiT/C10 rank | `python run.py recompute_deit_cifar10_current_rank` | Estimation folds, trajectory/covariance records, captured target updates |
| MobileNet full condition | `python run.py additional_rank_models --model mobilenetv3_small_100.lamb_in1k --dataset cifar10` | Reference manifests/inventory, CIFAR data and initialization |
| Mixer full condition | `python run.py mlp_rnn_extension --model mixer_b16_224.goog_in21k_ft_in1k --dataset cifar10` | Reference manifests/inventory, CIFAR data and initialization |
| Paired CIFAR-10 content | `python run.py all_model_content_worker deit_tiny fit 220` | `all_model_followups_config.conditions()` inventory, baseline initialization and fitted scaffold metadata |
| Paired CIFAR-100 content | `python run.py resnet_cifar100_content`, then `python run.py run_remaining_cifar100_content` | Same baseline inventory; fixed A/BX/BY/C partition in the source |
| Multi-position updates | `python run.py run_all_multistep_updates` | Twelve pre-final-task trajectory records and `predictors.pt` per condition |
| No replay | `python run.py run_noreplay_all` | Baseline initialization, replay scaffold paths and no-replay protocol |
| Fixed suppression | `python run.py all_model_causal --case 0 --mode suppress --strength 100` | Pre-final-task states, source manifests and fitted scaffold for each case |
| Energy matching | `python run.py cifar10_lambda100_energy_matched_queue --worker` (other three queues analogous) | Ordinary/scaffold continuation records, memory features and basis calibration products |
| Accuracy extraction | `python run.py accuracy_backfill --inventory`, then its `--job` and `--collect` modes | Saved features/logits or checkpoint states for each history |

Use the worker's `--help` where available; several legacy files execute work
directly and do not implement `--help`. Energy-matching queues retain their
original pause/resume and duty-cycle behavior. Run one GPU worker at a time.

For fixed suppression, case indices follow
`all_model_causal.cases()` and `configs/experiments/multistep_update_all/protocol.json`;
do not assume case zero identifies a particular model. The ResNet/CIFAR-10
paper records came from the unified-worker rerun represented by
`resnet_cifar10_unified_main.py`, not the older isolated pilot.

## Inputs intentionally absent

- Dataset images and cached pretrained weights.
- `states/checkpoint*.pt` and full recovered `sources/artifacts/*.pt` /
  `targets/artifacts/*.pt` records, including optimizer-update captures.
- Fitted tensor artifacts such as `predictors.pt`, `active_bases.pt`, and
  initial model states used by follow-up workers.
- Large no-replay checkpoint chains. `lossless_checkpoint.py` is included to
  read the original lossless format when those chains are supplied.

These are runtime data products, not omitted Python dependencies. Bundled JSON
configurations retain their intended relative paths. Supplying the original
checkpoint tree or reconstructing these products is required to rerun all
experiments. The release includes experiment sources and lightweight measurements; checkpoint-dependent workers need these additional inputs.

## Validation scope

Packaging checks cover archive CRC, file hashes, Python syntax, local module
dependencies, and the command-listing helpers. Full GPU training has not been
rerun for this release. Standalone frozen-result analysis, plotting, and
verification programs are not distributed in this package.

## Input and output paths

Reference protocols, inventories, and per-order manifests are distributed under
`configs/experiments/`. Their artifact paths still identify runtime products in
`results/`; checkpoints are not included. `scripts/package_paths.py` resolves
configuration reads through `configs/paths.json` only when a run-specific file
is absent. It does not redirect writes or supply missing checkpoint tensors.

Place restored runtime artifacts under their original `results/` subpaths.
New runs also write there. The release itself contains neither a `results/`
input directory nor an `output/pdf/` directory. The baseline snapshot used by
multi-step updates is `data/measurements/baseline.json`.
