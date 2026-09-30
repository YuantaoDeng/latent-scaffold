"""Print baseline training commands for one paper condition; never starts training.

These recreate endpoint trajectories. Follow-up response/update workers also
need captured states listed in docs/TRAINING.md; these commands alone do not produce
every downstream checkpoint format.
"""
import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
MODELS = {"resnet18": ("ResNet-18", "resnet18"),
          "deit_tiny": ("DeiT-Tiny", "deit_tiny_patch16_224"),
          "mobilenetv3_small": ("MobileNetV3-S", "mobilenetv3_small_100.lamb_in1k"),
          "mixer_b16": ("MLP-Mixer-B/16", "mixer_b16_224.goog_in21k_ft_in1k")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    a = p.parse_args()
    cfg = json.loads((ROOT / "configs/paper.json").read_text())
    label, model = MODELS[a.model]
    rank = next(x["rank"] for x in cfg["models"] if (x["model"], x["dataset"]) == (label, a.dataset))
    for role in ["source", "target"]:
        for order in cfg[role + "_orders"][a.dataset]:
            args = ["python", "run.py", "sequence_drift_experiment", "--model", model,
                    "--dataset", a.dataset, "--tasks", "5" if a.dataset == "cifar10" else "4",
                    "--classes-per-task", "2" if a.dataset == "cifar10" else "10",
                    "--order-seed", str(order), "--seeds", *map(str, cfg[role + "_seeds"][a.dataset]),
                    "--rank", str(rank), "--continual-method", "er", "--epochs", "5",
                    "--batch-size", "64", "--train-per-class", "450", "--probe-per-class", "50",
                    "--memory", "500", "--lr", "0.0003", "--pretrained", "--save-trajectory",
                    "--save-pre-future-checkpoint", "--output",
                    f"results/retrained/{a.model}/{a.dataset}/{role}_order{order}"]
            print(shlex.join(args))


if __name__ == "__main__":
    main()
