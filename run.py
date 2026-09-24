from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("cifar10", "cifar100", "stl10", "tinyimagenet")
STORE_TRUE = {
    "adam_amsgrad",
    "assignment_epoch_global",
    "cudnn_benchmark",
    "detach_fm_encoder",
    "eval_at_start",
    "eval_head",
    "gradient_centralization",
    "lambda_schedule_raw_epoch_semantics",
    "normalize_assignment_consensus",
    "velocity_tangent_projection",
}
SPECIAL = {"Kprime": "--Kprime", "T0": "--T0", "Tmult": "--Tmult"}


def arguments(config: dict) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if key == "add_bn":
            if not value:
                args.append("--no-add-bn")
            continue
        option = SPECIAL.get(key, "--" + key.replace("_", "-"))
        if key in STORE_TRUE:
            if value:
                args.append(option)
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"Unsupported boolean option: {key}")
        if isinstance(value, list):
            if value:
                args.append(option)
                args.extend(map(str, value))
            continue
        args.extend((option, str(value)))
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--check", action="store_true")
    parsed = parser.parse_args()
    directory = ROOT / parsed.dataset
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    command = [sys.executable, "train_from_scratch_init.py", *arguments(config)]
    if parsed.check:
        print(" ".join(command))
        return
    subprocess.run(command, cwd=directory, check=True)


if __name__ == "__main__":
    main()
