from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, cwd: Path = ROOT) -> str:
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}\n{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def git_context() -> dict[str, str]:
    commit = command("git", "rev-parse", "HEAD")
    return {
        "git_commit": commit,
        "git_branch": command("git", "branch", "--show-current"),
        "git_dirty": str(bool(command("git", "status", "--porcelain", "--untracked-files=no"))).lower(),
    }


def stratified_indices(labels: list[int], train_size: int, val_size: int, seed: int) -> dict[str, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    if set(groups) != set(range(10)):
        raise ValueError("Fashion-MNIST harus memiliki label 0 sampai 9.")
    if train_size <= 0 or val_size <= 0 or train_size % 10 or val_size % 10:
        raise ValueError("Ukuran train dan validation harus positif dan kelipatan 10.")
    rng = random.Random(seed)
    train, val = [], []
    for label in range(10):
        indices = groups[label][:]
        rng.shuffle(indices)
        n_train, n_val = train_size // 10, val_size // 10
        if n_train + n_val > len(indices):
            raise ValueError(f"Data kelas {label} tidak cukup untuk split yang diminta.")
        train.extend(indices[:n_train])
        val.extend(indices[n_train:n_train + n_val])
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "validation": val}


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Label dan prediksi harus memiliki panjang sama dan tidak kosong.")
    if any(value not in range(10) for value in targets + predictions):
        raise ValueError("Label dan prediksi harus berada dalam rentang 0 sampai 9.")
    matrix = [[0] * 10 for _ in range(10)]
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1
    f1 = []
    for label in range(10):
        tp = matrix[label][label]
        fp = sum(row[label] for row in matrix) - tp
        fn = sum(matrix[label]) - tp
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": sum(matrix[i][i] for i in range(10)) / len(targets),
        "f1_macro": statistics.mean(f1),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Mean ± sample std memerlukan minimal dua nilai finite.")
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)}


def paired_deltas(full: list[dict], ablated: list[dict], metric: str) -> dict:
    reference = {row["seed"]: row[metric] for row in full}
    comparison = {row["seed"]: row[metric] for row in ablated}
    if len(reference) != len(full) or len(comparison) != len(ablated) or reference.keys() != comparison.keys():
        raise ValueError("Ablation memerlukan pasangan seed unik yang sama dengan full model.")
    deltas = [reference[seed] - comparison[seed] for seed in sorted(reference)]
    return {**mean_std(deltas), "direction": "full_minus_ablation", "by_seed": dict(zip(sorted(reference), deltas))}


def best_candidate(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "FINISHED"]
    if not completed:
        raise ValueError("Tidak ada run selesai yang dapat dipilih.")
    if any(not math.isfinite(row["best_val_f1_macro"]) for row in completed):
        raise ValueError("Metrik seleksi harus finite.")
    # Run ID fixes ties without consulting the test set or wall-clock speed.
    return sorted(completed, key=lambda r: (-r["best_val_f1_macro"], r["run_id"]))[0]


def validate_config(config: dict) -> None:
    dataset, training, tuning = config["dataset"], config["training"], config["tuning"]
    for name in ("train_size", "val_size"):
        if dataset[name] <= 0 or dataset[name] % 10:
            raise ValueError(f"dataset.{name} harus positif dan kelipatan 10.")
    if dataset["train_size"] + dataset["val_size"] > 60000:
        raise ValueError("Subset train + validation melebihi data training resmi.")
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["lr"] <= 0:
        raise ValueError("Epoch, batch size, dan learning rate harus positif.")
    if not 0 <= training["dropout"] < 1 or training["num_workers"] != 0:
        raise ValueError("Dropout harus dalam [0, 1); gunakan num_workers=0 di Windows.")
    if training["cpu_threads"] < 1 or training["weight_decay"] < 0:
        raise ValueError("Jumlah thread harus positif dan weight decay tidak negatif.")
    if tuning["completed_trials"] < 20:
        raise ValueError("Tugas mewajibkan sedikitnya 20 trial tuning selesai.")
    if not 0 < tuning["lr_min"] <= tuning["lr_max"]:
        raise ValueError("Rentang learning rate tidak valid.")
    if not tuning["dropouts"] or not all(0 < p < 1 for p in tuning["dropouts"]):
        raise ValueError("Dropout tuning harus >0 agar ablation dropout memiliki komponen yang dapat dihapus.")
    if not tuning["batch_sizes"] or any(size < 1 for size in tuning["batch_sizes"]):
        raise ValueError("Batch size tuning harus positif.")
    if not training["augmentation"]:
        raise ValueError("Augmentasi full model harus aktif untuk ablation yang valid.")
    if len(config["seeds"]) != 3 or len(set(config["seeds"])) != 3:
        raise ValueError("Praktikum ini menggunakan tepat tiga seed unik.")
