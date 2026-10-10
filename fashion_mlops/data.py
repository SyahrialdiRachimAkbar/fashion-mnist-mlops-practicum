from __future__ import annotations

import sys
from pathlib import Path

from .core import CLASSES, ROOT, command, file_hash, read_json, stratified_indices, write_json

DATA = ROOT / "data"


def prepare(config: dict) -> dict:
    import numpy as np
    from torchvision.datasets import FashionMNIST

    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["selection"] != config["dataset"]:
            raise ValueError("Dataset sudah disiapkan dengan konfigurasi lain. Gunakan proyek terpisah untuk versi lain.")
        return fingerprint()
    if not (ROOT / ".git").exists():
        command("git", "init")
    if not (ROOT / ".dvc").exists():
        command(sys.executable, "-m", "dvc", "init")
    training = FashionMNIST(str(DATA / "raw"), train=True, download=True)
    testing = FashionMNIST(str(DATA / "raw"), train=False, download=True)
    selection = config["dataset"]
    indices = stratified_indices(training.targets.tolist(), selection["train_size"], selection["val_size"], selection["split_seed"])
    return _save_prepared(config, training, testing, indices, np)


def _save_prepared(config: dict, training, testing, indices: dict, np) -> dict:
    processed = DATA / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    write_json(DATA / "split_indices.json", {**indices, "test_source": "official_test", "split_seed": config["dataset"]["split_seed"]})
    for name, selected in indices.items():
        np.savez_compressed(processed / f"{name}.npz", images=training.data[selected].numpy(), labels=training.targets[selected].numpy())
    np.savez_compressed(processed / "test.npz", images=testing.data.numpy(), labels=testing.targets.numpy())
    train_images = training.data[indices["train"]].numpy().astype(np.float32) / 255.0
    mean, std = float(train_images.mean()), float(train_images.std())
    write_json(DATA / "manifest.json", {
        "dataset": "Fashion-MNIST",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "selection": config["dataset"],
        "classes": CLASSES,
        "shape": [1, 28, 28],
        "normalization": {"mean": mean, "std": std, "fit_on": "train"},
        "counts": {"train": len(indices["train"]), "validation": len(indices["validation"]), "test": len(testing)},
        "sha256": {str(path.relative_to(DATA)).replace("\\", "/"): file_hash(path) for path in [DATA / "split_indices.json", *sorted(processed.glob("*.npz"))]},
    })
    command(sys.executable, "-m", "dvc", "add", "data")
    storage = ROOT / "storage" / "dvc"
    storage.mkdir(parents=True, exist_ok=True)
    command(sys.executable, "-m", "dvc", "remote", "add", "--local", "--force", "-d", "storage", str(storage))
    command(sys.executable, "-m", "dvc", "push")
    return fingerprint()


def fingerprint() -> dict:
    import yaml

    pointer = ROOT / "data.dvc"
    if not pointer.exists():
        raise RuntimeError("Dataset belum berversi DVC. Jalankan prepare-data dahulu.")
    manifest = read_json(DATA / "manifest.json")
    for relative, expected in manifest["sha256"].items():
        if file_hash(DATA / relative) != expected:
            raise RuntimeError(f"Data berubah: {relative}. Jangan melanjutkan eksperimen dengan split berbeda.")
    if command(sys.executable, "-m", "dvc", "status", "--json", "data.dvc") not in ("{}", ""):
        raise RuntimeError("DVC mendeteksi perubahan dataset. Pulihkan versi yang sesuai sebelum training.")
    output = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
    return {
        "data_md5": output["md5"],
        "split_sha256": file_hash(DATA / "split_indices.json"),
        "manifest_sha256": file_hash(DATA / "manifest.json"),
        "dataset": "Fashion-MNIST",
    }


def load_split(name: str, *, allow_test: bool = False):
    import numpy as np

    if name == "test" and not allow_test:
        raise RuntimeError("Test set hanya boleh dibuka pada tahap evaluasi final.")
    if name not in ("train", "validation", "test"):
        raise ValueError(f"Split tidak dikenal: {name}")
    with np.load(DATA / "processed" / f"{name}.npz", allow_pickle=False) as arrays:
        return arrays["images"].copy(), arrays["labels"].copy()
