from __future__ import annotations

import platform
import random
import time
from importlib import metadata

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .core import CLASSES, ROOT, classification_metrics, command, git_context, read_json, write_json
from .data import DATA, load_split


class FashionCNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 10))

    def forward(self, images):
        return self.classifier(self.features(images))


class Images(Dataset):
    def __init__(self, images, labels, normalization: dict, augmentation: bool = False):
        self.images = torch.from_numpy(images.copy()).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.mean, self.std = normalization["mean"], normalization["std"]
        self.augmentation = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip()]) if augmentation else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].float().div(255.0)
        if self.augmentation:
            image = self.augmentation(image)
        return (image - self.mean) / self.std, self.labels[index]


def set_seed(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_loader(split: str, params: dict, *, training: bool = False, allow_test: bool = False, limit: int | None = None) -> DataLoader:
    images, labels = load_split(split, allow_test=allow_test)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    normalization = read_json(DATA / "manifest.json")["normalization"]
    dataset = Images(images, labels, normalization, augmentation=training and params["augmentation"])
    generator = torch.Generator().manual_seed(params["seed"])
    return DataLoader(dataset, batch_size=params["batch_size"], shuffle=training, num_workers=0, generator=generator)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[dict, list[int], list[int]]:
    model.eval()
    targets, predictions, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).tolist())
    return {"loss": loss_sum / len(targets), **classification_metrics(targets, predictions)}, targets, predictions


def log_context(config: dict, data_version: dict, purpose: str) -> dict:
    import mlflow

    context = git_context()
    mlflow.set_tags({**context, "purpose": purpose, "dataset": "Fashion-MNIST", "device": "cpu", "optimizing_metric": "val_f1_macro"})
    mlflow.log_params(data_version)
    mlflow.log_params({"python_version": platform.python_version(), "torch_version": torch.__version__, "platform": platform.platform(), "architecture": "cnn-16-32-fc64"})
    folder = ROOT / "outputs" / "runs" / mlflow.active_run().info.run_id
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", config)
    environment = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions() if dist.metadata.get("Name"))
    (folder / "requirements-lock.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    (folder / "git.diff").write_text(command("git", "diff", "HEAD"), encoding="utf-8")
    for path in [folder / "config.json", folder / "requirements-lock.txt", folder / "git.diff", DATA / "manifest.json", DATA / "split_indices.json", ROOT / "data.dvc"]:
        mlflow.log_artifact(str(path), artifact_path="context")
    return context


def train_run(config: dict, data_version: dict, params: dict, purpose: str, *, smoke: bool = False, trial=None) -> dict:
    import mlflow
    import mlflow.pytorch
    from mlflow.models import infer_signature

    set_seed(params["seed"], params["cpu_threads"])
    name = f"{purpose}-seed{params['seed']}-lr{params['lr']:.3g}-drop{params['dropout']}"
    with mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None) as active:
        run_id = active.info.run_id
        if trial is not None:
            trial.set_user_attr("run_id", run_id)
            mlflow.set_tag("optuna_trial", str(trial.number))
        context = log_context(config, data_version, purpose)
        mlflow.log_params(params)
        train_loader = make_loader("train", params, training=True, limit=128 if smoke else None)
        val_loader = make_loader("validation", params, limit=64 if smoke else None)
        mlflow.log_params({"train_count": len(train_loader.dataset), "validation_count": len(val_loader.dataset), "smoke": smoke})
        model = FashionCNN(params["dropout"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        folder = ROOT / "outputs" / "runs" / run_id
        best, history = None, []
        started = time.perf_counter()
        for epoch in range(1, params["epochs"] + 1):
            model.train()
            targets, predictions, loss_sum = [], [], 0.0
            for images, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                targets.extend(labels.tolist())
                predictions.extend(logits.detach().argmax(dim=1).tolist())
            training = {"loss": loss_sum / len(targets), **classification_metrics(targets, predictions)}
            validation, _, _ = evaluate(model, val_loader)
            metrics = {**{f"train_{k}": v for k, v in training.items()}, **{f"val_{k}": v for k, v in validation.items()}, "lr": params["lr"]}
            mlflow.log_metrics(metrics, step=epoch)
            history.append({"epoch": epoch, **metrics})
            if best is None or validation["f1_macro"] > best["best_val_f1_macro"]:
                best = {"best_epoch": epoch, "best_val_f1_macro": validation["f1_macro"], "best_val_accuracy": validation["accuracy"], "best_val_loss": validation["loss"]}
                torch.save(model.state_dict(), folder / "best.pt")
            print(f"{purpose} | seed={params['seed']} | epoch={epoch}/{params['epochs']} | val_f1={validation['f1_macro']:.4f}", flush=True)
        duration = time.perf_counter() - started
        model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
        model.eval()
        sample = next(iter(val_loader))[0][:2].numpy()
        with torch.inference_mode():
            output = model(torch.from_numpy(sample)).numpy()
        model_info = mlflow.pytorch.log_model(
            model, name="model", signature=infer_signature(sample, output), input_example=sample,
            code_paths=[str(ROOT / "fashion_mlops")],
            pip_requirements=[f"torch=={metadata.version('torch')}", f"torchvision=={metadata.version('torchvision')}", f"numpy=={metadata.version('numpy')}"],
        )
        if smoke:
            loaded = mlflow.pytorch.load_model(model_info.model_uri, map_location="cpu")
            images = next(iter(val_loader))[0]
            with torch.inference_mode():
                for batch_size in (1, 10, len(images)):
                    torch.testing.assert_close(loaded(images[:batch_size]), model(images[:batch_size]), rtol=0, atol=1e-7)
            mlflow.set_tag("model_roundtrip_verified", "true")
        write_json(folder / "history.json", history)
        write_json(folder / "preprocessing.json", {"input": "normalized float32 NCHW", "raw_shape": [28, 28], "classes": CLASSES, **read_json(DATA / "manifest.json")["normalization"]})
        plot_history(history, folder / "learning_curves.png")
        for file in ["best.pt", "history.json", "learning_curves.png", "preprocessing.json"]:
            mlflow.log_artifact(str(folder / file))
        mlflow.log_metrics({**best, "training_duration_seconds": duration})
        result = {"run_id": run_id, "purpose": purpose, "seed": params["seed"], "params": params, **best, "training_duration_seconds": duration, "model_uri": model_info.model_uri, "status": "FINISHED", **data_version, **context}
    write_json(folder / "result.json", result)
    return result


def restore_checkpoint(row: dict) -> FashionCNN:
    model = FashionCNN(row["params"]["dropout"])
    model.load_state_dict(torch.load(ROOT / "outputs" / "runs" / row["run_id"] / "best.pt", map_location="cpu", weights_only=True))
    return model.eval()


def plot_history(history: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    for axis, metric, title in zip(axes, ["loss", "f1_macro"], ["Cross entropy", "Macro F1"]):
        for split, color in [("train", "#365C8D"), ("val", "#BB5627")]:
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=split, color=color)
        axis.set(xlabel="Epoch", ylabel=title, xticks=epochs)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)
