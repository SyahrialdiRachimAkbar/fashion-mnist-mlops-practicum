from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .core import ROOT, best_candidate, mean_std, paired_deltas, read_json, write_json

STATE = ROOT / "outputs" / "state.json"


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def connect(config: dict):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(config["tracking_uri"])
    client = MlflowClient()
    try:
        client.search_experiments(max_results=1)
    except Exception as error:
        raise RuntimeError("MLflow server tidak dapat dihubungi. Jalankan scripts/start_mlflow.ps1 pada terminal terpisah.") from error
    mlflow.set_experiment(config["experiment_name"])
    return client


def load_state(config: dict, data_version: dict) -> dict:
    expected = {"config_sha256": config_digest(config), **data_version}
    if STATE.exists():
        state = read_json(STATE)
        if any(state.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Config atau versi data berubah setelah eksperimen dimulai. Gunakan proyek baru untuk eksperimen berbeda.")
        return state
    state = {**expected, "baseline": None, "tuning_parent_id": None, "best_trial": None, "full": [], "ablation_no_augmentation": [], "ablation_no_dropout": [], "final_evaluation": None, "registry": None}
    write_json(STATE, state)
    return state


def assert_not_finalized(state: dict) -> None:
    if state["final_evaluation"] is not None or state.get("selection_lock"):
        raise RuntimeError("Seleksi model telah dikunci untuk evaluasi final. Training/tuning tambahan ditolak untuk menjaga test set.")


def baseline(config: dict, data_version: dict) -> dict:
    from .learning import train_run

    state = load_state(config, data_version)
    if state["baseline"]:
        return state["baseline"]
    assert_not_finalized(state)
    connect(config)
    state["baseline"] = train_run(config, data_version, config["training"].copy(), "baseline")
    write_json(STATE, state)
    return state["baseline"]


def tune(config: dict, data_version: dict) -> dict:
    import mlflow
    import optuna
    from .learning import log_context, train_run

    state = load_state(config, data_version)
    if state["best_trial"]:
        return state["best_trial"]
    assert_not_finalized(state)
    if not state["baseline"]:
        raise RuntimeError("Jalankan baseline sebelum tuning.")
    client = connect(config)
    if not state["tuning_parent_id"]:
        with mlflow.start_run(run_name="optuna-sweep") as parent:
            state["tuning_parent_id"] = parent.info.run_id
            log_context(config, data_version, "tuning_parent")
            mlflow.log_params(config["tuning"])
        write_json(STATE, state)
    study = optuna.create_study(
        study_name="fashion-cnn-" + state["config_sha256"][:12],
        storage="sqlite:///" + (ROOT / "optuna.db").as_posix(),
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["tuning"]["sampler_seed"]), load_if_exists=True,
    )
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.RUNNING:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            if trial.user_attrs.get("run_id"):
                client.set_terminated(trial.user_attrs["run_id"], status="FAILED")
    search = config["tuning"]

    def objective(trial):
        params = {
            **config["training"],
            "lr": trial.suggest_float("lr", search["lr_min"], search["lr_max"], log=True),
            "batch_size": trial.suggest_categorical("batch_size", search["batch_sizes"]),
            "dropout": trial.suggest_categorical("dropout", search["dropouts"]),
        }
        row = train_run(config, data_version, params, "tuning", trial=trial)
        trial.set_user_attr("result", row)
        client.set_tag(row["run_id"], "optuna_trial", str(trial.number))
        return row["best_val_f1_macro"]

    with mlflow.start_run(run_id=state["tuning_parent_id"]):
        completed = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
        mlflow.set_tag("resumed_study", str(bool(study.trials)).lower())
        if completed < search["completed_trials"]:
            study.optimize(objective, n_trials=search["completed_trials"] - completed, n_jobs=1)
        trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        rows = [trial.user_attrs["result"] for trial in trials]
        if len(rows) < search["completed_trials"] or any(client.get_run(row["run_id"]).info.status != "FINISHED" for row in rows):
            raise RuntimeError("Tuning belum memiliki 20 trial dan run MLflow yang selesai.")
        state["best_trial"] = best_candidate(rows)
        state["tuning_run_ids"] = [row["run_id"] for row in rows]
        mlflow.log_metrics({"completed_trials": len(rows), "best_val_f1_macro": state["best_trial"]["best_val_f1_macro"]})
        mlflow.log_params({f"best_{key}": state["best_trial"]["params"][key] for key in ("lr", "batch_size", "dropout")})
        write_json(ROOT / "outputs" / "tuning_results.json", rows)
        mlflow.log_artifact(str(ROOT / "outputs" / "tuning_results.json"))
    write_json(STATE, state)
    return state["best_trial"]


def repeated_runs(config: dict, data_version: dict, *, ablation: bool = False) -> dict:
    import mlflow
    from .learning import log_context, train_run

    state = load_state(config, data_version)
    groups = ["ablation_no_augmentation", "ablation_no_dropout"] if ablation else ["full"]
    if all(len(state[group]) == 3 for group in groups):
        return state
    assert_not_finalized(state)
    if not state["best_trial"] or (ablation and len(state["full"]) != 3):
        raise RuntimeError("Selesaikan tuning dan tiga seed full model sebelum ablation.")
    connect(config)
    with mlflow.start_run(run_name="ablation" if ablation else "best-config-three-seeds"):
        log_context(config, data_version, "ablation_parent" if ablation else "seeds_parent")
        for group in groups:
            existing = {row["seed"] for row in state[group]}
            for seed in config["seeds"]:
                if seed in existing:
                    continue
                params = {**state["best_trial"]["params"], "seed": seed}
                if group == "ablation_no_augmentation":
                    params["augmentation"] = False
                if group == "ablation_no_dropout":
                    params["dropout"] = 0.0
                state[group].append(train_run(config, data_version, params, group))
                write_json(STATE, state)
            for metric in ["best_val_f1_macro", "best_val_accuracy"]:
                summary = mean_std([row[metric] for row in state[group]])
                mlflow.log_metrics({f"{group}_{metric}_{key}": value for key, value in summary.items()})
            if ablation:
                delta = paired_deltas(state["full"], state[group], "best_val_f1_macro")
                mlflow.log_metrics({f"{group}_f1_delta_mean": delta["mean"], f"{group}_f1_delta_std": delta["std"]})
    return state


def final_evaluation(config: dict, data_version: dict) -> dict:
    import mlflow
    from .learning import evaluate, make_loader, restore_checkpoint, set_seed

    state = load_state(config, data_version)
    if state["final_evaluation"]:
        return state["final_evaluation"]
    if not state["baseline"] or not state["best_trial"] or any(len(state[group]) != 3 for group in ["full", "ablation_no_augmentation", "ablation_no_dropout"]):
        raise RuntimeError("Selesaikan baseline, 20 trial, tiga seed, dan kedua ablation sebelum membuka test set.")
    client = connect(config)
    selected = best_candidate(state["full"])
    expected_lock = {"champion_run_id": selected["run_id"], "full_run_ids": sorted(row["run_id"] for row in state["full"])}
    if state.get("selection_lock") and state["selection_lock"] != expected_lock:
        raise RuntimeError("Daftar kandidat berubah setelah test set dibuka.")
    state["selection_lock"] = expected_lock
    write_json(STATE, state)
    rows = []
    for row in state["full"]:
        output = ROOT / "outputs" / "runs" / row["run_id"]
        cached = output / "test_metrics.json"
        if cached.exists():
            metrics = read_json(cached)
        else:
            set_seed(row["seed"], row["params"]["cpu_threads"])
            model = restore_checkpoint(row)
            loader = make_loader("test", row["params"], allow_test=True)
            metrics, targets, predictions = evaluate(model, loader)
            write_json(output / "test_predictions.json", {"targets": targets, "predictions": predictions})
            plot_confusion(targets, predictions, output / "confusion_matrix.png")
            from sklearn.metrics import classification_report
            write_json(output / "classification_report.json", classification_report(targets, predictions, labels=list(range(10)), target_names=read_json(ROOT / "data" / "manifest.json")["classes"], output_dict=True, zero_division=0))
            write_json(cached, metrics)
        for key, value in metrics.items():
            client.log_metric(row["run_id"], "test_" + key, value)
        for name in ["test_metrics.json", "test_predictions.json", "confusion_matrix.png", "classification_report.json"]:
            client.log_artifact(row["run_id"], str(output / name), "final_evaluation")
        rows.append({"run_id": row["run_id"], "seed": row["seed"], **metrics})
    result = {"champion_run_id": selected["run_id"], "models": rows, "summary": {metric: mean_std([row[metric] for row in rows]) for metric in ("f1_macro", "accuracy", "loss")}, "selection_metric": "best_val_f1_macro", "test_used_for_selection": False}
    state["final_evaluation"] = result
    write_json(STATE, state)
    write_json(ROOT / "outputs" / "final_evaluation.json", result)
    return result


def register(config: dict, data_version: dict) -> dict:
    import mlflow
    import mlflow.pytorch
    import torch
    from .learning import make_loader, restore_checkpoint

    state = load_state(config, data_version)
    if state["registry"]:
        return state["registry"]
    if not state["final_evaluation"]:
        raise RuntimeError("Registrasi hanya dilakukan setelah evaluasi final selesai.")
    client = connect(config)
    row = best_candidate(state["full"])
    name = config["registered_model_name"]
    version = mlflow.register_model(row["model_uri"], name)
    client.update_registered_model(name, description="CNN Fashion-MNIST CPU. Seleksi memakai validation macro F1; test set dipakai setelah seleksi dikunci.")
    client.update_model_version(name, version.version, description=f"run={row['run_id']}; seed={row['seed']}; data_md5={data_version['data_md5']}")
    for key, value in {**data_version, "source_run_id": row["run_id"], "selection_metric": "best_val_f1_macro"}.items():
        client.set_model_version_tag(name, version.version, key, str(value))
    client.set_registered_model_alias(name, "champion", version.version)
    uri = f"models:/{name}@champion"
    loaded = mlflow.pytorch.load_model(uri, map_location="cpu")
    images, labels = next(iter(make_loader("validation", row["params"])))
    with torch.inference_mode():
        expected = restore_checkpoint(row)(images[:10])
        actual = loaded(images[:10])
    if not torch.allclose(expected, actual, rtol=0, atol=1e-7):
        raise RuntimeError("Output model registry berbeda dari checkpoint sumber.")
    state["registry"] = {"name": name, "version": str(version.version), "alias": "champion", "uri": uri, "run_id": row["run_id"], "checkpoint_match": True}
    write_json(STATE, state)
    client.set_tag(row["run_id"], "champion_alias_verified", "true")
    return state["registry"]


def predict(config: dict, data_version: dict, count: int = 10) -> list[dict]:
    import mlflow.pytorch
    import torch
    from .core import CLASSES
    from .learning import make_loader

    state = load_state(config, data_version)
    if not state["registry"]:
        raise RuntimeError("Model champion belum terdaftar.")
    connect(config)
    model = mlflow.pytorch.load_model(state["registry"]["uri"], map_location="cpu")
    images, labels = next(iter(make_loader("validation", best_candidate(state["full"])["params"])))
    with torch.inference_mode():
        probabilities = model(images[:count]).softmax(dim=1)
    rows = [{"validation_index": i, "actual": CLASSES[label], "predicted": CLASSES[int(prob.argmax())], "confidence": float(prob.max())} for i, (prob, label) in enumerate(zip(probabilities, labels[:count].tolist()))]
    write_json(ROOT / "outputs" / "champion_predictions.json", rows)
    return rows


def plot_confusion(targets: list[int], predictions: list[int], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from .core import CLASSES

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ConfusionMatrixDisplay.from_predictions(targets, predictions, labels=list(range(10)), display_labels=CLASSES, cmap="Blues", colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title("Test set resmi: jumlah prediksi per kelas")
    fig.savefig(path, dpi=160)
    plt.close(fig)
