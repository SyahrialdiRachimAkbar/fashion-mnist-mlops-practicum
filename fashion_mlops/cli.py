from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys

from .core import ROOT, validate_config


def doctor() -> int:
    names = ["torch", "torchvision", "mlflow", "optuna", "dvc", "numpy", "sklearn", "matplotlib", "yaml"]
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    result = {"python": platform.python_version(), "executable": sys.executable, "missing": missing, "ready": not missing}
    print(json.dumps(result, indent=2))
    if missing:
        print("Jalankan scripts/setup.ps1 di PowerShell yang memiliki akses internet.", file=sys.stderr)
    return int(bool(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Praktikum CNN Fashion-MNIST: MLflow, Optuna, DVC")
    parser.add_argument("command", choices=["doctor", "prepare-data", "smoke", "baseline", "tune", "seeds", "ablation", "evaluate", "register", "predict", "report", "status", "all"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(doctor())
    try:
        import yaml
        from pathlib import Path
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        validate_config(config)
        if args.command == "status":
            from .reporting import status
            print(json.dumps(status(config), indent=2))
            return
        if args.command == "report":
            from .reporting import report
            print(report(config))
            return
        if doctor():
            raise SystemExit(1)
        from .data import fingerprint, prepare
        if args.command == "prepare-data":
            print(json.dumps(prepare(config), indent=2))
            return
        from filelock import FileLock
        (ROOT / "outputs").mkdir(exist_ok=True)
        with FileLock(str(ROOT / "outputs" / "pipeline.lock"), timeout=0):
            version = fingerprint()
            from . import workflow
            if args.command == "smoke":
                from .learning import train_run
                trial_config = {**config, "experiment_name": config["experiment_name"] + "-smoke"}
                workflow.connect(trial_config)
                result = train_run(trial_config, version, {**config["training"], "epochs": 1}, "smoke", smoke=True)
            elif args.command == "all":
                workflow.baseline(config, version)
                workflow.tune(config, version)
                workflow.repeated_runs(config, version)
                workflow.repeated_runs(config, version, ablation=True)
                workflow.final_evaluation(config, version)
                workflow.register(config, version)
                workflow.predict(config, version)
                from .reporting import report
                result = report(config)
            else:
                actions = {
                    "baseline": workflow.baseline,
                    "tune": workflow.tune,
                    "seeds": workflow.repeated_runs,
                    "ablation": lambda c, v: workflow.repeated_runs(c, v, ablation=True),
                    "evaluate": workflow.final_evaluation,
                    "register": workflow.register,
                    "predict": workflow.predict,
                }
                result = actions[args.command](config, version)
            print(json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
