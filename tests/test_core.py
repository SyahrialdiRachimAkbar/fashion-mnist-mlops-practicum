import math
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from fashion_mlops.core import best_candidate, classification_metrics, file_hash, mean_std, paired_deltas, read_json, stratified_indices, validate_config, write_json


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.labels = [label for label in range(10) for _ in range(30)]

    def test_split_is_balanced_disjoint_and_repeatable(self):
        split = stratified_indices(self.labels, 100, 50, 42)
        self.assertEqual(split, stratified_indices(self.labels, 100, 50, 42))
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(len(set(split["train"])), 100)
        self.assertEqual(Counter(self.labels[i] for i in split["train"]), dict.fromkeys(range(10), 10))
        self.assertEqual(Counter(self.labels[i] for i in split["validation"]), dict.fromkeys(range(10), 5))
        self.assertNotEqual(split, stratified_indices(self.labels, 100, 50, 123))

    def test_insufficient_class_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            stratified_indices(self.labels, 300, 10, 42)

    def test_missing_classes_are_rejected(self):
        with self.assertRaises(ValueError):
            stratified_indices([0, 1, 1], 10, 10, 42)

    def test_invalid_sizes_are_rejected(self):
        for train_size, val_size in [(0, 10), (-10, 10), (15, 10), (10, 0)]:
            with self.subTest(train_size=train_size, val_size=val_size), self.assertRaises(ValueError):
                stratified_indices(self.labels, train_size, val_size, 42)


class MetricTests(unittest.TestCase):
    def test_perfect_predictions_covering_all_classes(self):
        labels = list(range(10))
        self.assertEqual(classification_metrics(labels, labels), {"accuracy": 1.0, "f1_macro": 1.0})

    def test_macro_f1_uses_all_ten_known_classes(self):
        values = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(values["accuracy"], 0.75)
        self.assertAlmostEqual(values["f1_macro"], (2 / 3 + 4 / 5) / 10)

    def test_invalid_metrics_input_is_rejected(self):
        for labels, predictions in [([], []), ([0], []), ([10], [0]), ([0], [-1])]:
            with self.subTest(labels=labels, predictions=predictions), self.assertRaises(ValueError):
                classification_metrics(labels, predictions)

    def test_sample_std_uses_ddof_one(self):
        result = mean_std([0.7, 0.8, 0.9])
        self.assertAlmostEqual(result["mean"], 0.8)
        self.assertAlmostEqual(result["std"], 0.1)
        self.assertEqual(result["n"], 3)

    def test_insufficient_or_nonfinite_values_are_rejected(self):
        for values in [[], [0.5], [math.nan, 0.5], [math.inf, 0.5]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                mean_std(values)

    def test_ablation_matches_seeds_instead_of_row_order(self):
        full = [{"seed": 42, "f1": 0.8}, {"seed": 123, "f1": 0.9}, {"seed": 2026, "f1": 0.7}]
        ablated = [{"seed": 2026, "f1": 0.6}, {"seed": 42, "f1": 0.7}, {"seed": 123, "f1": 0.8}]
        result = paired_deltas(full, ablated, "f1")
        self.assertAlmostEqual(result["mean"], 0.1)
        self.assertAlmostEqual(result["std"], 0.0)

    def test_ablation_rejects_missing_and_duplicate_seeds(self):
        full = [{"seed": 42, "f1": 0.8}, {"seed": 123, "f1": 0.9}]
        for other in [[{"seed": 42, "f1": 0.8}], [{"seed": 42, "f1": 0.8}, {"seed": 42, "f1": 0.7}]]:
            with self.subTest(other=other), self.assertRaises(ValueError):
                paired_deltas(full, other, "f1")


class SelectionTests(unittest.TestCase):
    def test_selection_ignores_test_metrics_and_failed_runs(self):
        rows = [
            {"run_id": "a", "status": "FINISHED", "best_val_f1_macro": 0.8, "test_f1_macro": 1.0},
            {"run_id": "b", "status": "FINISHED", "best_val_f1_macro": 0.9, "test_f1_macro": 0.1},
            {"run_id": "c", "status": "FAILED", "best_val_f1_macro": 0.99},
        ]
        self.assertEqual(best_candidate(rows)["run_id"], "b")

    def test_tie_breaking_is_stable(self):
        rows = [{"run_id": rid, "status": "FINISHED", "best_val_f1_macro": 0.8} for rid in ["b", "a"]]
        self.assertEqual(best_candidate(rows)["run_id"], "a")

    def test_no_completed_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            best_candidate([])

    def test_invalid_selection_value_is_rejected(self):
        with self.assertRaises(ValueError):
            best_candidate([{"run_id": "a", "status": "FINISHED", "best_val_f1_macro": math.nan}])


class ArtifactTests(unittest.TestCase):
    def test_hash_detects_changed_content(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            write_json(path, {"values": [1, 2, 3]})
            initial = file_hash(path)
            self.assertEqual(read_json(path), {"values": [1, 2, 3]})
            write_json(path, {"values": [1, 2, 4]})
            self.assertNotEqual(initial, file_hash(path))

    def test_json_rejects_nonfinite_results(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                write_json(Path(folder) / "result.json", {"f1": math.nan})


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "dataset": {"train_size": 12000, "val_size": 3000, "split_seed": 42},
            "training": {"epochs": 5, "batch_size": 64, "lr": 0.001, "dropout": 0.3, "num_workers": 0, "cpu_threads": 4, "weight_decay": 0.0, "augmentation": True},
            "tuning": {"completed_trials": 20, "lr_min": 0.0001, "lr_max": 0.003, "dropouts": [0.2, 0.35, 0.5], "batch_sizes": [64, 128]},
            "seeds": [42, 123, 2026],
        }

    def test_default_rancangan_is_valid(self):
        validate_config(self.config)

    def test_fewer_than_twenty_trials_is_rejected(self):
        self.config["tuning"]["completed_trials"] = 19
        with self.assertRaises(ValueError):
            validate_config(self.config)

    def test_full_model_must_contain_components_for_ablation(self):
        self.config["tuning"]["dropouts"] = [0.0, 0.2]
        with self.assertRaises(ValueError):
            validate_config(self.config)
        self.config["tuning"]["dropouts"] = [0.2]
        self.config["training"]["augmentation"] = False
        with self.assertRaises(ValueError):
            validate_config(self.config)


if __name__ == "__main__":
    unittest.main()
